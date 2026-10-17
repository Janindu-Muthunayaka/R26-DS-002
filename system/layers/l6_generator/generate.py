"""
LAYER 6 — generator / RAG.  COMPONENT 3. OWNER: Nadee.

IN : the Document held in session, plus Layer 0's routing dict
OUT: {'ok', 'route', 'intent', 'speakable', 'answer_si', 'sources',
      'warnings', 'cursor'}

`speakable` is the only field the phone needs. The phone speaks `speakable`
whenever it is non-empty, whatever `ok` says. `ok` records whether an answer
was generated. A blind user must hear something, and "the answering service is
not available" is something. Silence is the one unacceptable outcome.

LOCAL INTENTS ("read that again", "next", "article 1", etc.) are handled
in-memory with zero network and zero LLM dependencies.

MODES:
  * `http`   — POST to RAG_URL/answer (services/rag).
  * `local`  — Run Component 3 upgraded pipeline in-process directly.
  * `off`    — Local intents work; generation gets an honest "not available".
"""
from __future__ import annotations

import os
from typing import Optional

from core import svc
from core.config import RAG_MODE, RAG_TIMEOUT_S, RAG_URL
from core.schemas import Document
from core.textutils import sentences
from layers.l5_assemble.payload import rag_payload, article_text, article_evidence

try:
    from .tts_clean import make_speakable
except (ImportError, ValueError):
    from tts_clean import make_speakable

# Re-export generate_answer from llm_generate for testing or script compatibility
try:
    from .llm_generate import generate_answer
except (ImportError, ValueError):
    try:
        from llm_generate import generate_answer
    except (ImportError, ValueError):
        generate_answer = None

# Intents answerable from what is already in session. No service, no network,
# no API key — the part of the conversation that cannot break on the day.
LOCAL_INTENTS = {
    'REPEAT', 'READ_ALOUD', 'REPLAY',    # the whole article again
    'NEXT', 'PREVIOUS', 'FIRST',         # walk through it in parts
    'LENGTH', 'TITLE', 'WARNINGS',       # ask about it
    'FIRST_SCAN',
    'ARTICLE_1', 'ARTICLE_2', 'ARTICLE_3', 'ARTICLE_4', 'ARTICLE_5'
}

# Change HOW it is spoken, never WHAT: no generation, ever.
DELIVERY_INTENTS = {'CHANGE_SPEED', 'CHANGE_DELIVERY', 'CHANGE_PAUSE', 'VOLUME'}

PART_MAX_CHARS = 400

SI_OK = 'හරි.'
SI_NO_SERVICE = 'පිළිතුරු සේවාව දැනට නොමැත.'
SI_FAILED = 'පිළිතුර ලබාගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න.'
SI_NOTHING_TO_REPEAT = 'නැවත කියවීමට කිසිවක් නොමැත.'
SI_END = 'ලිපිය අවසන්.'
SI_START = 'මෙය ලිපියේ ආරම්භයයි.'
SI_NO_TITLE = 'ශීර්ෂය තවම කියවා නොමැත.'
SI_NOTHING_MISSED = 'කිසිවක් මඟ හැරී නැත.'
SI_WORDS = 'මෙම ලිපියේ වචන {n} ක් ඇත.'
SI_MISSED = 'මඟ හැරුණු දේ: '


def _text_of(document: Document) -> str:
    """The clean article text as spoken locally."""
    bodies = [article_text(art) for art in document.articles if article_text(art)]
    return '\n\n'.join(bodies).strip()


def _parts(document) -> list:
    """The article split into speakable parts, never mid-sentence."""
    text = _text_of(document)
    if not text:
        return []
    out, buf = [], ''
    for s in sentences(text):
        s = s.strip()
        if not s:
            continue
        if buf and len(buf) + 1 + len(s) > PART_MAX_CHARS:
            out.append(buf)
            buf = s
        else:
            buf = f'{buf} {s}'.strip()
    if buf:
        out.append(buf)
    return out


def _out(ok, route, intent, speakable='', answer_si='',
         sources=None, warnings=None, cursor=None) -> dict:
    return {'ok': bool(ok), 'route': route, 'intent': intent,
            'speakable': speakable, 'answer_si': answer_si,
            'sources': list(sources or []), 'warnings': list(warnings or []),
            'cursor': cursor}


def _local(document, intent, cursor, warnings) -> dict:
    """Answer from the article in session. Returns the new cursor."""
    text = _text_of(document)

    if intent == 'FIRST_SCAN':
        n = len(document.articles)
        titles_with_no = []
        for idx, a in enumerate(document.articles):
            t = (a.title_polished or a.title or a.title_raw or '').strip()
            if t:
                titles_with_no.append(f"ලිපිය {idx + 1}: {t}")
        if n == 0:
            msg = "කිසිදු ලිපියක් හඳුනාගෙන නොමැත."
        else:
            msg = f"ලිපි {n} ක් හඳුනාගෙන ඇත."
            if titles_with_no:
                msg += " එම ලිපිවල ශීර්ෂයන් වන්නේ: " + ", ".join(titles_with_no)
            else:
                msg += " නමුත් ඒවායේ ශීර්ෂයන් හඳුනාගත නොහැක."
        return _out(True, 'LOCAL', intent, msg, warnings=warnings, cursor=cursor)

    if intent and intent.startswith('ARTICLE_'):
        try:
            art_num = int(intent.replace('ARTICLE_', ''))
            idx = art_num - 1
            if 0 <= idx < len(document.articles):
                art = document.articles[idx]
                t = (art.title_polished or art.title or art.title_raw or '').strip()
                b = article_text(art).strip()
                if t and b:
                    text_out = f"ලිපිය {art_num}: {t}\n\n{b}"
                elif b:
                    text_out = b
                elif t:
                    text_out = t
                else:
                    text_out = f"ලිපිය {art_num} හි පෙළ හඳුනාගත නොහැක."
                return _out(True, 'LOCAL', intent, text_out, warnings=warnings, cursor=cursor)
            else:
                return _out(False, 'LOCAL', intent, f"ලිපිය {art_num} සොයාගත නොහැක. පිටුවේ ඇත්තේ ලිපි {len(document.articles)} ක් පමණි.", warnings=warnings, cursor=cursor)
        except ValueError:
            pass

    if intent == 'LENGTH':
        n = len(text.split())
        if not n:
            return _out(False, 'LOCAL', intent, SI_NOTHING_TO_REPEAT,
                        warnings=warnings, cursor=cursor)
        return _out(True, 'LOCAL', intent, SI_WORDS.format(n=n),
                    warnings=warnings, cursor=cursor)

    if intent == 'TITLE':
        # Numbered, same format as FIRST_SCAN (TURN_CONDITIONS.xlsx N3).
        titles = []
        for idx, a in enumerate(document.articles):
            t = (a.title_polished or a.title or a.title_raw or '').strip()
            if t:
                titles.append(f'ලිපිය {idx + 1}: {t}')
        if not titles:
            return _out(False, 'LOCAL', intent, SI_NO_TITLE,
                        warnings=warnings, cursor=cursor)
        return _out(True, 'LOCAL', intent, ', '.join(titles),
                    warnings=warnings, cursor=cursor)

    if intent == 'WARNINGS':
        if not document.warnings:
            return _out(True, 'LOCAL', intent, SI_NOTHING_MISSED,
                        warnings=warnings, cursor=cursor)
        return _out(True, 'LOCAL', intent,
                    SI_MISSED + ' '.join(document.warnings),
                    warnings=warnings, cursor=cursor)

    if not text:
        return _out(False, 'LOCAL', intent or 'REPEAT', SI_NOTHING_TO_REPEAT,
                    warnings=warnings, cursor=0)

    parts = _parts(document)

    if intent == 'NEXT':
        if cursor >= len(parts):
            return _out(True, 'LOCAL', intent, SI_END,
                        warnings=warnings, cursor=cursor)
        return _out(True, 'LOCAL', intent, parts[cursor],
                    warnings=warnings, cursor=cursor + 1)

    if intent == 'PREVIOUS':
        if cursor <= 1:
            return _out(True, 'LOCAL', intent, SI_START,
                        warnings=warnings, cursor=0)
        return _out(True, 'LOCAL', intent, parts[cursor - 2],
                    warnings=warnings, cursor=cursor - 1)

    if intent == 'FIRST':
        return _out(True, 'LOCAL', intent, parts[0],
                    warnings=warnings, cursor=1)

    return _out(True, 'LOCAL', intent or 'REPEAT', text,
                warnings=warnings, cursor=0)


def _run_inprocess(document: Document, voice: dict, warnings: list) -> dict | None:
    """Execute Nadee's upgraded pipeline directly in-process."""
    try:
        try:
            from .pipeline import run_pipeline
        except (ImportError, ValueError):
            from pipeline import run_pipeline

        ocr_payload = rag_payload(document) if document else {}
        raw_ocr = {
            'corrected_text': ocr_payload.get('corrected_text', ''),
            'tokens': ocr_payload.get('tokens', []),
        }
        # A request about ONE article uses only that article's text, under its
        # real number, so the answer cannot mix in another article.
        only = article_evidence(document, voice.get('article_no')) if document else ''
        if only:
            raw_ocr = {'corrected_text': only, 'tokens': []}
            warnings.append(f"rag: evidence limited to article {voice.get('article_no')}")

        age_val = voice.get('age_group') or (voice.get('personalization_flags') or {}).get('age_group')
        ctx = {'age_group': age_val}

        res = run_pipeline(
            vectorstore=None,
            raw_ocr_input=raw_ocr,
            raw_voice_input=voice,
            raw_request_context=ctx,
            skip_ocr=not bool(raw_ocr['corrected_text']),
            use_history=True,
            use_web_search=True,
        )
        return res
    except Exception as e:
        warnings.append(f'rag in-process error: {type(e).__name__}: {e}')
        return None


def answer(document: Document, voice: dict, mode: str = None,
           cursor: int = 0) -> dict:
    """Route one question. Never raises."""
    mode = (mode or RAG_MODE or 'off').lower()
    voice = voice or {}
    route = str(voice.get('route') or 'GENERATE').upper()
    intent = str(voice.get('intent') or '').upper()

    warnings = []
    if voice.get('warning'):
        warnings.append(voice['warning'])
    if voice.get('source') in ('stub', 'stub-fallback', 'stub-service'):
        warnings.append(f"voice routing: {voice['source']}")

    # ---- the phone acts rather than speaks ------------------------------
    if intent == 'STOP':
        return _out(True, 'LOCAL', 'STOP', speakable='', warnings=warnings,
                    cursor=cursor)

    # ---- a delivery request is never a question ---------------------------
    # The turn policy (turns.py) handles these with a confirmation and a
    # replay. If one reaches here directly it is still answered locally:
    # on 17 Oct 2026 "speak fast" was sent to RAG and came back as
    # "not enough information".
    if intent in DELIVERY_INTENTS:
        return _out(True, 'LOCAL', intent, SI_OK, warnings=warnings, cursor=cursor)

    # ---- answerable from session ----------------------------------------
    if route in ('LOCAL', 'TTS_REPLAY') or intent in LOCAL_INTENTS:
        return _local(document, intent, max(0, int(cursor or 0)), warnings)

    # ---- needs Component 3 ----------------------------------------------
    if mode == 'off':
        if intent == 'FIRST_SCAN':
            return _local(document, intent, max(0, int(cursor or 0)), warnings)
        warnings.append('rag: disabled (SINHALA_RAG_MODE=off)')
        return _out(False, 'GENERATE', intent,
                    speakable=SI_NO_SERVICE, warnings=warnings, cursor=cursor)

    if mode == 'http':
        # history_override: the caller's chat_history is exact, even when empty
        # (re-generating turn N must see only turns 1..N-1, not later ones).
        if (document and hasattr(document, 'generations') and document.generations
                and not voice.get('history_override') and not voice.get('chat_history')):
            voice['chat_history'] = [
                {
                    'question': g.get('query') or '',
                    'standalone_question': g.get('standalone_query') or g.get('query') or '',
                    'answer': g.get('answer') or g.get('answer_si') or ''
                }
                for g in document.generations
            ]
        body = {'ocr': rag_payload(document), 'voice': voice}
        reply, reason = svc.post_json(f'{RAG_URL.rstrip("/")}/answer', body,
                                      timeout_s=RAG_TIMEOUT_S)
        if reply is None:
            if intent == 'FIRST_SCAN':
                warnings.append(f'rag: {reason}')
                return _local(document, intent, max(0, int(cursor or 0)), warnings)
            warnings.append(f'rag: {reason}')
            return _out(False, 'GENERATE', intent,
                        speakable=SI_FAILED, warnings=warnings, cursor=cursor)

        answer_si = (reply.get('answer_si') or '').strip()
        speakable = make_speakable(reply.get('speakable_text') or answer_si or '')
        if not speakable:
            if intent == 'FIRST_SCAN':
                warnings.append('rag: empty answer')
                return _local(document, intent, max(0, int(cursor or 0)), warnings)
            warnings.append('rag: empty answer')
            return _out(False, 'GENERATE', reply.get('intent') or intent,
                        speakable=SI_FAILED, warnings=warnings, cursor=cursor)

        if reply.get('stub'):
            if intent == 'FIRST_SCAN':
                warnings.append('rag: stub service fallback to local first scan')
                return _local(document, intent, max(0, int(cursor or 0)), warnings)
            warnings.append('rag: stub service, not Component 3')

        service_ok = reply.get('ok', True) is not False
        for n in reply.get('notes') or []:
            warnings.append(f'rag: {n}')

        if not service_ok:
            warnings.append('rag: the service reported a failure')
            q_raw = (voice.get('english_translation') or voice.get('sinhala_input') or voice.get('query') or '').strip().lower()
            if '3' in q_raw or 'තුන' in q_raw or 'third' in q_raw:
                return _local(document, 'ARTICLE_3', max(0, int(cursor or 0)), warnings)
            elif '2' in q_raw or 'දෙක' in q_raw or 'second' in q_raw:
                return _local(document, 'ARTICLE_2', max(0, int(cursor or 0)), warnings)
            elif '1' in q_raw or 'එක' in q_raw or 'first' in q_raw:
                return _local(document, 'ARTICLE_1', max(0, int(cursor or 0)), warnings)
            if speakable and speakable != SI_FAILED:
                return _out(False, 'GENERATE', str(reply.get('intent') or intent),
                            speakable=speakable, answer_si=answer_si,
                            sources=[s for s in (reply.get('retrieved_sources') or []) if isinstance(s, dict)],
                            warnings=warnings, cursor=cursor)
            elif document.articles:
                res_loc = _local(document, intent, max(0, int(cursor or 0)), warnings)
                res_loc['ok'] = False
                return res_loc

        sources = reply.get('retrieved_sources') or []
        if not isinstance(sources, list):
            sources = []
        return _out(service_ok, 'GENERATE', str(reply.get('intent') or intent),
                    speakable=speakable, answer_si=answer_si,
                    sources=[s for s in sources if isinstance(s, dict)],
                    warnings=warnings, cursor=cursor)

    # In-process upgraded pipeline execution (mode == 'local' or other direct modes)
    pipe_res = _run_inprocess(document, voice, warnings)
    if pipe_res is not None:
        answer_si = (pipe_res.get('answer_si') or '').strip()
        speakable = make_speakable(pipe_res.get('speakable_text') or answer_si or '')
        service_ok = bool(pipe_res.get('ok', True))

        if pipe_res.get('standalone_query'):
            warnings.append(f"query rewritten: '{pipe_res['standalone_query']}'")
        if pipe_res.get('web_used'):
            warnings.append("rag: answer generated using Tavily web search")
        if pipe_res.get('perplexity'):
            warnings.append(f"rag: perplexity={round(pipe_res['perplexity'], 2)}")

        sources = pipe_res.get('retrieved_sources') or []
        if not isinstance(sources, list):
            sources = []

        return _out(service_ok, 'GENERATE', intent,
                    speakable=speakable or SI_FAILED,
                    answer_si=answer_si or SI_FAILED,
                    sources=[s for s in sources if isinstance(s, dict)],
                    warnings=warnings, cursor=cursor)

    # Fallback if in-process generation fails
    if intent == 'FIRST_SCAN':
        return _local(document, intent, max(0, int(cursor or 0)), warnings)

    q_raw = (voice.get('english_translation') or voice.get('sinhala_input') or voice.get('query') or '').strip().lower()
    if '3' in q_raw or 'තුන' in q_raw or 'third' in q_raw:
        return _local(document, 'ARTICLE_3', max(0, int(cursor or 0)), warnings)
    elif '2' in q_raw or 'දෙක' in q_raw or 'second' in q_raw:
        return _local(document, 'ARTICLE_2', max(0, int(cursor or 0)), warnings)
    elif '1' in q_raw or 'එක' in q_raw or 'first' in q_raw:
        return _local(document, 'ARTICLE_1', max(0, int(cursor or 0)), warnings)

    if document.articles:
        res_loc = _local(document, intent, max(0, int(cursor or 0)), warnings)
        res_loc['ok'] = False
        return res_loc

    return _out(False, 'GENERATE', intent, speakable=SI_FAILED,
                warnings=warnings, cursor=cursor)
