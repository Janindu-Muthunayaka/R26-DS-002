"""
LAYER 6 — generator / RAG.  COMPONENT 3. OWNER: Nadee.

IN : the Document held in session, plus Layer 0's routing dict
OUT: {'ok', 'route', 'intent', 'speakable', 'answer_si', 'sources',
      'warnings', 'cursor'}

`speakable` is the only field the phone needs. **The phone speaks `speakable`
whenever it is non-empty, whatever `ok` says.** `ok` records whether an answer
was generated. A blind user must hear something, and "the answering service is
not available" is something. Silence is the one unacceptable outcome.

NOT EVERY QUESTION NEEDS COMPONENT 3. "read that again" needs the text already
in session. Routing those locally is not a stub — it is the correct
implementation, works with no network and no API key, and means the most
common follow-up survives a RAG outage during a viva.

MODES: `off` (default) — local intents work, a question needing generation
gets an honest "not available". `http` — POST to RAG_URL/answer.
"""
from __future__ import annotations

from core import svc
from core.config import RAG_MODE, RAG_TIMEOUT_S, RAG_URL
from core.schemas import Document
from core.textutils import sentences
from layers.l5_assemble.payload import rag_payload, article_text

# Intents answerable from what is already in session. No service, no network,
# no API key — the part of the conversation that cannot break on the day.
# These are NOT stubs: each one is the correct implementation.
LOCAL_INTENTS = {
    'REPEAT', 'READ_ALOUD', 'REPLAY',    # the whole article again
    'NEXT', 'PREVIOUS', 'FIRST',         # walk through it in parts
    'LENGTH', 'TITLE', 'WARNINGS',       # ask about it
}

# An article read start to finish is two or three thousand characters. A
# listener who wants the middle should not sit through the start, so "next"
# walks the article in parts. NOT MEASURED — a readability choice, stated as
# one. `sentences()` does the splitting, so a part never begins mid-sentence.
PART_MAX_CHARS = 400

# Spoken strings. NOT WRITTEN BY A NATIVE SPEAKER — have these checked before
# the viva. Two are lifted verbatim from Component 3's own source.
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
    """Answer from the article in session. Returns the new cursor.

    `cursor` counts PARTS ALREADY DELIVERED, so 0 means "not started" and
    parts[cursor] is always the next one. Keeping that one definition is what
    stops NEXT and PREVIOUS disagreeing about where the listener is.
    """
    text = _text_of(document)

    if intent == 'FIRST_SCAN':
        n = len(document.articles)
        titles = []
        for a in document.articles:
            t = (a.title_polished or a.title or a.title_raw or '').strip()
            if t:
                titles.append(t)
        if n == 0:
            msg = "කිසිදු ලිපියක් හඳුනාගෙන නොමැත."
        else:
            msg = f"ලිපි {n} ක් හඳුනාගෙන ඇත."
            if titles:
                msg += " එම ලිපිවල ශීර්ෂයන් වන්නේ: " + ", ".join(titles)
            else:
                msg += " නමුත් ඒවායේ ශීර්ෂයන් හඳුනාගත නොහැක."
        return _out(True, 'LOCAL', intent, msg, warnings=warnings, cursor=cursor)

    if intent == 'LENGTH':
        n = len(text.split())
        if not n:
            return _out(False, 'LOCAL', intent, SI_NOTHING_TO_REPEAT,
                        warnings=warnings, cursor=cursor)
        return _out(True, 'LOCAL', intent, SI_WORDS.format(n=n),
                    warnings=warnings, cursor=cursor)

    if intent == 'TITLE':
        titles = [a.title.strip() for a in document.articles if a.title.strip()]
        if not titles:
            return _out(False, 'LOCAL', intent, SI_NO_TITLE,
                        warnings=warnings, cursor=cursor)
        return _out(True, 'LOCAL', intent, ' '.join(titles),
                    warnings=warnings, cursor=cursor)

    if intent == 'WARNINGS':
        # The phone only says "some parts were skipped" after a capture. This
        # is how the listener finds out WHICH parts, on demand. The warnings
        # are English diagnostics from l3_segment; speaking them to a Sinhala
        # listener is imperfect and is a known limitation, not an oversight —
        # they are also read by developers.
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

    # REPEAT / READ_ALOUD / REPLAY: the whole article, and the walk restarts.
    return _out(True, 'LOCAL', intent or 'REPEAT', text,
                warnings=warnings, cursor=0)


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
        # So a transcript can never be read as evidence that Component 4 ran.
        warnings.append(f"voice routing: {voice['source']}")

    # ---- the phone acts rather than speaks ------------------------------
    if intent == 'STOP':
        return _out(True, 'LOCAL', 'STOP', speakable='', warnings=warnings,
                    cursor=cursor)

    # ---- answerable from session ----------------------------------------
    if route in ('LOCAL', 'TTS_REPLAY') or intent in LOCAL_INTENTS:
        return _local(document, intent, max(0, int(cursor or 0)), warnings)

    # ---- needs Component 3 ----------------------------------------------
    if mode != 'http':
        if intent == 'FIRST_SCAN':
            return _local(document, intent, max(0, int(cursor or 0)), warnings)
        warnings.append('rag: disabled (SINHALA_RAG_MODE=off)')
        return _out(False, 'GENERATE', intent,
                    speakable=SI_NO_SERVICE, warnings=warnings, cursor=cursor)

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
    speakable = (reply.get('speakable_text') or answer_si or '').strip()
    if not speakable:
        warnings.append('rag: empty answer')
        return _out(False, 'GENERATE', reply.get('intent') or intent,
                    speakable=SI_FAILED, warnings=warnings, cursor=cursor)

    if reply.get('stub'):
        warnings.append('rag: stub service, not Component 3')

    # Component 3 signals failure with `ok: false` and still fills
    # `speakable_text` with a sentence the user can hear. Without honouring
    # that flag a failure sentence — a perfectly non-empty string — comes back
    # looking exactly like a successful answer.
    service_ok = reply.get('ok', True) is not False
    for n in reply.get('notes') or []:
        warnings.append(f'rag: {n}')
    if not service_ok:
        warnings.append('rag: the service reported a failure')

    sources = reply.get('retrieved_sources') or []
    if not isinstance(sources, list):
        sources = []
    return _out(service_ok, 'GENERATE', str(reply.get('intent') or intent),
                speakable=speakable, answer_si=answer_si,
                sources=[s for s in sources if isinstance(s, dict)],
                warnings=warnings, cursor=cursor)
