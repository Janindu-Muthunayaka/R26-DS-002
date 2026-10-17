"""
Retrieval and generation.  COMPONENT 3.

WHAT IS NADEE'S AND KEPT VERBATIM
---------------------------------
  * `PROMPT` — the Sinhala system prompt, including "answer only from the
    evidence" and the "not enough information in the text that was read"
    fallback. That fallback is the grounding guarantee and it is hers.
  * `DETAIL_LEVEL_WORD_LIMITS`, `STYLE_CLASS_WORD_LIMITS`, `resolve_max_words`
    — including the case-insensitive match and the observation that
    `style_class` is the reliable signal because `detail_level` is sometimes
    absent.
  * `sinhala_purity` and the strict retry when an answer comes back
    code-switched.
  * The chunk metadata shape and the "always retrieve the current page" rule.
  * Multi-turn chat history and follow-up query rewriting.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `system/` on the path for core.llm and core.env ONLY. Both are stdlib-only.
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO / 'system') not in sys.path:
    sys.path.insert(0, str(_REPO / 'system'))

from core import llm                                   # noqa: E402
from ingest import records_from_text                    # noqa: E402
from store import VectorStore, record_hash, text_hash   # noqa: E402
from history import get_history, add_turn, format_history_for_prompt  # noqa: E402
from rewrite import rewrite_query                       # noqa: E402

def clean_ocr_sinhala(text: str) -> str:
    """Normalize common Sinhala OCR artifacts and character confusions."""
    if not text:
        return ""
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    text = text.replace('ප් ර', 'ප්‍ර').replace('ත්ර', 'ත්‍ර').replace('ක්‍ර', 'ක්‍ර').replace('ක් ර', 'ක්‍ර')
    text = text.replace('ඟාල්ල', 'ගාල්ල')
    return text


PROMPT = """
ඔබ දෘෂ්ටි විනාශයට ලක් වූ පුද්ගලයින් සඳහා සිංහල කියවීමේ සහායකයෙකි.

උපදෙස්:
- පිළිතුර සිංහල භාෂාවෙන් පමණක් ලබා දෙන්න.
- පහත දක්වා ඇති සාක්ෂි (Evidence) හි සඳහන් කරුණු ඇසුරෙන් අදාළ පිළිතුර ලබා දෙන්න. සාක්ෂි මගින් පිළිතුරු දිය නොහැකි නම් පමණක් "කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත." ලෙස පිළිතුරු දෙන්න.
- ප්‍රශ්නයේ "මේක", "මෙය", "එය" වැනි සර්වනාම තිබේ නම්, පෙර සංවාදය හෝ සාක්ෂි (Evidence) හි සඳහන් කරුණු ඇසුරෙන් එය කුමක්දැයි තේරුම් ගෙන අදාළ පිළිතුර ලබා දෙන්න.
- "පෙර සංවාදය" කොටස ඇත්තේ පරිශීලකයා කලින් ඇසූ දේ තේරුම් ගැනීමට පමණි; කරුණු සඳහා Evidence කොටස පමණක් භාවිතා කරන්න.
- පරිශීලකයා "තව කියන්න" වැනි ලෙස වැඩිදුර විස්තර ඉල්ලා ඇත්නම්, පෙර පිළිතුරේ දැනටමත් සඳහන් කළ කරුණු නැවත නොකියා, Evidence හි ඇති අලුත් කරුණු ලබා දෙන්න.
- {prompt_modifier}
- උපරිම වචන ගණන: {max_words}.

Intent: {intent}
Style: {style_class}

පෙර සංවාදය (සන්දර්භය පමණි, සාක්ෂියක් නොවේ):
{chat_history}

Evidence:
{evidence}

පරිශීලකයාගේ ප්‍රශ්නය: {query_text}

පිළිතුර:
"""

DETAIL_LEVEL_WORD_LIMITS = {
    'brief': 80, 'moderate': 200, 'detailed': 400,
    'structured': 300, 'full': 500,
}
STYLE_CLASS_WORD_LIMITS = {
    'simple': 80, 'moderate': 200, 'detailed': 400, 'stepbystep': 300,
}

SINHALA_RANGE = (0x0D80, 0x0DFF)
STRICT_SUFFIX = (' වැදගත්: පෙර පිළිතුරේ සිංහල නොවන වචන තිබුණි. මෙවර පිළිතුර '
                 'සම්පූර්ණයෙන්ම සිංහල අක්ෂර වලින් පමණක් ලියන්න — ඉංග්‍රීසි '
                 'වචනයක් හෝ අකුරක් හෝ නොමැතිව.')

NO_EVIDENCE = 'කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත.'
FAILED = 'පිළිතුර ලබාගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න.'


def resolve_max_words(style_class: str, flags: dict) -> int:
    detail = (flags or {}).get('detail_level')
    if detail in DETAIL_LEVEL_WORD_LIMITS:
        return DETAIL_LEVEL_WORD_LIMITS[detail]
    key = (style_class or '').lower().replace(' ', '').replace('_', '')
    return STYLE_CLASS_WORD_LIMITS.get(key, STYLE_CLASS_WORD_LIMITS['moderate'])


def sinhala_purity(text: str) -> float:
    letters = [c for c in text or '' if c.isalpha()]
    if not letters:
        return 1.0
    n = sum(1 for c in letters if SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1])
    return n / len(letters)


EMBED_BATCH = 64


def embed_records(records: list) -> tuple:
    """(records, vectors, reason). Batched, order-preserving."""
    if not records:
        return [], [], ''
    vectors = []
    for i in range(0, len(records), EMBED_BATCH):
        batch = records[i:i + EMBED_BATCH]
        vecs, reason = llm.embed([r['text'] for r in batch])
        if vecs is None:
            return [], [], reason
        vectors.extend(vecs)
    return records, vectors, ''


def index_records(store: VectorStore, records: list, skip_known=True) -> tuple:
    """Embed and store, skipping text already present. (added, reason)."""
    if skip_known:
        known = store.known_hashes()
        records = [r for r in records if record_hash(r) not in known]
    if not records:
        return 0, ''
    recs, vecs, reason = embed_records(records)
    if reason:
        return 0, reason
    n = store.add(recs, vecs)
    store.save()
    return n, ''


def retrieve(store: VectorStore, query_text: str, retrieved_chunk_id=None,
             top_k: int = 4) -> tuple:
    """Nadee's three-step rule, on this store. (documents, reason).

    1. Always pull in the page being read right now.
    2. Honour an anchor chunk the voice module already chose.
    3. Fill the rest semantically.
    """
    collected, seen = [], set()

    def add(docs):
        for d in docs:
            cid = d.get('metadata', {}).get('chunk_id') or d.get('chunk_id')
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            collected.append(d)

    qvec, reason = llm.embed([query_text or ''])
    if qvec is None:
        return [], reason
    q = qvec[0]

    # 1. Always pull in the current page chunks in natural reading order
    current_docs = list(store.get_where('source_type', 'ocr_current'))
    current_docs.sort(key=lambda d: d.get('chunk_id') or d.get('metadata', {}).get('chunk_id', ''))
    add(current_docs)

    # 2. Honour an anchor chunk if specified
    if retrieved_chunk_id:
        add([c for c in store.get_where('chunk_id', retrieved_chunk_id)])

    # 3. Only look into past articles if current article has no content or if specifically needed
    remaining = top_k - len(collected)
    if remaining > 0 and not current_docs:
        pool = store.search(q, k=remaining + len(collected))
        add([d for d in pool
             if d.get('metadata', {}).get('source_type') != 'ocr_current'
             and d.get('score', 0) >= 0.35
             ][:remaining])

    return collected[:top_k], ''


def generate(query_text, intent, style_class, prompt_modifier, max_words,
             evidence_docs, sinhala_query: str = '', chat_history: str = '') -> tuple:
    """(answer, reason). Never raises."""
    evidence = '\n---\n'.join(d['text'] for d in evidence_docs)
    if not evidence.strip():
        return NO_EVIDENCE, ''

    if sinhala_query and sinhala_query.strip() != (query_text or '').strip():
        display_question = f"{sinhala_query}\n(English translation: {query_text})"
    else:
        display_question = query_text or sinhala_query or ''

    def ask(modifier):
        filled = PROMPT.format(
            prompt_modifier=modifier or '', max_words=max_words,
            intent=intent or '', style_class=style_class or '',
            evidence=evidence, query_text=display_question,
            chat_history=chat_history or '(නැත)')
        return llm.chat([{'role': 'user', 'content': filled}],
                        temperature=0.2)

    answer, reason = ask(prompt_modifier)
    if answer is None:
        return None, reason

    # Nadee's guard: prompting alone cannot guarantee no code-switching.
    if sinhala_purity(answer) < 0.85:
        retry, reason2 = ask((prompt_modifier or '') + STRICT_SUFFIX)
        if retry and sinhala_purity(retry) >= sinhala_purity(answer):
            answer = retry
    return answer.strip(), ''


def run(store: VectorStore, ocr: dict, voice: dict, remember: bool = True
        ) -> dict:
    """The `/answer` contract. Returns Component 3's four fields plus notes."""
    ocr = ocr or {}
    voice = voice or {}
    notes = []

    route = str(voice.get('route') or 'GENERATE').upper()
    intent = voice.get('intent') or ''
    if route != 'GENERATE':
        return {'ok': True, 'intent': intent,
                'answer_si': f"'{route}' route සඳහා RAG පිළිතුරු ජනනය නොකරයි.",
                'retrieved_sources': [], 'speakable_text': None, 'notes': notes}

    session_id = voice.get('session_id') or 'default'
    caller_history = voice.get('chat_history')
    # history_override: use the caller's list exactly, even when it is empty
    # (a re-generated first turn must not see the turns that came after it).
    if isinstance(caller_history, list) and (caller_history or voice.get('history_override')):
        history = caller_history
    else:
        history = get_history(session_id)

    raw_current = (ocr.get('corrected_text') or '').strip()
    current = clean_ocr_sinhala(raw_current)

    # The page being read now REPLACES the previous one
    if current:
        store.delete_where('source_type', 'ocr_current')
        recs = records_from_text(current, 'ocr_current', 'ocr_current')
        n, reason = index_records(store, recs, skip_known=False)
        if reason:
            notes.append(f'indexing the current page failed: {reason}')

        if remember:
            base = text_hash(current)
            kept, reason = index_records(
                store, records_from_text(current, 'read', base,
                                         {'article_id': base}))
            if kept:
                notes.append(f'remembered {kept} chunk(s) from this article')

    query = voice.get('english_translation') or ''
    sinhala_query = voice.get('sinhala_input') or voice.get('text') or ''
    retrieval_query = sinhala_query or query

    # Multi-turn follow-up rewriting
    standalone_query = rewrite_query(retrieval_query, history)
    if standalone_query and standalone_query != retrieval_query:
        notes.append(f"query rewritten: '{standalone_query}'")

    docs, reason = retrieve(store, standalone_query or retrieval_query, voice.get('retrieved_chunk_id'))
    if reason:
        return {'ok': False, 'intent': intent, 'answer_si': FAILED,
                'retrieved_sources': [], 'speakable_text': FAILED,
                'notes': notes + [f'retrieval failed: {reason}']}

    chat_history_text = format_history_for_prompt(history)

    answer, reason = generate(
        query_text=query, intent=intent,
        style_class=voice.get('style_class'),
        prompt_modifier=voice.get('prompt_modifier'),
        max_words=resolve_max_words(voice.get('style_class'),
                                    voice.get('personalization_flags')),
        evidence_docs=docs,
        sinhala_query=standalone_query or sinhala_query,
        chat_history=chat_history_text)
    if answer is None:
        return {'ok': False, 'intent': intent, 'answer_si': FAILED,
                'retrieved_sources': [], 'speakable_text': FAILED,
                'notes': notes + [f'generation failed: {reason}']}

    is_abstention = answer.strip() == NO_EVIDENCE
    if is_abstention:
        notes.append('no evidence found in read text for this question')
    # A re-generation replaces an existing turn; it is not a new one.
    if not voice.get('no_history_write'):
        add_turn(retrieval_query, standalone_query, answer, session_id)

    return {
        'ok': True,
        'is_abstention': is_abstention,
        'intent': intent,
        'answer_si': answer,
        'retrieved_sources': [
            dict(d.get('metadata', {}), score=d.get('score')) for d in docs],
        'speakable_text': answer,
        'notes': notes,
    }
