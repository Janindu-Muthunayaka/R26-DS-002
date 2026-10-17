"""
LAYER 4C — optional LLM post-editing.  OWNER: Ishara.  DEFAULT: OFF.

READ THIS BEFORE TURNING IT ON
==============================

Handing corrupt Sinhala to a general-purpose language model and asking it to
"fix" the text is the most dangerous thing in this repository, for two
separate reasons.

**1. It can invent the news.** A model given a shattered sentence will not
return a shattered sentence. It will return a fluent, grammatical, plausible
one — with names, numbers and dates that were never on the page — and the
phone will read it to a blind user in the same confident voice it uses for the
real article. That is a worse failure than reading nothing, because reading
nothing is honest.

**2. It destroys the measurement.** Component 2's contribution is mT5 post-OCR
correction at CER 0.1197 -> 0.0757 over 217 page-disjoint sentences. If a
language model rewrites that output, the thing being measured is no longer the
model the thesis is about. `tests/test_polish.py` asserts the evaluation tools
cannot switch this on.

So this layer exists, and it is built to be distrusted:

  * OFF by default.
  * It runs AFTER mT5 and writes to a SEPARATE field. `article.body` — the
    mT5 output, the research artifact — is never overwritten.
  * Four guards. A post-edit that fails any of them is DISCARDED.
  * Every article it touches carries a warning to the phone, so no transcript
    can be read as unassisted output.

`auto` deliberately does NOT run on 'unreadable' text. That is where a repair
would be most welcome and where invention is most likely: with little real
signal left, fluency is all the model has to go on. On unreadable text the
honest system says "I could not read that, try again" — which is also the
response that gets the user a better photograph.

THE GUARDS
----------
1. similarity   character-level, against the mT5 text, >= POLISH_MIN_SIMILARITY
2. length       within [POLISH_MIN_LEN_RATIO, POLISH_MAX_LEN_RATIO]
3. script       Sinhala ratio must not fall
4. no inflation word count must not grow by more than 20%

Guard 1 is the one that matters. The rest catch the ways a model gets around
it: replying in English, padding, or answering the text instead of repairing.
"""
from __future__ import annotations

import difflib
import json
import os
import re

from core import llm, quality
from core.config import (POLISH_MAX_CHARS, POLISH_MAX_LEN_RATIO,
                         POLISH_MIN_LEN_RATIO, POLISH_MIN_SIMILARITY,
                         POLISH_MODE, POLISH_MODEL_ENV, POLISH_REASONING)
from core.schemas import Article

SYSTEM_PROMPT = (
    'You repair Sinhala OCR text from scanned Sri Lankan newspaper articles. Every token you keep is '
    'read aloud by text-to-speech to a blind listener, so garbage must be repaired into the real word '
    'or removed. You are a repairer, not a writer.\n'
    '\n'
    'INPUT: a JSON array of {"id", "title", "alt_titles" (optional: other OCR readings of the headline), "body"}.\n'
    'OUTPUT: ONLY a valid JSON array, one {"id", "title", "body"} per input, same ids, containing only the '
    'repaired text: no preamble, no markdown fences, no comments. Add "merge_with" / "split_at" / '
    '"split_title" only as the SEGMENTATION CHECK says.\n'
    '\n'
    'GARBAGE: (a) symbol or punctuation runs, box/bullet glyphs, stray single letters or digits, '
    'fragments of rule lines or photo edges; (b) Latin/Tamil/other-script characters standing in for '
    'Sinhala strokes (c, l, I, 1, |, 0, o) and legacy-font mojibake (e.g. "YS% ,xld"); (c) broken '
    'Unicode: orphan, doubled or stacked vowel signs or "්", stray zero-width joiners, words split or '
    'fused; (d) look-alike swaps: ට/ථ ච/ව ඩ/ඪ ද/ඳ බ/ඛ/ඩ ප/ඵ ග/ශ ශ/ෂ න/ණ ල/ළ ස/ඝ ම/ඹ එ/ඵ ි/ී ු/ූ '
    'ෙ/ේ ො/ෝ, missing or extra "්", missing "ං"; (e) non-words that are not a plausible name, number, '
    'date, abbreviation or Latin proper noun.\n'
    'FOR EACH: REPAIR it if the intended word is clear from the surviving letters and the sentence '
    '(smallest edit; decode mojibake only if certain); otherwise REMOVE it, and always remove pure '
    'noise, leaving no half-words or symbol runs and tidy spacing. KEEP valid words, names, numbers, '
    'dates, amounts, abbreviations and genuine Latin proper nouns. Never guess: if unsure between a '
    'repair and a deletion, delete.\n'
    '\n'
    'HARD LIMITS (outputs that break these are thrown away): NEVER add facts - no names, numbers, '
    'dates, places, quotes or clauses not in the input; never change a digit except a plain OCR swap '
    'inside a Sinhala word. NEVER remove readable words or real facts. Never summarise, paraphrase, '
    'reorder, translate, explain or continue the text. Fix grammar only where it is an OCR artifact. '
    'Keep roughly the same length (at most about a third shorter), the same sentences, mostly the same '
    'characters. Output Sinhala script; keep digits and genuine Latin names as they are.\n'
    '\n'
    'HEADLINE: apply the same rules to "title". Where alt_titles differ, pick for each word the reading '
    'that is a real Sinhala word and agrees with the body; output ONE clean headline. If every reading '
    'is empty or garbage, write a short neutral headline using only words and facts from the body.\n'
    'DUDS: return "[DISCARD]" for both title and body ONLY if the body has no readable news sentence '
    'at all; if even one real sentence survives, clean it and keep it.\n'
    '\n'
    'SEGMENTATION CHECK (the default is to report NOTHING). The articles were cut out automatically and '
    'are in reading order; a story may be split across two consecutive objects, or two stories put in '
    'one. Report it only when OBVIOUS from the text - never guess, never infer from topic similarity alone:\n'
    '  - "merge_with": "<id of the NEXT object>" only if the next object unmistakably continues this '
    'story: this body stops mid-sentence and the next finishes it, or the next has no headline and '
    'continues the same event with the same named people and places. A similar subject is NOT a merge.\n'
    '  - "split_at": "<first 3-8 words of the second story, copied EXACTLY from this body>" only if this '
    'body unmistakably holds two unrelated stories (a second headline-like line, then a different event, '
    'people and places), at the start of a sentence; optionally "split_title": "<that headline, copied EXACTLY>".\n'
    '  - If you are not certain, do NOT add these fields. Never rewrite or move text for it; only report.'
)


def _payload_item(p: dict) -> dict:
    """One article for the LLM. Only headline readings that DIFFER from the
    title are sent (title_mat is empty and title_hybrid usually equals
    title_raw): same information, fewer tokens."""
    item = {"id": p["id"], "title": p["title"]}
    alts = []
    for t in (p.get("title_raw"), p.get("title_hybrid"), p.get("title_mat")):
        t = (t or "").strip()
        if t and t != p["title"] and t not in alts:
            alts.append(t)
    if alts:
        item["alt_titles"] = alts
    item["body"] = p["body"]
    return item


def polish_model() -> str:
    """The model L4C uses: OPENAI_POLISH_MODEL, else the shared OPENAI_CHAT_MODEL."""
    return (os.getenv(POLISH_MODEL_ENV) or '').strip() or llm.chat_model()


def polish(body_text: str, mode: str = None) -> dict:
    """Legacy helper for single string polishing to keep tests passing.
    
    Returns:
        {'applied': bool, 'text': str, 'reason': str, 'similarity': float, 'score': dict}
    """
    from core.schemas import Box
    a = Article(index=0, box=Box(x1=0, y1=0, x2=10, y2=10), body=body_text, title="කුරුණෑගල නගර සභාවේ රැස්වීම්")
    res = polish_articles([a], mode=mode)[0]
    return {
        'applied': res['applied'],
        'text': res['body'],
        'reason': res['reason'],
        'similarity': res['similarity'],
        'score': res['measures']
    }


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _result(body: str, title: str, applied: bool, reason: str, similarity=None, measures=None) -> dict:
    return {'body': body, 'title': title, 'applied': bool(applied), 'reason': reason,
            'similarity': similarity, 'measures': measures or {}}


def check(original: str, candidate: str) -> tuple:
    """Run the four guards on the BODY text. Returns (accepted, reason, similarity)."""
    o, c = (original or '').strip(), (candidate or '').strip()
    if not c:
        return False, 'empty reply', 0.0

    sim = _similarity(o, c)
    if sim < POLISH_MIN_SIMILARITY:
        return False, (f'rewrote too much (similarity {sim:.2f} < '
                       f'{POLISH_MIN_SIMILARITY:.2f})'), sim

    ratio = len(c) / max(1, len(o))
    if not (POLISH_MIN_LEN_RATIO <= ratio <= POLISH_MAX_LEN_RATIO):
        return False, f'length changed by {ratio:.2f}x', sim

    if quality.sinhala_ratio(c) < quality.sinhala_ratio(o) - 0.01:
        return False, 'the reply is less Sinhala than the input', sim

    ow, cw = quality.word_count(o), quality.word_count(c)
    if cw > ow * 1.5 + 4:
        return False, f'word count grew {ow} -> {cw}', sim

    return True, '', sim


def _is_nonsense(text: str) -> bool:
    """Returns True if the text is empty or has fewer than 10 alphanumeric characters."""
    if not text:
        return True
    alphanumeric = re.sub(r'[^\w\s]', '', text)
    return len(alphanumeric.replace(' ', '')) < 10


def _body_makes_sense(body: str) -> bool:
    """Returns True if the body text is sensible and not complete gibberish."""
    if not body:
        return False
    if _is_nonsense(body):
        return False
    
    from core import quality
    m = quality.score(body)
    
    # If the text is marked as unreadable and is fatal, it does not make sense.
    if m['verdict'] == 'unreadable' and m['fatal']:
        return False
    # Loosened: allow articles that have numbers, dates, Latin acronyms, or fragmented OCR tokens
    if m['sinhala_ratio'] < 0.35 or m['short_ratio'] > 0.65:
        return False
        
    return True


def _title_makes_sense(title: str) -> bool:
    """Returns True if the title is real, sensible, and actual."""
    if not title:
        return False
        
    from core import quality
    # Loosened: allow titles with English or mixed text (sinhala_ratio >= 0.35)
    if quality.sinhala_ratio(title) < 0.35:
        return False
        
    # A title should not be too short in terms of letters (less than 3 characters is probably gibberish/fragment)
    alphanumeric = re.sub(r'[^\w\s]', '', title)
    if len(alphanumeric.replace(' ', '')) < 3:
        return False
        
    return True


def polish_articles(articles: list[Article], mode: str = None) -> list[dict]:
    """Post-edit title and body for all articles in a single batch. Returns a list of result dicts."""
    mode = (mode or POLISH_MODE or 'off').lower()
    
    results = [None] * len(articles)
    to_process = []
    
    for i, article in enumerate(articles):
        orig_raw = (getattr(article, 'title_raw', '') or '').strip()
        orig_mat = (getattr(article, 'title_mat', '') or '').strip()
        orig_hybrid = (getattr(article, 'title_hybrid', '') or '').strip()
        orig_title = (article.title or orig_hybrid or orig_raw or orig_mat or '').strip()
        orig_body = (article.body or article.body_raw or '').strip()

        if mode == 'off':
            results[i] = _result(orig_body, orig_title, False, 'polish: off')
            continue

        # If body doesn't make sense -> skip polish, keep original text unless it's pure nonsense/fatal
        if not _body_makes_sense(orig_body):
            if _is_nonsense(orig_body) or (quality.score(orig_body)['verdict'] == 'unreadable' and quality.score(orig_body)['fatal']):
                results[i] = _result("[DISCARD]", "[DISCARD]", True, 'polish: discarded, body does not make sense')
            else:
                results[i] = _result(orig_body, orig_title, False, 'polish: skipped (body below quality threshold, keeping unpolished)')
            continue

        # If none of the title candidates make sense, signal regeneration
        has_sensible_title = (_title_makes_sense(orig_title) or
                              _title_makes_sense(orig_raw) or
                              _title_makes_sense(orig_mat))
        needs_title_generation = not has_sensible_title

        verdict = quality.score(orig_body)['verdict']
        if mode == 'auto':
            if verdict == 'good' and not needs_title_generation:
                results[i] = _result(orig_body, orig_title, False, 'polish: not needed (text and title are good)')
                continue

        if len(orig_body) > POLISH_MAX_CHARS:
            results[i] = _result(orig_body, orig_title, False,
                           f'polish: skipped, body {len(orig_body)} chars is over the '
                           f'{POLISH_MAX_CHARS} cap')
            continue

        to_process.append({
            'idx': i,
            'id': str(i),
            'title': "" if needs_title_generation else orig_title,
            'title_raw': "" if needs_title_generation else orig_raw,
            'title_mat': "" if needs_title_generation else orig_mat,
            'title_hybrid': "" if needs_title_generation else orig_hybrid,
            'body': orig_body,
            'verdict': verdict,
            'needs_title_generation': needs_title_generation
        })

    if not to_process:
        return results

    ok, why = llm.available()
    if not ok:
        for p in to_process:
            results[p['idx']] = _result(p['body'], p['title'], False, f'polish: unavailable ({why})')
        return results

    # Batch LLM Call with Dual-Hypothesis Title Candidate Payload
    payload_json = [_payload_item(p) for p in to_process]
    prompt_payload = json.dumps(payload_json, ensure_ascii=False)
    
    reply_text, reason = llm.chat(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': prompt_payload}],
        model=polish_model(),
        temperature=0.0,                    # not sent to GPT-5 models (core/llm.py)
        reasoning_effort=POLISH_REASONING)  # short thinking: mechanical repair
        
    if reply_text is None:
        for p in to_process:
            results[p['idx']] = _result(p['body'], p['title'], False, f'polish: call failed ({reason})')
        return results

    # Parse JSON from LLM
    is_json = False
    try:
        cleaned = reply_text.strip('` \n')
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
            
        if '[' in cleaned and ']' in cleaned:
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            cleaned = cleaned[start:end]
            
        reply_json = json.loads(cleaned, strict=False)
        if isinstance(reply_json, list):
            is_json = True
    except (json.JSONDecodeError, ValueError):
        pass

    if not is_json:
        if len(to_process) == 1:
            p = to_process[0]
            # Fall back to treating the entire reply_text as the polished body of that single article
            reply_json = [{"id": p["id"], "title": p["title"], "body": reply_text.strip()}]
        else:
            for p in to_process:
                results[p['idx']] = _result(p['body'], p['title'], False, f'polish: call failed (invalid JSON: {reply_text[:50]}...)')
            return results

    # Map results back
    reply_map = {str(item.get("id", "")): item for item in reply_json if isinstance(item, dict)}

    for p in to_process:
        i = p['idx']
        orig_body = p['body']
        orig_title = p['title']
        
        reply_item = reply_map.get(str(i))
        if not reply_item:
            results[i] = _result(orig_body, orig_title, False, 'polish: call failed (LLM missed this article ID)')
            continue
            
        cand_title = str(reply_item.get('title', '')).strip()
        cand_body = str(reply_item.get('body', '')).strip()

        if cand_body == "[DISCARD]" or cand_title == "[DISCARD]":
            if len(orig_body.replace(' ', '')) >= 20:
                results[i] = _result(orig_body, orig_title, False, 'polish: LLM flagged as dud but text is present, keeping unpolished')
            else:
                results[i] = _result("[DISCARD]", "[DISCARD]", True, 'polish: applied, article discarded by LLM as a dud', 1.0)
            continue

        # Check if the candidate body makes sense
        if not _body_makes_sense(cand_body):
            results[i] = _result(orig_body, orig_title, False, 'polish: candidate body below quality threshold, keeping unpolished', 1.0)
            continue

        # Check if the candidate title makes sense
        if not _title_makes_sense(cand_title):
            if _is_nonsense(orig_title) and not _title_makes_sense(orig_title):
                results[i] = _result("[DISCARD]", "[DISCARD]", True, 'polish: discarded, LLM candidate title does not make sense', 1.0)
                continue
            cand_title = orig_title

        accepted, why, sim = check(orig_body, cand_body)
        if not accepted:
            results[i] = _result(orig_body, orig_title, False, f'polish: REJECTED body — {why}', sim)
            continue

        changed = 1.0 - sim
        results[i] = _result(cand_body, cand_title, True,
                       f'polish: applied, {changed:.1%} of body chars changed',
                       sim, {'before': p['verdict'],
                             'after': quality.score(cand_body)['verdict']})

    _attach_segment_hints(results, to_process, reply_map)
    return results


def _attach_segment_hints(results, to_process, reply_map):
    """results[i]['segment'] = the LLM's merge/split hints for every article
    it saw (empty dict when it reported nothing). Discarded articles get none.
    Hints only: segfix.py checks each one against the final text."""
    for p in to_process:
        i, r = p['idx'], results[p['idx']]
        if r is None or r.get('body') == '[DISCARD]':
            continue
        item = reply_map.get(str(i)) or {}
        hint = {}
        for key in ('merge_with', 'split_at', 'split_title'):
            v = item.get(key)
            if v is not None and str(v).strip():
                hint[key] = str(v).strip()
        r['segment'] = hint

