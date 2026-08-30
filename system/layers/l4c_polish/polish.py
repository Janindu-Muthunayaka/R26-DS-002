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
import re

from core import llm, quality
from core.config import (POLISH_MAX_CHARS, POLISH_MAX_LEN_RATIO,
                         POLISH_MIN_LEN_RATIO, POLISH_MIN_SIMILARITY,
                         POLISH_MODE)
from core.schemas import Article

SYSTEM_PROMPT = (
    'You are a Sinhala language expert repairing OCR errors in newspaper articles.\n'
    'You will receive a JSON array of objects, each containing "id", "title", and "body".\n'
    'RULES, in order of importance:\n'
    '1. Output ONLY a valid JSON array of objects. Output only the repaired text. Each object must have "id", "title" and "body". No preamble, no markdown.\n'
    '2. IDENTIFY DUDS: If a body has almost no meaningful Sinhala text (e.g. only a few nonsense characters, or completely empty), return "[DISCARD]" for both its "title" and "body". Do not attempt to repair a completely meaningless article.\n'
    '3. GENERATE TITLE: If a body has valid information but the title is empty, missing, gibberish (e.g., contains only a few characters, repetitive symbols, or looks nonsensical), or otherwise not a proper headline, USE the context of the body to GENERATE an appropriate short Sinhala headline for the "title" field.\n'
    '4. NEVER add new factual information to any body. No names, numbers, dates, places or clauses that are not already present.\n'
    '5. NEVER remove, summarise, reorder, translate or explain anything in any body.\n'
    '6. Repair only what is clearly an OCR error in the body: a broken word, a wrong dependent vowel sign, a split/merged word, a stray Latin fragment.\n'
    '7. If the title is present and meaningful, repair its OCR errors using the context of the body.\n'
    '8. Output Sinhala script. Leave digits and genuine Latin proper nouns as they are.\n'
    'The output bodies should be almost identical to the input bodies in length and in sentence order.'
)


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
    if cw > ow * 1.2 + 2:
        return False, f'word count grew {ow} -> {cw}', sim

    return True, '', sim


def _is_nonsense(text: str) -> bool:
    """Returns True if the text is empty or has fewer than 10 alphanumeric characters."""
    if not text:
        return True
    alphanumeric = re.sub(r'[^\w\s]', '', text)
    return len(alphanumeric.replace(' ', '')) < 10


def polish_articles(articles: list[Article], mode: str = None) -> list[dict]:
    """Post-edit title and body for all articles in a single batch. Returns a list of result dicts."""
    mode = (mode or POLISH_MODE or 'off').lower()
    
    results = [None] * len(articles)
    to_process = []
    
    for i, article in enumerate(articles):
        orig_title = (article.title or article.title_raw or '').strip()
        orig_body = (article.body or article.body_raw or '').strip()

        if mode == 'off':
            results[i] = _result(orig_body, orig_title, False, 'polish: off')
            continue

        if _is_nonsense(orig_body) and _is_nonsense(orig_title):
            results[i] = _result(orig_body, orig_title, False, 'polish: skipped, title and body are empty or nonsense')
            continue

        verdict = quality.score(orig_body)['verdict']
        if mode == 'auto':
            if verdict == 'good' and not _is_nonsense(orig_title):
                results[i] = _result(orig_body, orig_title, False, 'polish: not needed (text and title are good)')
                continue
            if verdict == 'unreadable':
                results[i] = _result(orig_body, orig_title, False,
                               'polish: skipped, body is unreadable — a model '
                               'given this would invent rather than repair')
                continue

        if len(orig_body) > POLISH_MAX_CHARS:
            results[i] = _result(orig_body, orig_title, False,
                           f'polish: skipped, body {len(orig_body)} chars is over the '
                           f'{POLISH_MAX_CHARS} cap')
            continue

        to_process.append({
            'idx': i,
            'id': str(i),
            'title': orig_title,
            'body': orig_body,
            'verdict': verdict
        })

    if not to_process:
        return results

    ok, why = llm.available()
    if not ok:
        for p in to_process:
            results[p['idx']] = _result(p['body'], p['title'], False, f'polish: unavailable ({why})')
        return results

    # Batch LLM Call
    payload_json = [{"id": p["id"], "title": p["title"], "body": p["body"]} for p in to_process]
    prompt_payload = json.dumps(payload_json, ensure_ascii=False)
    
    reply_text, reason = llm.chat(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': prompt_payload}],
        temperature=0.0)
        
    if reply_text is None:
        for p in to_process:
            results[p['idx']] = _result(p['body'], p['title'], False, f'polish: call failed ({reason})')
        return results

    # Parse JSON from LLM
    is_json = False
    try:
        reply_json = json.loads(reply_text.strip('` \n').removeprefix('json'))
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

        if cand_body == "[DISCARD]":
            results[i] = _result("[DISCARD]", "[DISCARD]", True, 'polish: applied, article discarded by LLM as a dud', 1.0)
            continue

        accepted, why, sim = check(orig_body, cand_body)
        if not accepted:
            results[i] = _result(orig_body, orig_title, False, f'polish: REJECTED body — {why}', sim)
            continue

        changed = 1.0 - sim
        results[i] = _result(cand_body, cand_title, True,
                       f'polish: applied, {changed:.1%} of body chars changed',
                       sim, {'before': p['verdict'],
                             'after': quality.score(cand_body)['verdict']})

    return results

