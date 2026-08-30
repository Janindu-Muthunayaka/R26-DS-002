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
    'You will receive a JSON object containing "title" and "body".\n'
    'RULES, in order of importance:\n'
    '1. Output ONLY a valid JSON object with keys "title" and "body". No preamble, no markdown, no quotes.\n'
    '2. NEVER add information to the body. No names, numbers, dates, places or clauses that are not already present.\n'
    '3. NEVER remove, summarise, reorder, translate or explain anything in the body.\n'
    '4. Repair only what is clearly an OCR error in the body: a broken word, a wrong dependent vowel sign, a split/merged word, a stray Latin fragment.\n'
    '5. If the title is empty or missing, USE the context of the body to GENERATE an appropriate short Sinhala headline for the "title" field.\n'
    '6. If the title is present, repair its OCR errors using the context of the body.\n'
    '7. Output Sinhala script. Leave digits and genuine Latin proper nouns as they are.\n'
    'The output body should be almost identical to the input body in length and in sentence order.'
)


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


def polish_article(article: Article, mode: str = None) -> dict:
    """Post-edit title and body. Returns a result dict; never raises."""
    mode = (mode or POLISH_MODE or 'off').lower()
    
    orig_title = (article.title or article.title_raw or '').strip()
    orig_body = (article.body or article.body_raw or '').strip()

    if mode == 'off':
        return _result(orig_body, orig_title, False, 'polish: off')

    if _is_nonsense(orig_body) and _is_nonsense(orig_title):
        return _result(orig_body, orig_title, False, 'polish: skipped, title and body are empty or nonsense')

    verdict = quality.score(orig_body)['verdict']
    if mode == 'auto':
        if verdict == 'good' and not _is_nonsense(orig_title):
            return _result(orig_body, orig_title, False, 'polish: not needed (text and title are good)')
        if verdict == 'unreadable':
            return _result(orig_body, orig_title, False,
                           'polish: skipped, body is unreadable — a model '
                           'given this would invent rather than repair')

    if len(orig_body) > POLISH_MAX_CHARS:
        return _result(orig_body, orig_title, False,
                       f'polish: skipped, body {len(orig_body)} chars is over the '
                       f'{POLISH_MAX_CHARS} cap')

    ok, why = llm.available()
    if not ok:
        return _result(orig_body, orig_title, False, f'polish: unavailable ({why})')

    prompt_payload = json.dumps({"title": orig_title, "body": orig_body}, ensure_ascii=False)
    
    reply_text, reason = llm.chat(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': prompt_payload}],
        temperature=0.0)
        
    if reply_text is None:
        return _result(orig_body, orig_title, False, f'polish: call failed ({reason})')

    # Parse JSON from LLM
    try:
        reply_json = json.loads(reply_text.strip('` \n').removeprefix('json'))
        cand_title = str(reply_json.get('title', '')).strip()
        cand_body = str(reply_json.get('body', '')).strip()
    except json.JSONDecodeError:
        return _result(orig_body, orig_title, False, 'polish: call failed (LLM did not return valid JSON)')

    # Check the body for safety (hallucination guards)
    # If the original body was mostly empty/nonsense, we might fail the similarity guard if the LLM expanded it.
    # But wait, we already rejected if BOTH title and body are empty/nonsense.
    # If the original body is not empty, run the check on it.
    accepted, why, sim = check(orig_body, cand_body)
    if not accepted:
        return _result(orig_body, orig_title, False, f'polish: REJECTED body — {why}', sim)

    changed = 1.0 - sim
    return _result(cand_body, cand_title, True,
                   f'polish: applied, {changed:.1%} of body chars changed',
                   sim, {'before': verdict,
                         'after': quality.score(cand_body)['verdict']})

