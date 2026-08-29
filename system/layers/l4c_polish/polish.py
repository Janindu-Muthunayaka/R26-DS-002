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

from core import llm, quality
from core.config import (POLISH_MAX_CHARS, POLISH_MAX_LEN_RATIO,
                         POLISH_MIN_LEN_RATIO, POLISH_MIN_SIMILARITY,
                         POLISH_MODE)

SYSTEM_PROMPT = (
    'You repair optical-character-recognition errors in Sinhala newspaper '
    'text. You are not an assistant and you are not a writer.\n'
    'RULES, in order of importance:\n'
    '1. Output ONLY the repaired text. No preamble, no explanation, no '
    'quotes, no markdown.\n'
    '2. NEVER add information. No names, numbers, dates, places or clauses '
    'that are not already present.\n'
    '3. NEVER remove, summarise, reorder, translate or explain anything.\n'
    '4. Repair only what is clearly an OCR error: a broken word, a wrong '
    'dependent vowel sign, a split or merged word, a stray Latin fragment '
    'where Sinhala belongs.\n'
    '5. If a word is unrecoverable, LEAVE IT EXACTLY AS IT IS. A word you '
    'cannot read is not a word you may guess.\n'
    '6. Output Sinhala script. Leave digits and genuine Latin proper nouns '
    'as they are.\n'
    'The output should be almost identical to the input in length and in '
    'sentence order.'
)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _result(text, applied, reason, similarity=None, measures=None) -> dict:
    return {'text': text, 'applied': bool(applied), 'reason': reason,
            'similarity': similarity, 'measures': measures or {}}


def check(original: str, candidate: str) -> tuple:
    """Run the four guards. Returns (accepted, reason, similarity).

    Separated from the network call so it can be tested exhaustively without
    one — which matters, because these four comparisons are the entire defence
    against a model inventing the news.
    """
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


def polish(text: str, mode: str = None) -> dict:
    """Post-edit `text`. Returns a result dict; never raises.

    `applied` False means the caller must keep its own text. In every such
    case `text` in the result is the ORIGINAL, so a caller that ignores
    `applied` still cannot accidentally use an unvetted rewrite.
    """
    original = (text or '').strip()
    mode = (mode or POLISH_MODE or 'off').lower()

    if mode == 'off':
        return _result(original, False, 'polish: off')
    if not original:
        return _result(original, False, 'polish: nothing to repair')

    verdict = quality.score(original)['verdict']
    if mode == 'auto':
        if verdict == 'good':
            return _result(original, False, 'polish: not needed (text is good)')
        if verdict == 'unreadable':
            return _result(original, False,
                           'polish: skipped, text is unreadable — a model '
                           'given this would invent rather than repair')

    if len(original) > POLISH_MAX_CHARS:
        return _result(original, False,
                       f'polish: skipped, {len(original)} chars is over the '
                       f'{POLISH_MAX_CHARS} cap')

    ok, why = llm.available()
    if not ok:
        return _result(original, False, f'polish: unavailable ({why})')

    candidate, reason = llm.chat(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': original}],
        temperature=0.0)
    if candidate is None:
        return _result(original, False, f'polish: call failed ({reason})')

    accepted, why, sim = check(original, candidate)
    if not accepted:
        return _result(original, False, f'polish: REJECTED — {why}', sim)

    changed = 1.0 - sim
    return _result(candidate.strip(), True,
                   f'polish: applied, {changed:.1%} of characters changed',
                   sim, {'before': verdict,
                         'after': quality.score(candidate)['verdict']})
