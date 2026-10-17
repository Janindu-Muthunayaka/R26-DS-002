"""
Is this text worth reading aloud?

THE PROBLEM THIS SOLVES. When a capture goes wrong the pipeline does not fail
— Tesseract returns *something*, mT5 corrects that something, and the phone
reads the result out in a confident voice. A sighted developer at a desk sees
the garbage on screen. A blind user hears fluent nonsense and has no way to
tell it apart from the news.

So: score the text, and when it is bad, SAY SO instead of reading it.

NO MODEL, NO NETWORK. Four cheap surface statistics.

  sinhala_ratio    Of the letters, how many are Sinhala? Tesseract failing on
                   Sinhala emits Latin fragments — `with`, `ikon`, `One`,
                   `ush`, `kinni`, `high` are all real observed outputs.
  short_ratio      Fraction of tokens of one or two characters. Broken glyph
                   segmentation shatters words into fragments.
  replacement      U+FFFD and stray control characters per 1000 chars.
  n_words          A capture that read four words did not read an article.

MEASURED, 27 Aug 2026, on this project's own outputs — the sweep in
`tools/out/sweep/`, the framing comparison in `tools/out/cer/`, and
`Work/Ishara/article_truth*.txt` as known-good. Reproduce with
`python tools/calibrate_quality.py`.

                        sinhala_ratio      short_ratio     words
    ground truth  n=2   1.000              0.169 - 0.179   112 - 142
    OCR raw       n=19  1.000              0.168 - 0.252   112 - 151
    after mT5     n=19  0.951 - 1.000      0.152 - 0.209   105 - 147

READ THAT TABLE HONESTLY. On these files the OCR is never catastrophic, so the
two continuous measures barely separate anything: `short_ratio` overlaps
ground truth completely. The thresholds below therefore sit ABOVE the whole
observed range on purpose — guards against failure modes this sample does not
contain, not a fitted boundary.

The measure that earns its place is `n_words`. One real capture,
`14f7798c_auto_raw.txt`, returned **zero characters** — psm 3 on a
single-column crop — and before this file existed the system would have gone on
to correct nothing, assemble nothing and read nothing.

A hint that is better than nothing, which is what the system had. NOT a
validated detector, and not a result.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Tuple

SINHALA = (0x0D80, 0x0DFF)

THRESHOLDS = {
    'sinhala_ratio_poor': 0.85,
    'sinhala_ratio_bad': 0.70,
    'short_ratio_poor': 0.30,
    'short_ratio_bad': 0.45,
    'replacement_per_1k_poor': 4.0,
    'replacement_per_1k_bad': 12.0,
    'min_words': 8,
}

_TOKEN = re.compile(r'\S+')


def sinhala_ratio(text: str) -> float:
    letters = [c for c in text or '' if c.isalpha()]
    if not letters:
        return 0.0
    n = sum(1 for c in letters if SINHALA[0] <= ord(c) <= SINHALA[1])
    return n / len(letters)


def _glyph_len(token: str) -> int:
    """Length of a token in script characters.

    NOT `isalnum()`. Sinhala dependent vowel signs (*pilla*) are combining
    marks, Unicode category Mn, and `isalnum()` is False for every one of them.
    Measured that way, "ක්‍රියාත්මක" counts as 5 rather than 10 and a third of
    ordinary Sinhala looks like fragments — the ground truth files scored
    identically to the 11 px OCR, which is how this was caught.
    """
    return sum(1 for c in token
               if unicodedata.category(c)[:1] in ('L', 'M', 'N'))


def short_token_ratio(text: str) -> float:
    toks = _TOKEN.findall(text or '')
    if not toks:
        return 1.0
    lens = [_glyph_len(t) for t in toks]
    return sum(1 for L in lens if L <= 2) / len(lens)


def replacement_per_1k(text: str) -> float:
    t = text or ''
    if not t:
        return 0.0
    bad = sum(1 for c in t
              if c == '�'
              or (unicodedata.category(c) == 'Cc' and c not in '\n\r\t'))
    return 1000.0 * bad / len(t)


def word_count(text: str) -> int:
    return len(_TOKEN.findall(text or ''))


def score(text: str) -> dict:
    """Every measure, plus a verdict and a reason. Never raises."""
    t = (text or '').strip()
    m = {
        'n_chars': len(t),
        'n_words': word_count(t),
        'sinhala_ratio': round(sinhala_ratio(t), 4),
        'short_ratio': round(short_token_ratio(t), 4),
        'replacement_per_1k': round(replacement_per_1k(t), 2),
    }
    T = THRESHOLDS
    reasons_bad, reasons_poor = [], []

    if m['n_words'] < T['min_words']:
        reasons_bad.append(f"only {m['n_words']} words were read")
    if m['sinhala_ratio'] < T['sinhala_ratio_bad']:
        reasons_bad.append(f"{1 - m['sinhala_ratio']:.0%} of the letters are "
                           f"not Sinhala")
    elif m['sinhala_ratio'] < T['sinhala_ratio_poor']:
        reasons_poor.append('some words came out in Latin letters')

    if m['short_ratio'] > T['short_ratio_bad']:
        reasons_bad.append(f"{m['short_ratio']:.0%} of the words are fragments")
    elif m['short_ratio'] > T['short_ratio_poor']:
        reasons_poor.append('many words are broken into fragments')

    if m['replacement_per_1k'] > T['replacement_per_1k_bad']:
        reasons_bad.append('the text contains undecodable characters')
    elif m['replacement_per_1k'] > T['replacement_per_1k_poor']:
        reasons_poor.append('a few characters could not be decoded')

    if reasons_bad:
        m['verdict'], m['reasons'] = 'unreadable', reasons_bad
    elif reasons_poor:
        m['verdict'], m['reasons'] = 'poor', reasons_poor
    else:
        m['verdict'], m['reasons'] = 'good', []

    # A SHORT article and a SHATTERED one are not the same failure, and only
    # one of them justifies withholding the text. A six-word news brief is a
    # real thing a newspaper prints; shattered script is not worth hearing.
    m['fatal'] = bool(reasons_bad) and any(
        not r.startswith('only ') for r in reasons_bad)
    return m


# Spoken when the text is not worth reading. NOT written by a native speaker.
SI_UNREADABLE = ('මෙම ඡායාරූපයෙන් පැහැදිලිව කියවිය නොහැක. '
                 'කරුණාකර නැවත උත්සාහ කරන්න.')
SI_POOR = 'මෙය පැහැදිලිව කියවා නොමැත.'


def verdict_for_user(text: str) -> Tuple[str, str, dict]:
    """(verdict, a sentence to speak or '', the measures).

    'unreadable' returns a sentence INSTEAD of the text. 'poor' returns a
    prefix to speak BEFORE it — the listener hears the caveat and then the
    text, and decides. Withholding a marginal read would be worse.
    """
    m = score(text)
    if m['verdict'] == 'unreadable':
        return ('unreadable' if m['fatal'] else 'short'), SI_UNREADABLE, m
    if m['verdict'] == 'poor':
        return 'poor', SI_POOR, m
    return 'good', '', m
