"""Correction must be fed sentences, not whole articles.

This guards a bug with no symptom. mT5 generates with max_length=128 tokens;
hand it a whole article and the tail is silently never produced. The output
still looks like correct Sinhala — it just stops. It was found and fixed once
in Pipeline v9 (Research Summary 12.5) and came back, because strong_dedup()
joins every line into one string and the caller was splitting on newlines.

Measured on a real capture, 21 Aug 2026: body_raw after dedup contained
1 line but 3 sentences, so exactly one 128-token call was made for the whole
article.
"""
import pytest

from core.textutils import sentences
from core.config import MT5_MAX_LENGTH


def test_the_conditions_that_caused_the_bug():
    """One line, several sentences — splitting on '\\n' gives one unit."""
    art = ("පළමු වාක්‍යය මෙයයි. දෙවන වාක්‍යය මෙයයි. තෙවන වාක්‍යය මෙයයි.")
    assert art.count('\n') == 0
    assert len(art.split('\n')) == 1, 'newline split would give one unit'
    assert len(sentences(art)) == 3, 'sentence split must give three'


def test_no_unit_can_exceed_the_generation_budget():
    """A sentence longer than the budget is split, so nothing is truncated
    without anyone noticing. OCR frequently loses the full stop, which is how
    a single 'sentence' becomes an entire article."""
    runaway = "වචනයක් " * 400            # no terminator at all
    units = sentences(runaway, max_chars=280)
    assert len(units) > 1, 'a terminator-free article was left as one unit'
    assert all(len(u) <= 280 for u in units), \
        f'longest unit {max(len(u) for u in units)} chars'


def test_content_is_preserved():
    """Splitting must not lose text. The dedup stage is allowed to remove
    things; the splitter is not."""
    art = ("කුරුණෑගල නගර සභාව. විවිධ සංවර්ධන ව්‍යාපෘති, කඩ කාමර බෙදා දීම! "
           "අක්‍රමිකතා සිදුව ඇති බවට? ජනතාව කරන චෝදනා.")
    joined = " ".join(sentences(art))
    assert "".join(joined.split()) == "".join(art.split()), \
        'characters were lost or reordered'


def test_terminators_are_kept():
    """The full stop is part of the sentence mT5 was trained on."""
    for u in sentences("එකක්. දෙකක්. තුනක්."):
        assert u.endswith('.'), u


def test_empty_and_whitespace():
    assert sentences('') == []
    assert sentences(None) == []
    assert sentences('   \n  ') == []


def test_a_single_sentence_stays_one_unit():
    s = "මෙය එක් වාක්‍යයක් පමණි."
    assert sentences(s) == [s]


def test_budget_is_below_the_token_limit():
    """280 characters is a proxy for 128 tokens. Sinhala is dense, so the
    proxy is deliberately conservative — if MT5_MAX_LENGTH ever rises, this
    is the place to revisit."""
    assert MT5_MAX_LENGTH == 128
    units = sentences("වචනයක් " * 200, max_chars=280)
    assert max(len(u) for u in units) <= 280
