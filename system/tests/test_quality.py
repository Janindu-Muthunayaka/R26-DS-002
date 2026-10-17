"""The readability gate.

A capture that goes wrong does not fail — Tesseract returns something, mT5
corrects that something, and the phone reads it aloud in a confident voice. A
sighted developer sees garbage on a screen; a blind user hears fluent nonsense
and cannot tell it from the news. This file is the guard.
"""
from core import quality as q

TRUTH = ('ඉංග්‍රීසි ජාතික ආක්‍රමණිකයන් හා ලක්දිව එකල පැවති එකම නිදහස් රට වූ '
         'සිංහලේ රදළ වරුන් අතර ඇති කර ගත් අවබෝධතා ගිවිසුමකි. කුරුණෑගල නගර '
         'සභාවේ විවිධ සංවර්ධන ව්‍යාපෘති කඩ කාමර බෙදා දීම ඇතුළුව සාකච්ඡා විය.')


def test_real_sinhala_scores_good():
    assert q.score(TRUTH)['verdict'] == 'good'


def test_combining_marks_count_toward_word_length():
    """THE BUG THIS PINS. Sinhala dependent vowel signs are combining marks,
    and `str.isalnum()` is False for every one of them. Measured that way,
    ordinary Sinhala looks like a third fragments — the ground truth files
    scored identically to the worst OCR, which is how it was caught."""
    assert q._glyph_len('ක්‍රියාත්මක') > 5
    assert q.short_token_ratio(TRUTH) < 0.30


def test_zero_characters_is_caught():
    """A real capture in tools/out/cer did this — psm 3 on a single-column
    crop returned nothing. Before the gate the system corrected nothing,
    assembled nothing and read nothing."""
    v, spoken, m = q.verdict_for_user('')
    assert v in ('unreadable', 'short') and spoken and m['n_words'] == 0


def test_latin_garbage_is_caught():
    """`with`, `ikon`, `One`, `ush`, `kinni`, `high` are all real observed
    Tesseract/mT5 outputs on this corpus."""
    bad = 'with ikon One ush kinni high report said today again sure for now'
    assert q.score(bad)['verdict'] == 'unreadable'


def test_fragmented_text_is_caught():
    assert q.score('ක ය ම න ද ව ග ර ත ප ල ස හ ක ය ම න ද')['verdict'] == 'unreadable'


def test_undecodable_characters_are_caught():
    assert q.score(' '.join(['කුරුණෑගල ���'] * 8))['verdict'] == 'unreadable'


def test_short_and_shattered_are_different_failures():
    """A six-word news brief is a real thing a newspaper prints and is read
    with a warning. Fragments are not worth hearing."""
    brief = 'කුරුණෑගල නගර සභාව අද රැස්විය.'
    assert q.verdict_for_user(brief)[0] == 'short'
    assert q.verdict_for_user('ක ය ම න ද ව ග ර ත ප')[0] == 'unreadable'
    assert q.score(brief)['fatal'] is False


def test_a_marginal_read_is_warned_about_not_withheld():
    v, spoken, _ = q.verdict_for_user(
        TRUTH + ' with ikon One ush kinni high report said today')
    assert v in ('poor', 'unreadable')
    assert spoken, 'the user must be told something'


def test_reasons_are_always_populated_when_not_good():
    for bad in ('', 'a b c', 'with ikon One ush kinni high and so on today'):
        m = q.score(bad)
        assert m['verdict'] != 'good' and m['reasons']


def test_score_never_raises():
    for weird in (None, '', '\x00\x01', '𝕘𝕒𝕣𝕓𝕒𝕘𝕖', '123 456 789', ' ' * 50):
        assert 'verdict' in q.score(weird)
