from core.textutils import strong_dedup, collapse_repeats, vote_lines, norm


def test_dedup():
    assert strong_dedup('abc def abc def ghi') == 'abc def ghi'
    assert collapse_repeats('aa bb aa bb') == 'aa bb'


def test_vote_picks_majority():
    assert vote_lines(['x y', 'x y', 'z w']) == 'x y'
    assert vote_lines([]) == ''


def test_norm_nfc():
    import unicodedata
    s = unicodedata.normalize('NFD', '\u0dda')
    assert norm(s) == '\u0dda'
