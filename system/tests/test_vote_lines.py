"""Multi-frame consensus.

The bug this fixes, measured on work/80654199 (three frames, one static
scene): the frames produced 105, 106 and 101 lines, the voter compared them
by INDEX, and 15 of 100 output lines came back near-duplicates of an earlier
line - whole passages read aloud twice - while 235 characters were lost.
"""
import difflib
import pytest

from core.textutils import vote_lines


def _dupes(text, ratio=0.80, minlen=8):
    lines = [l for l in text.split('\n') if len(l.strip()) >= minlen]
    n, seen = 0, []
    for l in lines:
        k = ''.join(l.split())
        if any(difflib.SequenceMatcher(None, k, s).ratio() >= ratio for s in seen):
            n += 1
        seen.append(k)
    return n


# deliberately dissimilar lines: a fixture whose lines all look alike cannot
# tell a real repeat from two lines that merely share a common word
A = ['කුරුණෑගල නගර සභාව', 'විගණනයක් කළ යුතු බව', 'ටෙන්ඩර් ප්‍රදානය කිරීම',
     'බැංකු ණය මගින් සපයාගෙන', 'මහජන නියෝජිතයන් විසින්']


def test_one_frame_passes_through():
    assert vote_lines(['\n'.join(A)]) == '\n'.join(A)


def test_identical_frames_are_unchanged():
    assert vote_lines(['\n'.join(A)] * 3) == '\n'.join(A)


def test_a_frame_with_an_EXTRA_line_does_not_cause_repeats():
    """THE BUG. One frame splits a line, so every later index is offset by
    one. Index-based voting then compared unrelated lines and emitted the
    same passage twice."""
    b = A[:1] + ['විගණනයක්', 'කළ යුතු බව'] + A[2:]   # 6 lines, not 5
    out = vote_lines(['\n'.join(A), '\n'.join(b), '\n'.join(A)])
    assert _dupes(out) == 0, out
    assert out.split('\n')[0] == A[0]
    assert out.split('\n')[-1] == A[-1]


def test_a_frame_with_a_MISSING_line_does_not_reorder_the_rest():
    c = A[:2] + A[3:]                                # 4 lines, not 5
    out = vote_lines(['\n'.join(A), '\n'.join(c), '\n'.join(A)]).split('\n')
    assert out == A, out


def test_a_corrupted_line_is_outvoted_by_the_other_two_frames():
    bad = list(A); bad[2] = 'ZZZZ ###### ~~~~'
    out = vote_lines(['\n'.join(A), '\n'.join(bad), '\n'.join(A)]).split('\n')
    assert out[2] == A[2]


def test_the_reference_line_count_is_preserved():
    """The other frames may correct a line; they may never insert or drop
    one. That is what stops the output from being reordered."""
    b = A + ['නගර සභාවේ හිටපු මන්ත්‍රී']
    c = A[:3]
    out = vote_lines(['\n'.join(A), '\n'.join(b), '\n'.join(c)]).split('\n')
    assert len(out) == len(A)          # A is the median length


def test_empty_input():
    assert vote_lines([]) == ''
    assert vote_lines(['', '  ']) == ''
