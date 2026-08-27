"""Which headline belongs to which article.

An article is a headline plus the body under it. The body was already
isolated; until 27 Aug 2026 the headline was located with a threshold that
could not tell a headline from a tall body line, and was never read.

MEASURED on the nine captures in F:/App/backend/inbox
(`python tools/measure_headline.py`):

    tallest BODY line      1.28x - 1.70x of the median line height
    tallest HEADLINE band  5.91x - 8.71x

The old threshold was 1.6 — inside the body range.
"""
import numpy as np
import pytest

pytest.importorskip('cv2')
import cv2  # noqa: E402

from core.config import (TITLE_MAX_GAP_LINES, TITLE_MIN_LINE_RATIO,
                         TITLE_MIN_X_OVERLAP)
from layers.l3_segment import closeup as C


def _page(bands, W=1200, H=1600):
    """A synthetic page: (y0, h, x0, w) filled black on white."""
    img = np.full((H, W, 3), 255, np.uint8)
    for y0, h, x0, w in bands:
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    return img


def test_the_threshold_sits_in_the_measured_gap():
    """1.70x was the tallest body line and 5.91x the shortest headline on the
    real captures. A threshold outside that gap misclassifies one or the
    other, and 1.6 — the previous value — misclassified every capture's
    tallest body line as a headline."""
    assert 1.70 < TITLE_MIN_LINE_RATIO < 5.91


def test_a_tall_body_line_is_not_a_headline():
    med = 40.0
    img = _page([(100, int(med * 1.7), 100, 900)])      # tallest body seen
    assert C.headline_bands(img, med) == []


def test_a_real_headline_is_found():
    med = 40.0
    img = _page([(100, int(med * 6), 100, 900)])        # 6x, a real headline
    assert len(C.headline_bands(img, med)) == 1


def test_headline_is_attached_to_the_body_below_it():
    med = 40.0
    img = _page([(100, 240, 100, 900),                  # headline
                 (400, 30, 100, 900)])                  # body starts
    box = C.headline_for_block(img, (400, 1200), (100, 1000), med)
    assert box is not None
    x0, y0, x1, y1 = box
    assert y1 < 400 and y0 >= 90


def test_a_masthead_far_above_the_body_is_refused():
    """The real failure this exists for: these pages carry a masthead, a page
    number and a section strip above the headline, all headline-sized.
    Measured, a real headline sits 36-97 px above its body while the masthead
    is 186-225 px above what follows it."""
    med = 40.0
    far = int(med * TITLE_MAX_GAP_LINES) + 60
    img = _page([(100, 240, 100, 900),                  # masthead, far above
                 (100 + 240 + far, 30, 100, 900)])      # body
    assert C.headline_for_block(img, (100 + 240 + far, 1200),
                                (100, 1000), med) is None


def test_a_headline_over_the_gutter_is_refused():
    """A page number or a masthead logo sits above a gutter, not above the
    article's own columns."""
    med = 40.0
    img = _page([(100, 240, 60, 300),      # far left, off the block
                 (400, 30, 700, 400)])     # body columns are on the right
    assert C.headline_for_block(img, (400, 1200), (700, 1100), med) is None


def test_side_by_side_headline_boxes_merge_into_one_row():
    """A two-column headline gives two boxes at the same height, and the words
    of one headline line are far enough apart that the morphological close
    does not join them. Sorting by y then gives NEGATIVE gaps — measured -227,
    -260, -329 on real captures — which no gap rule can read."""
    rows = C._merge_rows([(100, 340, 0, 500), (110, 350, 600, 1100)],
                         join_gap=40)
    assert len(rows) == 1
    assert rows[0][2] == 0 and rows[0][3] == 1100


def test_two_separated_groups_do_not_merge():
    rows = C._merge_rows([(100, 340, 0, 500), (900, 1140, 0, 500)],
                         join_gap=40)
    assert len(rows) == 2


def test_no_headline_means_no_title_not_a_wrong_one():
    med = 40.0
    img = _page([(400, 30, 100, 900)])                  # body only
    assert C.headline_for_block(img, (400, 1200), (100, 1000), med) is None


def test_constants_are_the_measured_ones():
    assert TITLE_MIN_LINE_RATIO == 3.0
    assert TITLE_MAX_GAP_LINES == 3.0
    assert 0.0 < TITLE_MIN_X_OVERLAP <= 1.0


def test_layer_4a_is_off_by_default_and_the_signature_is_unchanged():
    from core.schemas import Article, Box
    from layers.l4a_title import title as l4a
    a = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1))
    assert l4a.extract(None, a) is a          # the contract every caller uses
    assert a.title == ''


# ---- the headline as a hard article boundary ----------------------------
def test_a_headline_splits_one_block_into_two_articles():
    """`blocks()` splits on white gaps only, and says so in its own docstring:
    it "does NOT distinguish 'next article' from 'sub-heading'". A headline is
    INK, so on a tight page the gap before the next story is not wide enough
    to split and the crop carries two articles.

    MEASURED on the 70 real captures in system/work: 13 of the 48 that reached
    the column crop had a headline INSIDE it. After this split: 0.
    """
    from layers.l3_segment.layout import split_at_headlines
    # one block, with a headline crossing it at y500-600
    out = split_at_headlines([(100, 1000)], [(500, 600, 0, 1000)],
                             band_x=(0, 1000), min_pitch=1.0, p=50.0)
    assert len(out) == 2
    assert out[0] == (100, 500), 'the story above ends at the headline'
    assert out[1] == (500, 1000), 'the headline opens the story below'


def test_the_headline_goes_with_the_article_below_it():
    from layers.l3_segment.layout import split_at_headlines
    out = split_at_headlines([(0, 1000)], [(400, 500, 0, 1000)],
                             band_x=(0, 1000), min_pitch=1.0, p=50.0)
    assert out[1][0] == 400, 'the headline belongs to the article it heads'


def test_a_headline_in_another_column_does_not_split_this_article():
    """A headline off to one side heads a story in a different column."""
    from layers.l3_segment.layout import split_at_headlines
    out = split_at_headlines([(100, 1000)], [(500, 600, 1400, 1900)],
                             band_x=(0, 1000), min_pitch=1.0, p=50.0)
    assert out == [(100, 1000)]


def test_no_headline_leaves_the_blocks_alone():
    from layers.l3_segment.layout import split_at_headlines
    assert split_at_headlines([(0, 900)], [], (0, 900), 1.0, 50.0) == [(0, 900)]


def test_a_sliver_left_by_a_split_is_dropped():
    """A split that leaves less than min_pitch x pitch of text is not an
    article, it is the tail of the one above."""
    from layers.l3_segment.layout import split_at_headlines
    out = split_at_headlines([(100, 1000)], [(120, 200, 0, 1000)],
                             band_x=(0, 1000), min_pitch=1.0, p=50.0)
    assert all(b[1] - b[0] >= 50 for b in out)


def test_the_layout_gate_is_below_the_closeup_gate():
    """Layout is tried FIRST and YOLO only when it refuses. Measured: the old
    is_closeup gate sent 29% of real captures to the article detector, and
    layout succeeds on 16 of those 20. Corpus full pages are still refused by
    the gutter gate with the p75 gate off entirely (12 of 12)."""
    from core.config import CLOSEUP_MIN_P75, LAYOUT_MIN_P75
    assert LAYOUT_MIN_P75 < CLOSEUP_MIN_P75
    assert LAYOUT_MIN_P75 > 0, 'a frame with no readable text must still be refused'
