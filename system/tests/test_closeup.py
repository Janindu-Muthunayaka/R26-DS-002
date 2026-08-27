"""The close-up path, tested against the app's real captures.

These are regression tests for a bug that produced NO error message: the
pipeline read the wrong article and reported zero articles. Nothing threw, so
only a test that checks the geometry can catch a repeat.

The real captures live outside the repo, so those tests skip when the folder
is not mounted. The synthetic tests always run.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from layers.l3_segment.closeup import (text_lines, text_bbox, is_closeup,
                                       analyse)
from core.config import CLOSEUP_MIN_P75, CLOSEUP_OCR_SCALE

CAPTURES = Path(os.getenv('R26_CAPTURES', r'F:/App/backend/inbox'))


def _page(glyph_h=44, cols=4, W=2400, H=3200, hand=False, hand_w=180):
    """Synthetic close-up: several columns of bars, optionally with a big dark
    blob at the left edge standing in for a thumb.

    The thumb is placed with a realistic gap to the text. On the app's real
    captures the text bbox starts at x=315-617 while the thumb ends well
    before it, so the gap is comfortably wider than the 25px morphological
    kernel. KNOWN LIMITATION, deliberately not hidden: if a finger actually
    overlaps the text, the kernel bridges them, the merged blob becomes
    page-tall and is rejected on height, and lines ARE lost. See
    test_a_finger_touching_the_text_is_a_known_limitation.
    """
    img = np.full((H, W, 3), 240, np.uint8)
    margin, gut = 280, 60
    colw = (W - 2 * margin - (cols - 1) * gut) // cols
    pitch = int(glyph_h * 1.8)
    for c in range(cols):
        x0 = margin + c * (colw + gut)
        y = 400
        while y + glyph_h < H - 100:
            img[y:y + glyph_h, x0:x0 + colw - 20] = 40
            y += pitch
    if hand:
        img[:, :hand_w] = 30        # a thumb down the left edge
    return img


def test_lines_are_found_on_a_synthetic_closeup():
    lines, med = text_lines(_page())
    assert len(lines) > 20, f'only {len(lines)} lines found'
    assert 30 < med < 70, med


def test_a_hand_at_the_edge_does_not_destroy_detection():
    """The failure that made a page mask unusable: a dark thumb is called ink
    by a global Otsu and welds the frame into one blob. Shape-based detection
    must be indifferent to it."""
    clean, _ = text_lines(_page(hand=False))
    handy, _ = text_lines(_page(hand=True))
    assert len(handy) >= 0.8 * len(clean), (
        f'{len(handy)} lines with a hand vs {len(clean)} without — '
        'the hand is being treated as text')


def test_bbox_crops_the_hand_away():
    img = _page(hand=True)
    H, W = img.shape[:2]
    lines, _ = text_lines(img)
    x1, y1, x2, y2 = text_bbox(lines, W, H)
    assert x1 >= 180, f'crop starts at x={x1}, the thumb reaches 180'
    assert x2 > W * 0.8 and y2 > H * 0.5


def test_a_finger_touching_the_text_is_a_known_limitation():
    """Documents the boundary rather than pretending it is not there. A thumb
    that overlaps the text column merges with it and lines are lost. If this
    ever starts passing, the detector improved and the note above is stale."""
    clean, _ = text_lines(_page(hand=False))
    touching, _ = text_lines(_page(hand=True, hand_w=300))   # into the margin
    assert len(touching) < len(clean), (
        'a finger overlapping the text no longer degrades detection — good '
        'news, but update the docstring in closeup.py')


def test_the_closeup_gate_sits_below_the_whole_article_framing():
    """CHANGED 24 Aug 2026, deliberately, and this test changed with it.

    It used to assert CLOSEUP_MIN_P75 == 28.0 because 28 was the app's
    NEAR_READY value and tying the two together seemed principled. It was not:
    28 is a design choice with no CER behind it, and it REFUSED the only
    framing that holds a whole article.

    Measured: the article first fits at glyph_p75 22, and reading it there
    costs nothing - best mT5 CER 0.0497 against 0.0570 for the close, clipped
    framing on the same 20 lines. So the gate has to sit below 22.

    The test now pins the REASON rather than the number, so raising the gate
    back above the whole-article framing fails loudly.
    """
    assert CLOSEUP_MIN_P75 <= 22.0, (
        'the gate is back above the framing that holds a whole article — '
        'the close-up path will refuse exactly the captures it should read')
    assert CLOSEUP_MIN_P75 >= 16.0, (
        'below this the projection method starts meeting corpus full pages '
        '(measured p75 18-24), where it returns one column out of seven')


def test_the_ocr_scale_is_no_longer_a_fixed_factor():
    """CLOSEUP_OCR_SCALE is superseded and must not creep back into use.

    A fixed 0.40 gives 15 px at glyph_p75 38 and 8.8 px at 22 - and 8.8 px
    measured mT5 CER 0.2193 against 0.0760 for the same frame at 13.2 px.
    What matters is the glyph height Tesseract ends up seeing, so the path
    scales to a target instead.
    """
    from core.config import CLOSEUP_TARGET_GLYPH, CLOSEUP_MIN_GLYPH_PX
    from core.imaging import closeup_scale

    assert CLOSEUP_TARGET_GLYPH >= CLOSEUP_MIN_GLYPH_PX
    # at every distance the path accepts, what reaches Tesseract clears the
    # height at which diacritics disappear
    for p75 in (CLOSEUP_MIN_P75, 22, 25, 28, 33, 38, 45):
        assert p75 * closeup_scale(p75) >= CLOSEUP_MIN_GLYPH_PX - 1e-6, p75
    # and the old constant would NOT have cleared it at the new framing
    assert 22 * CLOSEUP_OCR_SCALE < CLOSEUP_MIN_GLYPH_PX


def test_a_small_glyph_page_is_not_a_closeup():
    """Corpus pages measure p75 19-22 and must NOT take the close-up path."""
    ok, p75 = is_closeup(_page(glyph_h=14))
    assert not ok, f'p75 {p75} wrongly classified as a close-up'


def test_a_large_glyph_page_is_a_closeup():
    ok, p75 = is_closeup(_page(glyph_h=48))
    assert ok, f'p75 {p75} not classified as a close-up'


# ---- against the real captures -------------------------------------------
def _real():
    if not CAPTURES.is_dir():
        pytest.skip(f'captures not mounted at {CAPTURES}')
    files = sorted(CAPTURES.glob('burst_*.jpg'))
    if not files:
        pytest.skip('no burst_*.jpg found')
    return files


def test_every_ACCEPTED_capture_is_detected_as_a_closeup():
    """Only frames Layer 2 accepts need to take the close-up path.

    burst_20260820_105901_g26_s3995_4.jpg is deliberately excluded by that
    rule and it is worth knowing why: sharpness 83.6 against 175 and 487 for
    its two siblings, measured p75 11 against 30 and 32. It is a blurred frame
    that survived on-device selection — the build record's open item 7 — and
    the backend's glyph gate rejects it. Note the sharpness gate (45) would
    NOT have caught it; the p75 gate did.
    """
    from core.imaging import imread_upright, capture_verdict, glyph_p75
    bad, checked = [], 0
    for f in _real():
        img = imread_upright(f)
        if img is None:
            continue
        if capture_verdict(glyph_p75(img))[0] in ('reject', 'unknown'):
            continue                      # L2 drops it before this path
        checked += 1
        info = analyse(img)
        if not info['is_closeup']:
            bad.append(f"{f.name}: p75 {info['glyph_p75']}")
    assert checked >= 5, f'only {checked} frames passed L2'
    assert not bad, ('these would go down the YOLO path, which read the wrong '
                     'article:\n  ' + '\n  '.join(bad))


def test_every_real_capture_yields_text_lines():
    """Measured 21 Aug 2026: 53-73 lines per frame, median height 39-54 px."""
    from core.imaging import imread_upright
    bad = []
    from core.imaging import capture_verdict, glyph_p75
    for f in _real():
        img = imread_upright(f)
        if img is None or capture_verdict(glyph_p75(img))[0] == 'reject':
            continue
        info = analyse(img)
        if not (30 <= info['n_lines'] <= 250):
            bad.append(f"{f.name}: {info['n_lines']} lines")
        elif not (15 <= info['median_line_h'] <= 80):
            bad.append(f"{f.name}: median line height "
                       f"{info['median_line_h']:.0f}px")
    assert not bad, '\n  '.join(bad)


def test_the_crop_actually_removes_something():
    """If the bbox is the whole frame the hand is still in it."""
    from core.imaging import imread_upright
    from core.imaging import capture_verdict, glyph_p75
    kept = []
    for f in _real():
        img = imread_upright(f)
        if img is None or capture_verdict(glyph_p75(img))[0] == 'reject':
            continue
        kept.append(analyse(img)['bbox_frac'])
    assert kept, 'no captures read'
    assert min(kept) < 0.95, 'no frame was cropped at all'
    assert max(kept) <= 1.0
