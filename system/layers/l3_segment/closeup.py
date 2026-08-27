"""
LAYER 3 (close-up variant) — the phone path.  OWNER: Ishara.

WHY THIS EXISTS, measured 21 Aug 2026
-------------------------------------
The capture app's READY band puts the phone far closer than anything the
article detector was trained on. Corpus half-pages measure glyph_p75 median
22; the app's own captures measure 33. At that distance the frame IS one
article, cut off on every side.

On `burst_20260820_105855_g27_s3615_4.jpg` the detector did not simply miss
it. It returned ONE confident box, (0,2692)-(2172,3264), covering the
NEIGHBOURING article's headline along the bottom edge — and because a box was
returned, l3_segment's `if not boxes` whole-frame fallback never triggered.
The pipeline then rescaled that strip by 0.19 and Tesseract produced 0
characters. It read the wrong article, badly, and reported no error.

So this path does not segment articles at all. It finds the text, crops away
everything that is not text — a hand at the frame edge, the page margin, a
neighbouring headline — and hands Tesseract one multi-column crop with the
page-segmentation mode that actually handles columns.

WHAT IS MEASURED AND WHAT IS NOT
--------------------------------
  measured   line detection: 53-73 text lines on all nine of the app's
             captures, median line height 39-54 px
  measured   psm 6 splices adjacent columns together mid-sentence; psm 3
             preserves column order (see core/config.py)
  by eye     CLOSEUP_OCR_SCALE = 0.40 beat 0.6 and 1.0 on a 220-character
             sample. There is no ground truth for that page, so this is a
             judgement, not a CER measurement. It is stated as such.
  partial    headlines. title_lines() locates them, but reading them is
             Layer 4A, which belongs to another team member and is a stub.
             So the close-up path emits a title REGION and no title TEXT.
             `article.title` stays empty until that layer is delivered — the
             system says so rather than silently dropping the headline.
"""
import cv2
import numpy as np

from core.config import (CLOSEUP_MIN_P75, GLYPH_H_MIN, GLYPH_H_MAX,
                         TITLE_MIN_LINE_RATIO, TITLE_MAX_GAP_LINES,
                         TITLE_ROW_JOIN_LINES, TITLE_MIN_X_OVERLAP)
from core.imaging import glyph_p75


# --------------------------------------------------------------------------
def text_lines(img, kern_w=25):
    """Text lines found by SHAPE, deliberately not by brightness.

    A page mask is the obvious approach and it fails here: a thumb at the
    frame edge is dark, so a global Otsu calls it ink, and it welds the whole
    frame into one connected blob. Measured on a real capture: 36% of the
    frame was classified as ink and every contour merged into one.

    A text line, by contrast, has a shape — wider than tall, a bounded
    height, not a speck. A hand fails all three tests without anyone needing
    to know it is a hand.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_w, 3))
    cs, _ = cv2.findContours(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cs]
    cand = [r for r in boxes
            if r[2] > 2 * r[3]              # wider than tall
            and GLYPH_H_MIN < r[3] < H * 0.10   # a line, not a page-tall blob
            and r[2] > W * 0.03]            # not a speck
    if not cand:
        return [], 0.0
    med = float(np.median([r[3] for r in cand]))
    return [r for r in cand if 0.45 * med <= r[3] <= 3.0 * med], med


def text_bbox(lines, W, H, pad=20):
    """Bounding box of the detected text.

    This is what removes the hand, the margin and the neighbouring article —
    with no brightness heuristic and no assumption about where the hand is.
    Measured across the nine captures it trimmed 315-617 px from the left on
    six frames and from the right on the other three, purely from where the
    text happened to be.
    """
    if not lines:
        return 0, 0, W, H
    return (max(0, min(r[0] for r in lines) - pad),
            max(0, min(r[1] for r in lines) - pad),
            min(W, max(r[0] + r[2] for r in lines) + pad),
            min(H, max(r[1] + r[3] for r in lines) + pad))


def _merge_rows(bands, join_gap):
    """Merge headline bands that belong to the same LINE of a headline.

    A headline line is not one contour. It is several: the words are far
    enough apart that the morphological close does not join them, and a
    two-column headline produces two side-by-side boxes at the same height.
    Sorting by y alone gives a sequence with NEGATIVE gaps — measured -227,
    -260, -329 on real captures — which no gap rule can read.

    So: merge anything that overlaps vertically or nearly touches, then reason
    about the gaps BETWEEN the merged rows. That is the sequence a reader
    sees.
    """
    if not bands:
        return []
    bands = sorted(bands, key=lambda b: b[0])
    rows = [list(bands[0])]
    for y0, y1, x0, x1 in bands[1:]:
        if y0 - rows[-1][1] <= join_gap:          # overlaps or nearly touches
            rows[-1][1] = max(rows[-1][1], y1)
            rows[-1][2] = min(rows[-1][2], x0)
            rows[-1][3] = max(rows[-1][3], x1)
        else:
            rows.append([y0, y1, x0, x1])
    return [tuple(r) for r in rows]


def headline_bands(img, med_line_h, min_ratio=TITLE_MIN_LINE_RATIO):
    """Every headline-sized band in the frame, at any height.

    THE THRESHOLD IS MEASURED, 27 Aug 2026, on the nine captures in
    `F:/App/backend/inbox` (`tools/measure_headline.py` reproduces it):

        tallest BODY line      1.28x - 1.70x of the median line height
        tallest HEADLINE band  5.91x - 8.71x

    Nothing lands between 1.70 and 5.91. The old value here was **1.6**, which
    sits INSIDE the body range — so the tallest body line of every capture was
    being reported as a headline. 3.0 is the middle of the empty gap.

    Unlike the previous version this does NOT restrict the search to above the
    body: the next article's headline below the block is exactly what article
    isolation has to see.
    """
    if med_line_h <= 0:
        return []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 5))
    cs, _ = cv2.findContours(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for x, y, w, h in (cv2.boundingRect(c) for c in cs):
        if (h >= med_line_h * min_ratio
                and h < H * 0.25              # not a page-tall blob
                and w > W * 0.15):            # a headline spans some width
            out.append((y, y + h, x, x + w))
    return sorted(out)


def headline_for_block(img, block, block_x, med_line_h):
    """The headline belonging to THIS article, or None.

    An article is a headline plus the body under it. The body block is already
    isolated; this finds the headline that belongs to it, and REFUSES when it
    cannot tell.

    Three tests, all of which must pass. Each rejects something seen on a real
    capture:

      gap       the headline sits directly on its body. Measured 36-97 px on
                eight captures, against 186-225 px separating the masthead and
                the section strip from the headline below them. The bound is
                TITLE_MAX_GAP_LINES x the median line height.
      x-overlap the headline sits above its OWN columns. The page number and
                the masthead logo sit above the gutter or off to one side.
      one group the rows immediately above the body must form ONE contiguous
                group. Two groups means something is between them, and this
                cannot tell which one is the headline.

    REFUSING IS THE POINT. `burst_20260820_105901_g26_s3995_2.jpg` has the
    masthead, a page number and the section strip "ප්‍රාදේශීය පුවත්" stacked
    above the real headline; reading those to a listener as the headline would
    be worse than saying nothing. When this returns None the article is read
    without a headline, exactly as before.
    """
    if med_line_h <= 0 or not block:
        return None
    by0 = block[0]
    bands = [b for b in headline_bands(img, med_line_h) if b[1] <= by0]
    if not bands:
        return None

    rows = _merge_rows(bands, join_gap=med_line_h * TITLE_ROW_JOIN_LINES)
    if not rows:
        return None

    max_gap = med_line_h * TITLE_MAX_GAP_LINES
    if by0 - rows[-1][1] > max_gap:
        return None                       # nothing sits on this body

    group = [rows[-1]]
    for r in reversed(rows[:-1]):
        if group[-1][0] - r[1] <= max_gap:
            group.append(r)
        else:
            break

    y0 = min(r[0] for r in group); y1 = max(r[1] for r in group)
    x0 = min(r[2] for r in group); x1 = max(r[3] for r in group)

    bx0, bx1 = block_x
    overlap = max(0, min(x1, bx1) - max(x0, bx0))
    if overlap < (x1 - x0) * TITLE_MIN_X_OVERLAP:
        return None                       # above the gutter, not the article

    return (x0, y0, x1, y1)


def title_lines(img, lines, med_h, min_ratio=TITLE_MIN_LINE_RATIO):
    """Lines much taller than the body — the headline.

    GEOMETRY ONLY. This finds WHERE the title is and stops there. Title OCR is
    Layer 4A and belongs to another team member; this exists so that when their
    layer is delivered it has a region to work on, and so the close-up path is
    not silently dropping the headline without saying so.

    text_lines() rejects headlines on height (they exceed 3x the body median),
    so they never appear in `lines`. They are recovered here with a separate,
    looser pass.
    """
    if med_h <= 0:
        return []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 5))
    cs, _ = cv2.findContours(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_top = min((r[1] for r in lines), default=H)
    out = []
    for r in (cv2.boundingRect(c) for c in cs):
        x, y, w, h = r
        if (h >= med_h * min_ratio            # taller than body text
                and h < H * 0.25              # not a page-tall blob
                and w > W * 0.15              # a headline spans some width
                and y < max(body_top, H * 0.4)):   # above or level with body
            out.append(r)
    return sorted(out, key=lambda r: r[1])


def is_closeup(img, threshold=CLOSEUP_MIN_P75):
    """Is this frame a close-up of a single article?

    Decided by the capture app's own READY threshold rather than by a new
    tuned constant. The auto-shutter only fires in the READY band (p75 28-40),
    so every phone capture satisfies this by construction, and no corpus page
    does — corpus full pages measure p75 median 19, half pages 22.
    """
    p75 = glyph_p75(img)
    return (p75 is not None and p75 >= threshold), p75


def analyse(img):
    """Everything the close-up path needs, in one pass over the frame."""
    H, W = img.shape[:2]
    closeup, p75 = is_closeup(img)
    lines, med_h = text_lines(img)
    x1, y1, x2, y2 = text_bbox(lines, W, H)
    titles = title_lines(img, lines, med_h)
    tb = text_bbox(titles, W, H, pad=10) if titles else None
    return {
        'is_closeup': closeup,
        'glyph_p75': p75,
        'n_lines': len(lines),
        'median_line_h': med_h,
        'bbox': (x1, y1, x2, y2),
        'bbox_frac': ((x2 - x1) * (y2 - y1)) / float(W * H) if W and H else 0.0,
        'n_title_lines': len(titles),
        'title_bbox': tb,
    }
