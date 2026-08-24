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

from core.config import CLOSEUP_MIN_P75, GLYPH_H_MIN, GLYPH_H_MAX
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


def title_lines(img, lines, med_h, min_ratio=1.6):
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
