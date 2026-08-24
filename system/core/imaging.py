"""Shared image measurements. Every layer that needs glyph height uses THIS,
so the number means the same thing everywhere.

TWO different glyph measurements live here and they are NOT interchangeable:

  glyph_p75(frame)  — capture gate. Whole captured frame. "Is this photograph
                      good enough to OCR at all?" Compared against
                      CAPTURE_MIN_GLYPH_P75.

  glyph_p90(crop)   — OCR resize target. A single text-region crop. "How much
                      should this region be scaled before Tesseract sees it?"
                      Compared against OCR_TARGET_GLYPH_P90.

They were a single function against a single constant until 20 Aug 2026, which
made the backend accept 154 of 168 corpus pages while the capture app accepted
8. See core/config.py for the measurements.
"""
import io

import cv2, numpy as np
from PIL import Image, ImageOps

from .config import (CAPTURE_MIN_GLYPH_P75, CAPTURE_REJECT_BELOW_P75,
                     GLYPH_H_MIN, GLYPH_H_MAX,
                     OCR_TARGET_GLYPH_P90, OCR_SCALE_MIN, OCR_SCALE_MAX)


# ==========================================================================
# LOADING — the single place EXIF orientation is handled
# ==========================================================================
# CameraX writes rotation into EXIF and does NOT rotate pixels, so a captured
# frame is only upright if something applies the orientation tag.
#
# Who applies it is library- and version-dependent, which is the actual hazard:
#   * PIL does NOT apply it on Image.open  -- you must call exif_transpose
#   * OpenCV DOES apply it in imread, and in imdecode on current versions
#     (measured: 4.13 applies it in both; IMREAD_IGNORE_ORIENTATION opts out).
#     Older releases did not, and the notebooks that fed PIL images to the
#     detector saw pages turned 90 degrees, which deskew cannot fix because it
#     corrects fractions of a degree, not quarter turns.
#
# Relying on either default is how that bug comes back on a different machine.
# These two functions decode through PIL and apply the transform EXPLICITLY, so
# the result is upright regardless of which OpenCV is installed. Every read of a
# captured image goes through them.

def _pil_to_bgr(im):
    im = ImageOps.exif_transpose(im)
    if im.mode not in ('RGB', 'L'):
        im = im.convert('RGB')
    arr = np.array(im)
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def imread_upright(path):
    """cv2.imread + EXIF orientation applied. Returns BGR, or None."""
    try:
        with Image.open(str(path)) as im:
            return _pil_to_bgr(im)
    except Exception:
        return cv2.imread(str(path))          # non-image or unreadable


def imdecode_upright(data: bytes):
    """cv2.imdecode + EXIF orientation applied. Returns BGR, or None."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return _pil_to_bgr(im)
    except Exception:
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _component_heights(img, hmin=GLYPH_H_MIN, hmax=GLYPH_H_MAX):
    """Connected-component heights, in px.

    THE SPECIFICATION — a percentile of this is meaningless without it:
      * greyscale, global Otsu threshold, ink treated as foreground
      * 8-connectivity
      * components with hmin <= height <= hmax retained; the rest are
        speckle (below) or rules, borders and photo edges (above)
      * measured on whatever image is passed in, at its own scale

    Reproduces the OK/MARGINAL verdict of layout/page_diagnostics.csv on
    165 of 168 corpus pages (98.2%) at the default filter. It does NOT
    reproduce that file's exact values.
    """
    g = to_gray(img)
    if g.size == 0:
        return None
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    h = h[(h >= hmin) & (h <= hmax)]
    return h if len(h) else None


def glyph_percentiles(img, ps=(50, 75, 90)):
    """All three percentiles from one components pass. Use when reporting."""
    h = _component_heights(img)
    if h is None:
        return {f'p{p}': None for p in ps}
    return {f'p{p}': float(np.percentile(h, p)) for p in ps}


def glyph_p75(img):
    """CAPTURE metric — p75 component height on a whole frame, px."""
    h = _component_heights(img)
    return float(np.percentile(h, 75)) if h is not None else None


def glyph_p90(img):
    """OCR-TARGET metric — p90 component height on a text-region crop, px."""
    h = _component_heights(img)
    return float(np.percentile(h, 90)) if h is not None else None


def sharpness(img):
    return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())


def scale_for_target(p90, target=OCR_TARGET_GLYPH_P90):
    """Capped at 1.0 — never upscale. See config for the measurement."""
    if not p90 or p90 <= 0:
        return 1.0
    return float(min(OCR_SCALE_MAX, max(OCR_SCALE_MIN, target / p90)))


def capture_verdict(p75):
    """Accept/reject a captured frame. Argument is glyph_p75, NOT p90."""
    if not p75:
        return 'unknown', 'no text found'
    if p75 < CAPTURE_REJECT_BELOW_P75:
        return 'reject', (f'glyph {p75:.0f}px (need >={CAPTURE_MIN_GLYPH_P75:.0f}) '
                          f'— much closer')
    if p75 < CAPTURE_MIN_GLYPH_P75:
        return 'warn', (f'glyph {p75:.0f}px (want >={CAPTURE_MIN_GLYPH_P75:.0f}) '
                        f'— closer')
    return 'ok', ''


def rescale_to_optimum(img):
    """Returns (resized, p90, scale). The single place rescaling happens.

    p90 here is the OCR-target metric on a region crop — not the capture gate.
    """
    p90 = glyph_p90(img)
    s = scale_for_target(p90)
    if abs(s - 1.0) > 1e-3:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    return img, p90, s
