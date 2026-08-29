"""
LAYER 4A — title extraction.

An article is a headline plus the body under it. Until 27 Aug 2026 this
returned the article unchanged, so the system read the body and silently
dropped the headline — the part a listener uses to decide whether to keep
listening.

WHAT IS MEASURED. Locating the headline is
`l3_segment/closeup.headline_for_block()`; the constants behind it are
measured and their provenance is in core/config.py. This file only READS the
region it is given.

Measured on the nine captures in `F:/App/backend/inbox`
(`python tools/measure_headline.py --ocr`):

  * `sin_raw` with **psm 11** reads a well-framed headline near-correctly —
    'කුරුණෑගල නගර විගණනයක් ලබා' against a true
    කුරුණෑගල නගර සභාවේ ... විගණනයක් ලබා දෙන්න.
  * `sin_custom` is garbage on raw pixels. It is Janindu's **MAT** model,
    trained on skeletonised glyphs; README.md in this folder warns about
    exactly this and it is why the warning is there.
  * psm 7 collapses a multi-line headline to two characters. psm 6 splices.

KNOWN LIMITATIONS, stated rather than hidden
  * **Coloured headlines are lost.** These pages print part of the headline in
    red; a grayscale Otsu threshold drops it. On the measured capture the red
    word අකුමිකතා is missing from every reading.
  * **A clipped headline reads partially** — a capture problem, not an OCR one.
  * The result is NOT corrected by mT5. Component 2 is trained on body
    sentences; headline fragments have not been measured.

OFF BY DEFAULT (`SINHALA_TITLE_MODE`). `stub` keeps the previous behaviour.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import cv2

from core.config import (TITLE_MODE, TITLE_TARGET_BAND_PX, TITLE_TESS_LANG,
                         TITLE_TESS_PSM, TITLE_TESSDATA)
from core.schemas import Article

MAX_REGION_PX = 4000 * 1500


from .title_extractor import extract_and_stitch_title, binarize_strip

def read_title_region(img, box, band_h: float = 0.0) -> tuple:
    """OCR one headline region using CRAFT text detection and Tesseract."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = img.shape[:2]
    x0, y0 = max(0, x0 - 8), max(0, y0 - 8)
    x1, y1 = min(w, x1 + 8), min(h, y1 + 8)
    if x1 <= x0 or y1 <= y0:
        return '', 'empty region'
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return '', 'empty crop'
    if crop.shape[0] * crop.shape[1] > MAX_REGION_PX:
        return '', 'region too large to be a headline'

    # Stage 3: CRAFT Line Extraction + Stitching
    try:
        stitched_strip = extract_and_stitch_title(crop)
    except Exception as e:
        return '', f'CRAFT extraction failed: {e}'
        
    if stitched_strip is None:
        return '', 'CRAFT could not extract/stitch any text lines'

    # Stage 4: Binarization + Morphological Smoothing (MAT)
    try:
        # Assuming light background for now, fallback logic could be added if needed
        _, binarized_bgr = binarize_strip(stitched_strip, is_dark_bg=False)
    except Exception as e:
        return '', f'Binarization failed: {e}'

    if not TITLE_TESSDATA.is_dir():
        return '', f'tessdata not found at {TITLE_TESSDATA}'

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as t:
            tmp = t.name
        cv2.imwrite(tmp, binarized_bgr)
        env = dict(os.environ)
        env['TESSDATA_PREFIX'] = str(TITLE_TESSDATA)
        # Use PSM 13 (raw line) as the image is now a single horizontal strip
        r = subprocess.run(
            ['tesseract', tmp, 'stdout', '--oem', '1',
             '--psm', '13', '-l', 'sin_raw'],
            capture_output=True, encoding='utf-8', env=env, timeout=30)
        if r.returncode != 0:
            return '', f'tesseract failed: {(r.stderr or "").strip()[:120]}'
        return ' '.join(r.stdout.split()), ''
    except FileNotFoundError:
        return '', 'tesseract is not on PATH'
    except subprocess.TimeoutExpired:
        return '', 'tesseract timed out'
    except Exception as e:
        return '', f'{type(e).__name__}: {e}'
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def extract(img, article: Article) -> Article:
    """Fill `article.title` from the region Layer 3 located.

    Signature unchanged — the contract in this folder's README has always been
    `extract(img, article) -> article`, and every caller depends on it.
    """
    if (TITLE_MODE or 'stub').lower() == 'stub' or img is None:
        return article

    regions = [r for r in article.regions if r.label == 'title']
    if not regions:
        return article

    band_h = 0.0
    try:
        from layers.l3_segment.closeup import headline_bands, text_lines
        _lines, _med = text_lines(img)
        bs = headline_bands(img, _med)
        if bs:
            hs = sorted(b[1] - b[0] for b in bs)
            band_h = float(hs[len(hs) // 2])
    except Exception:
        band_h = 0.0

    texts = []
    for r in regions:
        txt, why = read_title_region(
            img, (r.box.x1, r.box.y1, r.box.x2, r.box.y2), band_h)
        if txt:
            texts.append(txt)
        elif why:
            article.note = (article.note + '; ' if article.note else '') + \
                f'title: {why}'

    if texts:
        article.title_raw = ' '.join(texts)
        article.title = article.title_raw
    return article
