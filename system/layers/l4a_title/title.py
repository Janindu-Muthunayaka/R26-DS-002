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


import concurrent.futures
from .title_extractor import extract_and_stitch_title, binarize_strip, mat_process


class TitleResult(tuple):
    """Tuple subclass preserving (hybrid_text, why) for backward compatibility
    while exposing raw, mat, hybrid, and confidence scores as attributes."""
    def __new__(cls, hybrid: str, why: str = '', raw: str = '', mat: str = '',
                conf_raw: float = 0.0, conf_mat: float = 0.0):
        instance = super().__new__(cls, (hybrid, why))
        instance.hybrid = hybrid
        instance.why = why
        instance.raw = raw or hybrid
        instance.mat = mat or hybrid
        instance.conf_raw = conf_raw
        instance.conf_mat = conf_mat
        return instance


def _ocr_strip(img_bgr: np.ndarray, lang: str, psm: str = '13') -> tuple[str, float]:
    """Runs Tesseract on an image strip and returns (text, mean_confidence)."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as t:
            tmp = t.name
        cv2.imwrite(tmp, img_bgr)
        env = dict(os.environ)
        env['TESSDATA_PREFIX'] = str(TITLE_TESSDATA)

        # Run TSV to extract word-level text and confidences
        r_tsv = subprocess.run(
            ['tesseract', tmp, 'stdout', '--oem', '1',
             '--psm', psm, '-l', lang, '-c', 'tessedit_create_tsv=1'],
            capture_output=True, encoding='utf-8', errors='replace', env=env, timeout=120)

        words = []
        confs = []
        if r_tsv.returncode == 0 and r_tsv.stdout:
            for line in r_tsv.stdout.splitlines():
                parts = line.split('\t')
                if len(parts) >= 12:
                    w = parts[11].strip()
                    try:
                        c = float(parts[10])
                    except (ValueError, IndexError):
                        c = -1.0
                    if c >= 0 and w:
                        words.append(w)
                        confs.append(c)

        text = ' '.join(words).strip()
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0

        # Fallback to plain stdout if TSV parsing returned empty text
        if not text:
            r_plain = subprocess.run(
                ['tesseract', tmp, 'stdout', '--oem', '1',
                 '--psm', psm, '-l', lang],
                capture_output=True, encoding='utf-8', errors='replace', env=env, timeout=120)
            if r_plain.returncode == 0 and r_plain.stdout:
                text = ' '.join(r_plain.stdout.split()).strip()
                avg_conf = 50.0 if text else 0.0

        return text, avg_conf
    except Exception:
        return '', 0.0
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def read_title_region(img, box, band_h: float = 0.0) -> TitleResult:
    """OCR one headline region running RAW and MAT models in parallel
    with a confidence-driven hybrid decision gate."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = img.shape[:2]
    x0, y0 = max(0, x0 - 8), max(0, y0 - 8)
    x1, y1 = min(w, x1 + 8), min(h, y1 + 8)
    if x1 <= x0 or y1 <= y0:
        return TitleResult('', 'empty region')
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return TitleResult('', 'empty crop')
    if crop.shape[0] * crop.shape[1] > MAX_REGION_PX:
        return TitleResult('', 'region too large to be a headline')

    # Stage 3: CRAFT Line Extraction + Stitching
    try:
        stitched_strip = extract_and_stitch_title(crop)
    except Exception as e:
        return TitleResult('', f'CRAFT extraction failed: {e}')

    if stitched_strip is None:
        return TitleResult('', 'CRAFT could not extract/stitch any text lines')

    # Stage 4: Binarization + Morphological Smoothing (MAT)
    try:
        _, binarized_bgr = binarize_strip(stitched_strip, is_dark_bg=False)
    except Exception as e:
        return TitleResult('', f'Binarization failed: {e}')

    if not TITLE_TESSDATA.is_dir():
        return TitleResult('', f'tessdata not found at {TITLE_TESSDATA}')

    # Generate Medial Axis Transform (MAT) skeletonized strip
    try:
        mat_bgr, _ = mat_process(binarized_bgr)
    except Exception:
        mat_bgr = binarized_bgr

    # Approach 1: Run RAW and MAT pipelines in parallel
    text_raw, conf_raw = '', 0.0
    text_mat, conf_mat = '', 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_raw = executor.submit(_ocr_strip, binarized_bgr, 'sin_raw', '13')
        f_mat = executor.submit(_ocr_strip, mat_bgr, 'sin_custom', '13')
        try:
            text_raw, conf_raw = f_raw.result()
        except Exception:
            text_raw, conf_raw = '', 0.0
        try:
            text_mat, conf_mat = f_mat.result()
        except Exception:
            text_mat, conf_mat = '', 0.0

    # Dynamic Confidence-based Selection (Approach 1)
    if text_mat and text_raw:
        if conf_mat > conf_raw:
            text_hybrid = text_mat
        else:
            text_hybrid = text_raw
    elif text_mat:
        text_hybrid = text_mat
    elif text_raw:
        text_hybrid = text_raw
    else:
        text_hybrid = ''

    return TitleResult(
        hybrid=text_hybrid,
        why='' if text_hybrid else 'both OCR models produced empty text',
        raw=text_raw,
        mat=text_mat,
        conf_raw=conf_raw,
        conf_mat=conf_mat
    )


def extract(img, article: Article) -> Article:
    """Fill `article.title_raw`, `article.title_mat`, `article.title_hybrid`,
    and `article.title` from the region Layer 3 located.
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

    raw_texts = []
    mat_texts = []
    hybrid_texts = []

    for r in regions:
        res = read_title_region(
            img, (r.box.x1, r.box.y1, r.box.x2, r.box.y2), band_h)
        if isinstance(res, TitleResult):
            if res.raw:
                raw_texts.append(res.raw)
            if res.mat:
                mat_texts.append(res.mat)
            if res.hybrid:
                hybrid_texts.append(res.hybrid)
            elif res.why:
                article.note = (article.note + '; ' if article.note else '') + \
                    f'title: {res.why}'
        else:
            txt, why = res
            if txt:
                hybrid_texts.append(txt)
                raw_texts.append(txt)
            elif why:
                article.note = (article.note + '; ' if article.note else '') + \
                    f'title: {why}'

    if raw_texts:
        article.title_raw = ' '.join(raw_texts)
    if mat_texts:
        article.title_mat = ' '.join(mat_texts)
    if hybrid_texts:
        article.title_hybrid = ' '.join(hybrid_texts)

    article.title = article.title_hybrid or article.title_raw or article.title_mat
    return article
