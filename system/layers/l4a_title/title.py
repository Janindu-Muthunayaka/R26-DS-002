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
from pathlib import Path

import cv2
import numpy as np

from core.config import TITLE_MODE, TITLE_TESSDATA
from core.schemas import Article

MAX_REGION_PX = 4000 * 1500

from .title_extractor import extract_and_stitch_title, binarize_strip


class TitleResult(tuple):
    """Tuple subclass preserving (hybrid_text, why) for backward compatibility
    while exposing raw, mat, hybrid, confidence scores, and intermediate image paths."""
    def __new__(cls, hybrid: str, why: str = '', raw: str = '', mat: str = '',
                conf_raw: float = 0.0, conf_mat: float = 0.0, images: dict = None):
        instance = super().__new__(cls, (hybrid, why))
        instance.hybrid = hybrid
        instance.why = why
        instance.raw = raw
        instance.mat = mat
        instance.conf_raw = conf_raw
        instance.conf_mat = conf_mat
        instance.images = images or {}
        return instance


def _ocr_strip(img_bgr: np.ndarray, lang: str, psm: str = '13') -> tuple[str, float]:
    """Runs Tesseract directly in-memory via stdin streaming and returns (text, mean_confidence)."""
    try:
        ok, buf = cv2.imencode('.png', img_bgr)
        if not ok:
            return '', 0.0
        img_bytes = buf.tobytes()

        env = dict(os.environ)
        env['TESSDATA_PREFIX'] = str(TITLE_TESSDATA)

        # Run TSV to extract word-level text and confidences
        r_tsv = subprocess.run(
            ['tesseract', 'stdin', 'stdout', '--oem', '1',
             '--psm', psm, '-l', lang, '-c', 'tessedit_create_tsv=1'],
            input=img_bytes,
            capture_output=True, env=env, timeout=120)

        words = []
        confs = []
        tsv_out = r_tsv.stdout.decode('utf-8', errors='replace') if r_tsv.stdout else ''
        if r_tsv.returncode == 0 and tsv_out:
            for line in tsv_out.splitlines():
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
                ['tesseract', 'stdin', 'stdout', '--oem', '1',
                 '--psm', psm, '-l', lang],
                input=img_bytes,
                capture_output=True, env=env, timeout=120)
            plain_out = r_plain.stdout.decode('utf-8', errors='replace') if r_plain.stdout else ''
            if r_plain.returncode == 0 and plain_out:
                text = ' '.join(plain_out.split()).strip()
                avg_conf = 50.0 if text else 0.0

        return text, avg_conf
    except Exception as e:
        print(f"  [Title OCR] Error during _ocr_strip: {e}")
        return '', 0.0


def read_title_region(img, box, band_h: float = 0.0, save_dir: Path | None = None, prefix: str = "") -> TitleResult:
    """OCR one headline region with the RAW model (sin_raw).

    `band_h` is unused and kept only for callers such as tools/measure_headline.py."""
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

    images: dict[str, str] = {}
    if save_dir is not None:
        try:
            crop_name = f"{prefix}crop.png"
            cv2.imwrite(str(save_dir / crop_name), crop)
            images['crop'] = crop_name
        except Exception as e:
            print(f"  [Title] Warning saving crop image: {e}")

    # Stage 3: CRAFT Text Detection + Line Extraction + Stitching
    stitched_strip = None
    craft_density = None
    try:
        res_craft = extract_and_stitch_title(crop)
        if isinstance(res_craft, tuple):
            stitched_strip, craft_density = res_craft
        else:
            stitched_strip = res_craft
    except Exception as e:
        return TitleResult('', f'CRAFT extraction failed: {e}', images=images)

    if stitched_strip is None:
        return TitleResult('', 'CRAFT could not extract/stitch any text lines', images=images)

    if save_dir is not None:
        try:
            craft_name = f"{prefix}craft.png"
            # Prefer density heatmap overlay if available, otherwise stitched strip
            vis_img = craft_density if craft_density is not None else stitched_strip
            cv2.imwrite(str(save_dir / craft_name), vis_img)
            images['craft'] = craft_name
        except Exception as e:
            print(f"  [Title] Warning saving CRAFT image: {e}")

    # Stage 4: Binarization (applied to stitched horizontal strip)
    try:
        _, binarized_bgr = binarize_strip(stitched_strip, is_dark_bg=False)
    except Exception as e:
        return TitleResult('', f'Binarization failed: {e}', images=images)

    if save_dir is not None and binarized_bgr is not None:
        try:
            bin_name = f"{prefix}bin.png"
            cv2.imwrite(str(save_dir / bin_name), binarized_bgr)
            images['bin'] = bin_name
        except Exception as e:
            print(f"  [Title] Warning saving binarized image: {e}")

    if not TITLE_TESSDATA.is_dir():
        return TitleResult('', f'tessdata not found at {TITLE_TESSDATA}', images=images)

    # RAW only: sin_raw on the binarized strip. The MAT model (sin_custom) and
    # its skeletonization are not used, so they are not computed.
    text_raw, conf_raw = _ocr_strip(binarized_bgr, 'sin_raw', '13')
    return TitleResult(
        hybrid=text_raw,
        why='' if text_raw else 'raw OCR produced empty text',
        raw=text_raw,
        conf_raw=conf_raw,
        images=images
    )


def extract(img, article: Article, job_dir: Path | None = None) -> Article:
    """Fill `article.title_raw`, `article.title_hybrid` (= raw), `article.title`,
    and `article.title_images` from the region Layer 3 located. RAW model only;
    `article.title_mat` is left empty.
    """
    mode = (TITLE_MODE or 'mat').lower()
    if mode in ('stub', 'off') or img is None:
        if mode in ('stub', 'off'):
            article.note = (article.note + '; ' if article.note else '') + f'title: Layer 4A is in {mode} mode'
        return article

    regions = [r for r in article.regions if r.label == 'title']
    if not regions:
        article.note = (article.note + '; ' if article.note else '') + 'title: no title region detected by L3'
        return article

    raw_texts = []
    hybrid_texts = []
    all_images: dict[str, str] = {}

    for r_idx, r in enumerate(regions):
        prefix = f"art_{article.index}_title_{r_idx}_"
        res = read_title_region(
            img, (r.box.x1, r.box.y1, r.box.x2, r.box.y2),
            save_dir=job_dir, prefix=prefix
        )
        if isinstance(res, TitleResult):
            if res.raw:
                raw_texts.append(res.raw)
            if res.hybrid:
                hybrid_texts.append(res.hybrid)
            elif res.why:
                article.note = (article.note + '; ' if article.note else '') + \
                    f'title: {res.why}'
            if getattr(res, 'images', None):
                if not all_images:
                    all_images.update(res.images)
                else:
                    for k, v in res.images.items():
                        all_images[f"{k}_{r_idx}"] = v
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
    if hybrid_texts:
        article.title_hybrid = ' '.join(hybrid_texts)
    # title_mat stays empty: the MAT model is no longer run.
    article.title = article.title_raw

    article.title_images = all_images
    return article

