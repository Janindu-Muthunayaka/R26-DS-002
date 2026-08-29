#!/usr/bin/env python3
"""
try_closeup_ocr.py — find a reading strategy that works on close-up captures.

THE PROBLEM, measured on burst_20260820_105855_g27_s3615_4.jpg:
  * the frame is ONE article at close range, cut off on every side
  * YOLO (trained on full and half pages) returned a single box covering the
    NEIGHBOURING article's headline at the bottom, and missed the article
    filling the frame
  * that box was then rescaled by 0.19 and produced 0 characters

So this bypasses YOLO entirely and asks a narrower question: given a close-up
frame, what combination of crop, scale and Tesseract page-segmentation mode
actually produces Sinhala text?

Nothing here is committed to the pipeline. It is an experiment whose output
decides what gets committed.

    python tools/try_closeup_ocr.py --root <root> "F:\\App\\backend\\inbox\\burst_*.jpg"

WHY --psm MATTERS: core/config.py uses `--psm 6` = "assume a single uniform
block of text". That is correct for Pipeline v9, which fed Tesseract one clean
single-column region at a time. A close-up frame is several columns, and psm 6
will read straight across them. psm 3 is Tesseract's own column-aware layout
analysis. This measures the difference instead of assuming it.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

PSMS = [3, 4, 6, 11]
PSM_NAME = {3: "auto, column-aware", 4: "single column of variable sizes",
            6: "single uniform block (current setting)", 11: "sparse text"}


def text_lines(img, kern_w=25):
    """Text lines by SHAPE, not brightness. A hand or dark surround is one huge
    blob and fails the aspect/height tests, so no page mask is needed."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_w, 3))
    cs, _ = cv2.findContours(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b = [cv2.boundingRect(c) for c in cs]
    cand = [r for r in b if r[2] > 2 * r[3] and 8 < r[3] < H * 0.10
            and r[2] > W * 0.03]
    if not cand:
        return [], 0.0
    med = float(np.median([r[3] for r in cand]))
    return [r for r in cand if 0.45 * med <= r[3] <= 3.0 * med], med


def text_bbox(lines, W, H, pad=20):
    """Bounding box of the detected text — this is what crops out the hand,
    the page edge and the dark surround, without any brightness heuristic."""
    if not lines:
        return 0, 0, W, H
    x1 = max(0, min(r[0] for r in lines) - pad)
    y1 = max(0, min(r[1] for r in lines) - pad)
    x2 = min(W, max(r[0] + r[2] for r in lines) + pad)
    y2 = min(H, max(r[1] + r[3] for r in lines) + pad)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--root", default=os.getenv("SINHALA_ROOT"))
    ap.add_argument("--chars", type=int, default=220)
    ap.add_argument("--save-crops", default=None,
                    help="directory to write the cropped images for inspection")
    a = ap.parse_args()

    from core import config
    if a.root:
        config.set_root(a.root)
    from core.imaging import imread_upright, glyph_p75, glyph_p90
    import pytesseract

    paths = []
    for pat in a.images:
        p = Path(pat)
        paths.extend(sorted(p.parent.glob(p.name))
                     if any(c in pat for c in "*?") else [p])
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no images matched")
        return 1

    # target: what the LINE height should become. The research says base glyphs
    # read best around p90 22-30 and diacritics die under ~11px, so scaling is
    # floored rather than left to a bare ratio.
    for path in paths:
        img = imread_upright(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        lines, med_h = text_lines(img)
        x1, y1, x2, y2 = text_bbox(lines, W, H)
        crop = img[y1:y2, x1:x2]
        print("\n" + "=" * 72)
        print(f"{path.name}   {W}x{H}")
        print(f"  detected lines {len(lines)}   median line height {med_h:.0f}px")
        print(f"  text bbox      ({x1},{y1})-({x2},{y2})  "
              f"= {100*(x2-x1)*(y2-y1)/(W*H):.0f}% of frame  "
              f"(the rest is hand, margin, neighbouring article)")
        print(f"  crop glyph     p75 {glyph_p75(crop)}  p90 {glyph_p90(crop)}")

        if a.save_crops:
            d = Path(a.save_crops); d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{path.stem}_crop.jpg"), crop)

        for scale in (1.0, 0.6, 0.4):
            sub = crop if scale == 1.0 else cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            eff_line = med_h * scale
            print(f"\n  --- scale {scale:.1f}  ({sub.shape[1]}x{sub.shape[0]}, "
                  f"line height ~{eff_line:.0f}px) ---")
            for psm in PSMS:
                try:
                    txt = pytesseract.image_to_string(
                        cv2.cvtColor(sub, cv2.COLOR_BGR2RGB),
                        lang=config.TESS_LANG,
                        config=f"--oem 1 --psm {psm}")
                except Exception as e:
                    print(f"    psm {psm:<2}  ERROR {e}")
                    continue
                flat = " ".join(txt.split())
                print(f"    psm {psm:<2} {len(flat):5d} chars  "
                      f"({PSM_NAME[psm]})")
                if flat:
                    print(f"          {flat[:a.chars]}")
    print("\n" + "=" * 72)
    print("Pick the (scale, psm) with the most PLAUSIBLE Sinhala, not simply "
          "the most characters — psm 11 often emits many short fragments.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
