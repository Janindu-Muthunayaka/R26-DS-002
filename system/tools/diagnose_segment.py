#!/usr/bin/env python3
"""
diagnose_segment.py — why did this image produce zero articles?

run_pipeline.py says "0 articles". This says WHERE they were lost, stage by
stage, with the numbers that decided each drop. It does not guess.

    python tools/diagnose_segment.py --root <root> "F:\\App\\backend\\inbox\\burst_....jpg"

Reports, in pipeline order:
  1. YOLO raw detections           (did the detector see anything at all?)
  2. border_filter survivors        (edge-touching boxes are dropped)
  3. reading order / de-overlap
  4. the box list actually used     (note the whole-frame fallback)
  5. per box: the region finder's intermediates, including the width
     threshold that silently removes column-width text lines
  6. per region: Tesseract character count

Needs cv2, numpy, ultralytics, pytesseract. No server, no phone.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--root", default=os.getenv("SINHALA_ROOT"))
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--ocr", action="store_true",
                    help="also run Tesseract per region (slower)")
    a = ap.parse_args()

    from core import config
    if a.root:
        config.set_root(a.root)
    from core.imaging import imread_upright, glyph_p75, glyph_p90
    from layers.l3_segment.geometry import (border_filter, deoverlap,
                                            page_reading_order)

    img = imread_upright(a.image)
    if img is None:
        print("could not read", a.image)
        return 2
    H, W = img.shape[:2]
    print(f"image      {Path(a.image).name}   {W}x{H} (after EXIF transpose)")
    print(f"glyph      p75 {glyph_p75(img)}   p90 {glyph_p90(img)}")

    # ---- 1. YOLO ---------------------------------------------------------
    from ultralytics import YOLO
    w = next((p for p in config.YOLO_WEIGHTS if p.exists()), None)
    if w is None:
        print("YOLO weights not found")
        return 2
    conf = a.conf if a.conf is not None else config.YOLO_CONF
    r = YOLO(str(w)).predict(img, imgsz=config.YOLO_IMGSZ, conf=conf,
                             verbose=False)[0]
    raw = [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]
    confs = [float(c) for c in r.boxes.conf.cpu().numpy()] if len(raw) else []
    print(f"\n1. YOLO raw detections at conf>={conf}:  {len(raw)}")
    for b, c in zip(raw, confs):
        frac = ((b[2]-b[0]) * (b[3]-b[1])) / (W * H)
        edge = (b[0] <= 8 or b[1] <= 8 or b[2] >= W-8 or b[3] >= H-8)
        print(f"   ({b[0]:6.0f},{b[1]:6.0f})-({b[2]:6.0f},{b[3]:6.0f})  "
              f"conf {c:.2f}  {frac:5.1%} of frame  "
              f"{'TOUCHES EDGE' if edge else ''}")
    if not raw:
        print("   -> nothing detected. The detector was trained on full and "
              "half page framings; a close-up of one article is outside that.")

    # ---- 2..4 the geometry chain ----------------------------------------
    bf = border_filter(raw, W, H)
    print(f"\n2. after border_filter:  {len(bf)}  (dropped {len(raw)-len(bf)})")
    if raw and len(bf) < len(raw):
        med = np.median([(x2-x1)*(y2-y1) for x1, y1, x2, y2 in raw])
        print(f"   rule: edge-touching AND area < 0.45 x median({med:.0f})")

    ordered = [bf[i] for i in page_reading_order(bf)] if bf else []
    boxes = deoverlap(ordered) if ordered else []
    fallback_used = not boxes
    if fallback_used:
        boxes = [[0, 0, W, H]]
    print(f"\n3-4. boxes used: {len(boxes)}"
          f"{'   <-- WHOLE-FRAME FALLBACK (segment.py line ~39)' if fallback_used else ''}")

    # ---- 5. the region finder, instrumented ------------------------------
    print("\n5. region finder (the --no-layout fallback heuristic):")
    total_regions = 0
    for i, bx in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in bx]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            print(f"   box {i}: empty crop")
            continue
        cw = crop.shape[1]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(g, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, cw // 12), 3))
        lines = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)
        cs, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        bs_all = [cv2.boundingRect(c) for c in cs]
        wmin = cw * 0.2
        bs = [b for b in bs_all if b[3] > 4 and b[2] > wmin]
        widths = sorted((b[2] for b in bs_all), reverse=True)[:8]
        print(f"\n   box {i}  {x2-x1}x{y2-y1}px")
        print(f"     contours found        {len(bs_all)}")
        print(f"     width threshold       {wmin:.0f}px  (0.2 x box width {cw})")
        print(f"     survive the filter    {len(bs)}")
        print(f"     widest contours       {widths}")
        if bs_all and len(bs) == 0:
            print(f"     -> EVERY line was dropped. The threshold is relative "
                  f"to the BOX width. On a whole-frame fallback box, one text "
                  f"column is narrower than 20% of the frame, so real body "
                  f"lines are discarded as noise.")
        total_regions += len(bs)

        if a.ocr and bs:
            import pytesseract
            from core.imaging import rescale_to_optimum
            from core.config import TESS_CONFIG, TESS_LANG
            bs.sort(key=lambda b: b[1])
            for k, (bx_, by_, bw_, bh_) in enumerate(bs[:6]):
                sub = crop[by_:by_+bh_, bx_:bx_+bw_]
                sub, p90, sc = rescale_to_optimum(sub)
                txt = pytesseract.image_to_string(
                    cv2.cvtColor(sub, cv2.COLOR_BGR2RGB),
                    lang=TESS_LANG, config=TESS_CONFIG).strip()
                print(f"     region {k}: {bw_}x{bh_}  scale {sc:.2f}  "
                      f"{len(txt)} chars   {txt[:60]!r}")

    print("\n" + "=" * 66)
    print(f"regions found in total: {total_regions}")
    if total_regions == 0:
        print("No regions -> no OCR -> l5_assemble drops the article for having "
              "no text -> 'articles 0'.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
