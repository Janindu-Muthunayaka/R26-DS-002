"""
preprocess_helper.py  —  Standalone helper called by wrapper_preprocess.py
===========================================================================
This script is invoked via the 1_Preprocess venv so surya is available.
It accepts custom input/output paths as CLI args and calls the stage
functions directly, without modifying MainPreProcess.py.

Usage (called internally by wrapper_preprocess.py):
    <preprocess_venv_python> preprocess_helper.py
        --inputs    <Inputs dir>
        --outputs   <Processes dir>
        --images    img1.jpg img2.png ...
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

# ── Add 1_Preprocess to path so its stages are importable ────────────────────
HELPER_DIR     = Path(__file__).resolve().parent          # 3_FrontEnd/
PREPROCESS_DIR = HELPER_DIR.parent / "1_Preprocess"

if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

# ── Import stage runners ──────────────────────────────────────────────────────
from stage1_LayoutDetection import run as layout_detect_run
from stage2_Crops           import run as crop_run
from stage3_Sentences       import run as sentence_run

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Preprocess helper (surya venv)")
    ap.add_argument("--inputs",  required=True, help="Inputs folder path")
    ap.add_argument("--outputs", required=True, help="Processes output folder path")
    ap.add_argument("--images",  nargs="+", required=True, help="Image filenames to process")
    args = ap.parse_args()

    inputs_dir  = Path(args.inputs)
    outputs_dir = Path(args.outputs)

    # Clear and recreate output structure
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)

    layoutdet_dir = outputs_dir / "LayoutDetection"
    crop_dir      = outputs_dir / "Crops"
    sentence_dir  = outputs_dir / "Sentences"
    for d in (layoutdet_dir, crop_dir, sentence_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Validate and collect images
    images_to_process = []
    for name in args.images:
        p = inputs_dir / name
        if p.exists() and p.suffix.lower() in IMG_EXTS:
            images_to_process.append(p)
        else:
            print(f"[PreProcess] WARNING: '{name}' not found in {inputs_dir}, skipping.")

    if not images_to_process:
        print("[PreProcess] No valid images to process.")
        sys.exit(1)

    print(f"[PreProcess] Processing {len(images_to_process)} image(s)…")

    for img_path in images_to_process:
        print(f"\n{'='*60}")
        print(f"[PreProcess] Processing: {img_path.name}")

        # Stage 1: Layout Detection
        annotated_path, boxes = layout_detect_run(img_path, layoutdet_dir)

        if not boxes:
            print(f"[PreProcess] Stage 1: no layout regions found for {img_path.name}, skipping.")
            continue

        # Stage 2: Crop & Deskew
        crop_paths = crop_run(img_path, boxes, crop_dir)
        if not crop_paths:
            print(f"[PreProcess] Stage 2: no crops produced for {img_path.name}, skipping.")
            continue

        # Stage 3: Sentence Strips
        original_stem = img_path.stem
        for crop_path in crop_paths:
            sentence_run(crop_path, sentence_dir, original_stem)

    print(f"\n[PreProcess] Complete. Results in: {outputs_dir}")


if __name__ == "__main__":
    main()
