"""
Main pipeline entry point.
Calls Stage 1 → Stage 2 → Stage 3 for every image found in DATASET_DIR.

Pipeline:
  Stage 1 – Layout Detection : Detect layout boxes, draw RED outlines + labels.
  Stage 2 – Crop & Deskew    : Crop heading/title boxes, deskew if needed.
  Stage 3 – Sentence Strips  : Split crops into lines, stitch into 512px strips.

Output structure:
  Result/
    LayoutDetection/    ← annotated images with RED boxes (Stage 1)
    Crops/              ← cropped heading regions per image (Stage 2)
      A/                  SectionHeader_00.png, Title_00.png …
      B/                  …
    Sentences/          ← sentence strips per image (Stage 3)
      A/                  SectionHeader_00_strip.png …
      B/                  …
"""

import sys
import random
from pathlib import Path

# ── Path config ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent          # …/Code
DATASET_DIR = BASE_DIR / "InputImages"             # …/InputImages
RESULT_DIR  = BASE_DIR / "Result"                      # …/Code/Result

LAYOUTDET_DIR = RESULT_DIR / "LayoutDetection"
CROP_DIR      = RESULT_DIR / "Crops"
SENTENCE_DIR  = RESULT_DIR / "Sentences"

for d in (LAYOUTDET_DIR, CROP_DIR, SENTENCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Import stages ──────────────────────────────────────────────────────────────
from stage1_LayoutDetection import run as layout_detect_run
from stage2_Crops           import run as crop_run
from stage3_Sentences       import run as sentence_run

# ── Supported image extensions ─────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    )


def main():
    all_images = collect_images(DATASET_DIR)
    if not all_images:
        print(f"[Main] No images found in {DATASET_DIR}")
        sys.exit(0)

    # ── Random sample ─────────────────────────────────────────────────────────
    SAMPLE_SIZE = 9
    images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
    images.sort()   # sort sampled list for readable output

    print(f"[Main] Found {len(all_images)} image(s). Randomly sampled {len(images)}.\n")
    for p in images:
        print(f"  • {p.name}")
    print()

    for img_path in images:
        print(f"{'='*60}")
        print(f"[Main] Processing: {img_path.name}")

        # ── Stage 1 : Layout detection (RED boxes + labels) ────────
        annotated_path, boxes = layout_detect_run(img_path, LAYOUTDET_DIR)
        if not boxes:
            print(f"[Main] Stage 1 found no layout regions for {img_path.name}, skipping.")
            continue

        # ── Stage 2 : Crop & deskew heading/title regions ──────────
        crop_paths = crop_run(img_path, boxes, CROP_DIR)
        if not crop_paths:
            print(f"[Main] Stage 2 produced no crops for {img_path.name}, skipping.")
            continue

        # ── Stage 3 : Stitch lines into sentence strips ────────────
        original_stem = img_path.stem   # e.g. "A", "B"
        for crop_path in crop_paths:
            sentence_run(crop_path, SENTENCE_DIR, original_stem)

    print(f"\n[Main] Pipeline complete.  Results in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
