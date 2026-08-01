"""
Main pipeline entry point.
Calls Stage 1 → Stage 2 → Stage 3 for every image found in DATASET_DIR.

Pipeline:
  Stage 1 – Layout Detection : Detect layout boxes (PaddleOCR), unwarp, save JSON + viz.
  Stage 2 – Crop & Detect    : Crop blocks, run line detection (PaddleOCR), save binarised lines.
  Stage 3 – Sentence Strips  : Stitch detected lines from each block into 512px strips.

Output structure:
  Result/
    LayoutDetection/    ← annotated images with layout boxes (Stage 1)
      _corrected/       ← unwarped perspective-corrected source images
    Crops/              ← binarised individual line crops per block (Stage 2)
      A/
        title_000/
          line_001.png, line_002.png
          title_000_meta.json
        body_001/
          line_001.png...
    Sentences/          ← sentence strips per block (Stage 3)
      A/
        title_000_strip.png
        body_001_strip.png
    Preprocessed/       ← Final MAT-processed strips for Tesseract (Stage 4)
      A/
        title_000_strip.png
        body_001_strip.png
"""

import sys
import random
import argparse
from pathlib import Path

# ── Path config ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = Path(r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\1_Preprocess\InputImages")
RESULT_DIR  = Path(r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\1_Preprocess\Result")

# ── Import stages ──────────────────────────────────────────────────────────────
from stage1_LayoutDetection import run as layout_detect_run
from stage2_Crops           import run as crop_run
from stage3_Sentences       import stitch_images
from stage4_Binarize        import binarize_strip
from stage5_MAT             import mat_process
import cv2

# ── Supported image extensions ─────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", help="Inputs folder path")
    ap.add_argument("--outputs", help="Processes output folder path")
    ap.add_argument("--images", nargs="+", help="Image filenames to process")
    args, unknown = ap.parse_known_args()

    if args.inputs and args.outputs:
        dataset_dir = Path(args.inputs)
        result_dir = Path(args.outputs)
        if args.images:
            images = [dataset_dir / name for name in args.images if (dataset_dir / name).exists()]
        else:
            images = collect_images(dataset_dir)
    else:
        dataset_dir = DATASET_DIR
        result_dir = RESULT_DIR
        all_images = collect_images(dataset_dir)
        if not all_images:
            print(f"[Main] No images found in {dataset_dir}")
            sys.exit(0)
        SAMPLE_SIZE = 9
        images = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
        images.sort()

    layoutdet_dir = result_dir / "LayoutDetection"
    crop_dir      = result_dir / "Crops"
    sentence_dir  = result_dir / "Sentences"
    binarize_dir  = result_dir / "Binarized"
    preprocess_dir= result_dir / "Preprocessed"

    for d in (layoutdet_dir, crop_dir, sentence_dir, binarize_dir, preprocess_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Main] Processing {len(images)} image(s).\n")
    for p in images:
        print(f"  • {p.name}")
    print()

    for img_path in images:
        print(f"{'='*60}")
        print(f"[Main] Processing: {img_path.name}")

        # ── Stage 1 : Layout detection (PaddleOCR boxes + labels) ────────
        annotated_path, corrected_img_path, boxes = layout_detect_run(img_path, layoutdet_dir)
            
        if not boxes:
            print(f"[Main] Stage 1 found no layout regions for {img_path.name}, skipping.")
            continue

        # ── Stage 2 : Crop & line detection ──────────
        # Returns a dict of {block_id: list[line_paths]}
        all_blocks_dict = crop_run(img_path, corrected_img_path, boxes, crop_dir)
        if not all_blocks_dict:
            print(f"[Main] Stage 2 produced no crops for {img_path.name}, skipping.")
            continue

        # ── Stage 3 & 4 & 5 : In-memory Word Processing ────────────
        original_stem = img_path.stem   # e.g. "A", "B"
        
        for block_id, words in all_blocks_dict.items():
            if not words:
                continue
                
            binarized_words = []
            mat_words = []
            mat_debug_words = []
            
            # Create a directory to store individual MAT-processed words for this block
            word_out_dir = preprocess_dir / original_stem / block_id
            word_out_dir.mkdir(parents=True, exist_ok=True)
            
            for word in words:
                word_img = word["img"]
                is_dark_bg = word["is_dark_bg"]
                word_id = word["word_id"]
                
                # Binarize
                binary_mask, binarized_img = binarize_strip(word_img, is_dark_bg)
                binarized_words.append(binarized_img)
                
                # MAT
                mat_bgr, mat_debug = mat_process(binarized_img)
                mat_words.append(mat_bgr)
                mat_debug_words.append(mat_debug)
                
                # Save the individual MAT word for the OCR model
                word_path = word_out_dir / f"{word_id}_mat.png"
                cv2.imwrite(str(word_path), mat_bgr)
            
            # ── Stitch for UI Dashboard ──
            # The dashboard expects a single strip per block
            sent_dir = sentence_dir / original_stem
            sent_dir.mkdir(parents=True, exist_ok=True)
            bin_dir = binarize_dir / original_stem
            bin_dir.mkdir(parents=True, exist_ok=True)
            prep_dir = preprocess_dir / original_stem
            prep_dir.mkdir(parents=True, exist_ok=True)
            
            # Stitch raw words (optional, if UI expects raw strip)
            raw_strip = stitch_images([w["img"] for w in words])
            if raw_strip is not None:
                cv2.imwrite(str(sent_dir / f"{block_id}_strip.png"), raw_strip)
                
            bin_strip = stitch_images(binarized_words)
            if bin_strip is not None:
                cv2.imwrite(str(bin_dir / f"{block_id}_strip_binarized.png"), bin_strip)
                # Also copy for UI
                cv2.imwrite(str(prep_dir / f"{block_id}_strip_binarized.png"), bin_strip)
                
            mat_strip = stitch_images(mat_words)
            if mat_strip is not None:
                cv2.imwrite(str(prep_dir / f"{block_id}_strip.png"), mat_strip)
                
            debug_strip = stitch_images(mat_debug_words)
            if debug_strip is not None:
                cv2.imwrite(str(prep_dir / f"{block_id}_strip_mat_debug.png"), debug_strip)
            
            print(f"  [Main] {block_id}: Processed {len(words)} words, stitched UI strips.")
            
    print(f"\n[Main] Pipeline complete.  Results in: {result_dir}")


if __name__ == "__main__":
    main()
