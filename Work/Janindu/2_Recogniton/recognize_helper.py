"""
recognize_helper.py  —  Standalone helper called by wrapper_recognize.py
=========================================================================
Runs inside the 2_Recogniton venv.
Finds sentence strips produced by preprocessing, groups them by original
image stem, runs Tesseract OCR on each strip using the custom model,
and writes frontend_summary.json.

Usage (called internally by wrapper_recognize.py):
    <recognize_venv_python> recognize_helper.py
        --processes <Processes dir>
        --outputs   <Outputs dir>
        --inputs    <Inputs dir>    (to copy original image for display)
        --images    img1.jpg img2.png ...
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import pytesseract
from PIL import Image

HELPER_DIR    = Path(__file__).resolve().parent
RECOGNIZE_DIR = HELPER_DIR
BASE_DIR      = HELPER_DIR.parent
FRONTEND_DIR  = BASE_DIR / "3_FrontEnd"

# Path to our custom tesseract model data directory
TESSDATA_DIR = RECOGNIZE_DIR / "tessdata"
CUSTOM_LANG  = "sin_custom"

import io
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _find_strips_for_stem(sentences_dir: Path, stem: str) -> list[Path]:
    """Return all sentence strips. If stem is provided, filter by prefix."""
    if not sentences_dir.exists():
        return []
    files = []
    for p in sentences_dir.iterdir():
        if not p.is_file(): continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
        
        if not stem or p.stem.startswith(stem + "_"):
            if not p.name.endswith("_preMAT.png") and not p.name.endswith("_binarized.png") and not p.name.endswith("_mat_debug.png"):
                files.append(p)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Recognize helper (Tesseract)")
    ap.add_argument("--processes", required=True)
    ap.add_argument("--outputs",   required=True)
    ap.add_argument("--inputs",    required=True)
    ap.add_argument("--images",    nargs="+", required=True)
    args = ap.parse_args()

    processes_dir = Path(args.processes)
    outputs_dir   = Path(args.outputs)
    inputs_dir    = Path(args.inputs)

    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Recognize] Using Tesseract with tessdata: {TESSDATA_DIR}")
    print(f"[Recognize] Language model: {CUSTOM_LANG}")

    # CRITICAL: TESSDATA_PREFIX must be set to the tessdata directory.
    # Do NOT use --tessdata-dir in config string - it breaks on Windows paths with spaces.
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

    # OEM 1 = LSTM engine only (custom traineddata has no legacy component)
    # PSM 8 = Assume a single word
    tess_word_config = '--oem 1 --psm 8'

    for name in args.images:
        stem = Path(name).stem
        print(f"\n{'='*60}")
        print(f"[Recognize] Processing original image: {name}")

        # Find layout image via glob (filename varies per image stem)
        layout_dir = processes_dir / "LayoutDetection" / stem
        layout_imgs = list(layout_dir.glob("*_layout.png")) if layout_dir.exists() else []
        layout_img  = layout_imgs[0] if layout_imgs else None
        crop_dir    = processes_dir / "Crops" / stem
        sent_dir    = processes_dir / "Preprocessed" / stem

        rel_layout = _rel(layout_img, outputs_dir) if layout_img and layout_img.exists() else ""
        
        crops = []
        if crop_dir.exists():
            # Include ALL pngs: both orig_ and binarized crops
            crops = [_rel(p, outputs_dir) for p in sorted(crop_dir.glob("*.png"))]

        strips_to_process = _find_strips_for_stem(sent_dir, "")
        if not strips_to_process:
             strips_to_process = _find_strips_for_stem(processes_dir / "Preprocessed", stem)

        if not strips_to_process:
            print(f"[Recognize] No strips found for '{stem}'.")
            out_stem_dir = outputs_dir / stem
            out_stem_dir.mkdir(parents=True, exist_ok=True)
            _write_summary(out_stem_dir, stem, name, "", [], [],
                           error="No preprocessed strips found",
                           layout_image=rel_layout, crop_images=crops)
            continue

        print(f"[Recognize] Found {len(strips_to_process)} strip(s): {[s.name for s in strips_to_process]}")

        split_results: list[dict] = []
        all_texts:     list[str]  = []

        for idx, strip_path in enumerate(strips_to_process, 1):
            strip_stem    = strip_path.stem
            strip_out_dir = outputs_dir / stem / strip_stem
            strip_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [{idx}/{len(strips_to_process)}] {strip_path.name}")

            # ── Copy images to output BEFORE OCR so they always display ──────
            strip_dest = strip_out_dir / strip_path.name
            if not strip_dest.exists():
                shutil.copy2(str(strip_path), str(strip_dest))
            rel_strip = _rel(strip_dest, outputs_dir)

            binarized_name = strip_path.stem + "_binarized.png"
            binarized_src  = processes_dir / "Binarized" / stem / binarized_name
            binarized_dest = strip_out_dir / binarized_name
            rel_strip_binarized = ""
            if binarized_src.exists():
                if not binarized_dest.exists():
                    shutil.copy2(str(binarized_src), str(binarized_dest))
                rel_strip_binarized = _rel(binarized_dest, outputs_dir)
                
            orig_strip_name = strip_path.name
            if "_word_" in orig_strip_name:
                orig_strip_name = orig_strip_name.replace("_strip.png", ".png")
            orig_strip_src = processes_dir / "Sentences" / stem / orig_strip_name
            orig_strip_dest = strip_out_dir / ("orig_" + strip_path.name)
            rel_strip_orig = ""
            if orig_strip_src.exists():
                if not orig_strip_dest.exists():
                    shutil.copy2(str(orig_strip_src), str(orig_strip_dest))
                rel_strip_orig = _rel(orig_strip_dest, outputs_dir)
                    
            # ── Block-level images (Paddle & CRAFT) ───────────────────────────
            block_id = strip_path.name.replace("_strip.png", "")
            base_block_id = block_id.split("_word_")[0]
            
            block_orig_src = processes_dir / "Crops" / stem / f"{base_block_id}_block.png"
            block_orig_dest = strip_out_dir / f"{base_block_id}_block.png"
            rel_block_orig = ""
            if block_orig_src.exists():
                if not block_orig_dest.exists():
                    shutil.copy2(str(block_orig_src), str(block_orig_dest))
                rel_block_orig = _rel(block_orig_dest, outputs_dir)

            block_craft_src = processes_dir / "Crops" / stem / f"{base_block_id}_craft.png"
            block_craft_dest = strip_out_dir / f"{base_block_id}_craft.png"
            rel_block_craft = ""
            if block_craft_src.exists():
                if not block_craft_dest.exists():
                    shutil.copy2(str(block_craft_src), str(block_craft_dest))
                rel_block_craft = _rel(block_craft_dest, outputs_dir)
            # MAT Debug strip (from Preprocessed)
            mat_debug_name = strip_path.stem + "_mat_debug.png"
            mat_debug_src  = strip_path.with_name(mat_debug_name)
            mat_debug_dest = strip_out_dir / mat_debug_name
            rel_strip_mat_debug = ""
            if mat_debug_src.exists():
                if not mat_debug_dest.exists():
                    shutil.copy2(str(mat_debug_src), str(mat_debug_dest))
                rel_strip_mat_debug = _rel(mat_debug_dest, outputs_dir)

            try:
                # Find the directory containing individual words
                block_id = strip_path.name.replace("_strip.png", "")
                word_dir = processes_dir / "Preprocessed" / stem / block_id
                
                predicted_words = []
                if word_dir.exists():
                    word_files = sorted(word_dir.glob("*_mat.png"))
                    for word_file in word_files:
                        img = Image.open(word_file)
                        # Process each word individually with PSM 8
                        word_pred = pytesseract.image_to_string(img, lang=CUSTOM_LANG, config=tess_word_config).strip()
                        if word_pred:
                            predicted_words.append(word_pred)
                
                predicted = " ".join(predicted_words).strip()
                all_texts.append(predicted)
                print(f"     -> predicted: {predicted[:60] if predicted else '[empty]'} ({len(predicted_words)} words)")

                split_results.append({
                    "strip_name":        strip_path.name,
                    "strip_stem":        strip_stem,
                    "predicted_text":    predicted,
                    "cer":               0.0,
                    "wer":               0.0,
                    "strip_image":       rel_strip,
                    "strip_image_binarized": rel_strip_binarized,
                    "strip_image_mat_debug": rel_strip_mat_debug,
                    "strip_image_orig":  rel_strip_orig,
                    "block_image_orig":  rel_block_orig,
                    "block_image_craft": rel_block_craft,
                    "detail_html":       "",
                    "aksharas":          [],
                    "seg_debug":         None,
                    "error":             None,
                })

            except Exception as exc:
                import traceback
                traceback.print_exc()
                split_results.append({
                    "strip_name":        strip_path.name,
                    "strip_stem":        strip_stem,
                    "predicted_text":    "",
                    "cer":               100.0, "wer": 100.0,
                    "strip_image":       rel_strip,       # ← still set even on error
                    "strip_image_binarized": rel_strip_binarized,
                    "strip_image_mat_debug": rel_strip_mat_debug,
                    "strip_image_orig":  rel_strip_orig,
                    "block_image_orig":  rel_block_orig,
                    "block_image_craft": rel_block_craft,
                    "detail_html":       "",
                    "error":             str(exc),
                })

        combined_text = " ".join(t for t in all_texts if t).strip()
        out_stem_dir  = outputs_dir / stem
        out_stem_dir.mkdir(parents=True, exist_ok=True)

        orig_src  = inputs_dir / name
        orig_dest = out_stem_dir / name
        if orig_src.exists() and not orig_dest.exists():
            shutil.copy2(str(orig_src), str(orig_dest))

        rel_orig = _rel(orig_dest, outputs_dir) if orig_dest.exists() else ""

        _write_summary(out_stem_dir, stem, name, combined_text,
                       all_texts, split_results,
                       original_image=rel_orig,
                       layout_image=rel_layout,
                       crop_images=crops)
        print(f"[Recognize] OK  {stem}  ->  '{combined_text or '[no text]'}'")

    print(f"\n[Recognize] Complete. Results in: {outputs_dir}")


def _rel(p: Path, base: Path) -> str:
    try:
        return os.path.relpath(str(p), str(FRONTEND_DIR)).replace("\\", "/")
    except ValueError:
        return str(p)


def _write_summary(out_dir: Path, stem: str, fname: str,
                   combined_text: str, texts: list[str],
                   splits: list[dict], *,
                   error: str = "", original_image: str = "",
                   layout_image: str = "", crop_images: list[str] = []) -> None:
    summary = {
        "stem":           stem,
        "fname":          fname,
        "predicted_text": combined_text,
        "texts":          texts,
        "splits":         splits,
        "original_image": original_image,
        "layout_image":   layout_image,
        "crop_images":     crop_images,
        "generated_at":   datetime.now().isoformat(),
    }
    if error:
        summary["error"] = error
    with open(out_dir / "frontend_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
