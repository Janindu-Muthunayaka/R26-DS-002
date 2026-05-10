"""
recognize_helper.py  —  Standalone helper called by wrapper_recognize.py
=========================================================================
Runs inside the 2_Recogniton venv (PyTorch / EfficientNet).
Finds sentence strips produced by preprocessing, groups them by original
image stem, runs OCR on each strip, and writes frontend_summary.json.

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

# ── Add 2_Recogniton to path so all stage modules are importable ──────────────
HELPER_DIR    = Path(__file__).resolve().parent          # 3_FrontEnd/
FRONTEND_DIR  = HELPER_DIR
BASE_DIR      = HELPER_DIR.parent
RECOGNIZE_DIR = BASE_DIR / "2_Recogniton"

if str(RECOGNIZE_DIR) not in sys.path:
    sys.path.insert(0, str(RECOGNIZE_DIR))

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
        
        # If stem is empty, we take everything in this dir (it's already a stem folder)
        # If stem is provided, we filter by prefix (old flat structure)
        if not stem or p.stem.startswith(stem + "_"):
            files.append(p)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Recognize helper (PyTorch venv)")
    ap.add_argument("--processes", required=True)
    ap.add_argument("--outputs",   required=True)
    ap.add_argument("--inputs",    required=True, help="Inputs folder (for original image copy)")
    ap.add_argument("--images",    nargs="+", required=True)
    args = ap.parse_args()

    processes_dir = Path(args.processes)
    outputs_dir   = Path(args.outputs)
    inputs_dir    = Path(args.inputs)
    sentences_dir = processes_dir / "Sentences"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Imports (need PyTorch venv) ────────────────────────────────────────────
    from stage1_config import (
        PipelineConfig, _DEVICE,
        MODEL_PATH, CLASS_MAP, WORK_ROOT, LABEL_CSV,
    )
    from stage4_classification import _load_model
    from MainRecognize import (
        run_single_image_dynamic,
        FallbackOptimizer,
    )

    # Load model once
    print(f"[Recognize] Device: {_DEVICE}")
    print(f"[Recognize] Loading model: {MODEL_PATH}")
    with open(CLASS_MAP, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    model = _load_model(MODEL_PATH, len(idx_to_class), _DEVICE)

    # Optimizer: try MetaOptimizer first, fall back gracefully
    try:
        from MainRecognize import MetaOptimizer
        optimizer = MetaOptimizer()
        print("[Recognize] Using MetaOptimizer (Random Forest)")
    except Exception as e:
        print(f"[Recognize] MetaOptimizer unavailable ({e}), using FallbackOptimizer.")
        optimizer = FallbackOptimizer()

    # ── Process each original image ───────────────────────────────────────────
    for name in args.images:
        stem = Path(name).stem
        print(f"\n{'='*60}")
        print(f"[Recognize] Processing original image: {name}")

        # discovery of preprocessing artifacts
        layout_img = processes_dir / "LayoutDetection" / f"{stem}_boxes.png"
        crop_dir   = processes_dir / "Crops" / stem
        sent_dir   = processes_dir / "Sentences" / stem

        rel_layout = _rel(layout_img, outputs_dir) if layout_img.exists() else ""
        
        crops = []
        if crop_dir.exists():
            crops = [_rel(p, outputs_dir) for p in sorted(crop_dir.glob("*.png"))]
            
        strips_to_process = _find_strips_for_stem(sent_dir, "") # In this dir they are already filtered by stem
        if not strips_to_process:
             # Fallback to the old way if they are all in the root Sentences dir
             strips_to_process = _find_strips_for_stem(processes_dir / "Sentences", stem)

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

            strip_cfg = PipelineConfig(
                input_folder        = str(strip_path.parent),
                label_csv           = LABEL_CSV,
                model_path          = MODEL_PATH,
                class_map           = CLASS_MAP,
                work_root           = str(strip_out_dir),
                output_folder       = str(strip_out_dir),
                results_folder      = str(strip_out_dir),
                word_spacer_enabled = True,
            )

            try:
                result    = run_single_image_dynamic(
                    img_path     = str(strip_path),
                    ground_truth = "",
                    optimizer    = optimizer,
                    model        = model,
                    idx_to_class = idx_to_class,
                    base_cfg     = strip_cfg,
                    run_idx      = idx,
                    tess_text    = None,
                )
                predicted = result.get("predicted_text", "")
                all_texts.append(predicted)
                print(f"     -> {predicted or '[empty]'}")

                # Copy strip image to output for display
                strip_dest = strip_out_dir / strip_path.name
                if not strip_dest.exists():
                    shutil.copy2(str(strip_path), str(strip_dest))

                rel_strip = _rel(strip_dest, outputs_dir)
                rel_html  = ""
                html_p    = strip_out_dir / "index.html"
                if html_p.exists():
                    rel_html = _rel(html_p, outputs_dir)

                split_results.append({
                    "strip_name":     strip_path.name,
                    "strip_stem":     strip_stem,
                    "predicted_text": predicted,
                    "cer":            result.get("cer", 0.0),
                    "wer":            result.get("wer", 0.0),
                    "strip_image":    rel_strip,
                    "detail_html":    rel_html,
                    "error":          result.get("error"),
                })

            except Exception as exc:
                import traceback
                traceback.print_exc()
                split_results.append({
                    "strip_name":     strip_path.name,
                    "strip_stem":     strip_stem,
                    "predicted_text": "",
                    "cer": 100.0, "wer": 100.0,
                    "strip_image": "", "detail_html": "",
                    "error": str(exc),
                })

        # ── Write per-image summary ───────────────────────────────────────────
        combined_text = " ".join(t for t in all_texts if t).strip()
        out_stem_dir  = outputs_dir / stem
        out_stem_dir.mkdir(parents=True, exist_ok=True)

        # Copy original image for thumbnail display
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
    """Return path relative to FRONTEND_DIR for consistent serving."""
    try:
        # We want the path relative to 3_FrontEnd so we can identify if it's in Processes or Outputs
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
