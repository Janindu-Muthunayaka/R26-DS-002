from __future__ import annotations
"""
Stage 3 – Sentence Strip Stitching  (title_extractor_p1.py)
──────────────────────────────────────────────────────────
Reads the pre-binarised line-crop PNGs produced by Stage 2 (one per text
line), resizes each to a common height, and stitches them left-to-right
with a white gap into a single horizontal strip — ready for the MAT OCR.

The CCA / connected-component logic from the old Surya-based stage is
REMOVED; line splitting is now handled upstream by PaddleOCR TextDetection
in Stage 2, so Stage 3 only needs to stitch.

Input  : a *block directory*  <crop_root>/<stem>/<block_id>/
            line_001.png, line_002.png, …   (already binarised, sorted)
Output : <sentence_root>/<original_stem>/<block_id>_strip.png

Public API (same as the old stage):
    run(crop_path, sentence_root, original_stem) -> Path | None

crop_path here is expected to be either:
  • A block directory   (new style, Stage 2 output)
  • An individual line file (old single-file style — handled as a 1-line strip)
"""


import cv2
import numpy as np
from pathlib import Path

# ── Tunables ──────────────────────────────────────────────────────────────────
STRIP_HEIGHT = 384    # px – uniform height for the output strip
GAP_PX       = 40     # px – white gap between stitched line pieces
MIN_LINE_H   = 8      # px – skip lines shorter than this (degenerate guard)
MIN_LINE_W   = 20     # px – skip lines narrower than this

# ── CRAFT Logic ───────────────────────────────────────────────────────────────
_text_detector = None

def _get_text_detector():
    global _text_detector
    if _text_detector is None:
        try:
            try:
                import torchvision.models.vgg
                if not hasattr(torchvision.models.vgg, 'model_urls'):
                    torchvision.models.vgg.model_urls = {
                        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
                    }
            except Exception:
                pass
            from craft_text_detector import Craft
            import torch
            print("  [CRAFT] Loading CRAFT TextDetection model …")
            cuda = torch.cuda.is_available()
            _text_detector = Craft(output_dir=None, crop_type="poly", cuda=cuda)
            print("  [CRAFT] CRAFT TextDetection model ready.")
        except ImportError:
            print("  [CRAFT] ERROR: craft-text-detector not installed!")
            _text_detector = None
    return _text_detector

def _poly_to_bbox(poly: list | np.ndarray) -> tuple[int, int, int, int]:
    """Convert a polygon to an axis-aligned bounding box [x1,y1,x2,y2]."""
    poly = np.array(poly)
    return (int(np.min(poly[:, 0])), int(np.min(poly[:, 1])),
            int(np.max(poly[:, 0])), int(np.max(poly[:, 1])))

def perspective_warp(img: np.ndarray, points: np.ndarray, expand_pct: float = 0.02) -> np.ndarray:
    """Warp a 4-point polygon into a perfectly straightened rectangle, slightly expanded."""
    points = np.array(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    points = points + (points - center) * expand_pct
    
    width_top = np.linalg.norm(points[0] - points[1])
    width_bottom = np.linalg.norm(points[2] - points[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(points[0] - points[3])
    height_right = np.linalg.norm(points[1] - points[2])
    max_height = int(max(height_left, height_right))
    
    if max_width == 0 or max_height == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points, dst_pts)
    warped = cv2.warpPerspective(img, M, (max_width, max_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

def extract_lines_with_craft(img: np.ndarray) -> list[np.ndarray]:
    """Uses CRAFT to detect text polygons, warps them into straight line crops, and sorts them."""
    detector = _get_text_detector()
    if detector is None:
        return [img]  # Fallback if CRAFT not available
        
    det_result = detector.detect_text(img)
    dt_polys = det_result.get("boxes", [])
    if len(dt_polys) == 0:
        return [img]
        
    # Extract heatmaps for background polarity (optional, simplifying for title extraction)
    # 2D Reading Order Sort (Top-to-Bottom, Left-to-Right)
    poly_data = []
    for p in dt_polys:
        arr = np.array(p)
        cx, cy = np.mean(arr, axis=0)
        h = np.max(arr[:, 1]) - np.min(arr[:, 1])
        poly_data.append({'poly': p, 'cx': cx, 'cy': cy, 'h': h})
        
    if poly_data:
        poly_data.sort(key=lambda d: d['cy'])
        avg_h = np.mean([d['h'] for d in poly_data])
        y_threshold = avg_h * 0.5
        
        lines = []
        current_line = [poly_data[0]]
        for d in poly_data[1:]:
            if abs(d['cy'] - current_line[-1]['cy']) < y_threshold:
                current_line.append(d)
            else:
                lines.append(current_line)
                current_line = [d]
        lines.append(current_line)
        
        dt_polys = []
        for line in lines:
            line.sort(key=lambda d: d['cx'])
            dt_polys.extend([d['poly'] for d in line])

    line_crops = []
    for poly in dt_polys:
        # Vertical Filter
        pts = np.array(poly, dtype=np.float32)
        width_top = np.linalg.norm(pts[0] - pts[1])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        max_w = int(max(width_top, width_bottom))
        height_left = np.linalg.norm(pts[0] - pts[3])
        height_right = np.linalg.norm(pts[1] - pts[2])
        max_h = int(max(height_left, height_right))
        
        if max_w < max_h:
            continue # skip vertical words
            
        line_crop = perspective_warp(img, poly)
        lh, lw = line_crop.shape[:2]
        if lh < MIN_LINE_H or lw < MIN_LINE_W:
            continue
            
        # Pad slightly with white background (simplifying background color logic for titles)
        line_crop = cv2.copyMakeBorder(line_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        line_crops.append(line_crop)
        
    return line_crops if line_crops else [img]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize line crop to target_h using high-quality Lanczos4 interpolation."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_h, 10, 3), dtype=np.uint8) * 255
        
    scale = target_h / h
    new_w = max(1, int(w * scale))
    
    # Use Lanczos4 for premium anti-aliasing during the stitch-resize
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _white_gap(height: int, width: int = GAP_PX) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def _load_line_images_from_dir(block_dir: Path) -> list[np.ndarray]:
    """
    Load line PNG files from a block directory in sorted order.
    Returns a list of BGR numpy arrays.
    """
    line_files = sorted(block_dir.glob("*line_*.png"))
        
    imgs: list[np.ndarray] = []
    for lf in line_files:
        img = cv2.imread(str(lf))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < MIN_LINE_H or w < MIN_LINE_W:
            continue
        imgs.append(img)
    return imgs


def stitch_images(line_imgs: list[np.ndarray]) -> np.ndarray | None:
    """Resize all lines to STRIP_HEIGHT and stitch horizontally."""
    if not line_imgs:
        return None

    pieces: list[np.ndarray] = []
    for i, img in enumerate(line_imgs):
        pieces.append(_resize_to_height(img, STRIP_HEIGHT))
        if i < len(line_imgs) - 1:
            pieces.append(_white_gap(STRIP_HEIGHT))

    strip = cv2.hconcat(pieces)
    
    # Discard if length is higher than 15,000px
    if strip.shape[1] > 15000:
        print(f"  [Sentence] Skipping strip: width ({strip.shape[1]}) exceeds 15000 limit.")
        return None

    return strip

def extract_and_stitch_title(img_bgr: np.ndarray) -> np.ndarray | None:
    """End-to-end processing for a title region crop: CRAFT line extraction + Stitching."""
    line_imgs = extract_lines_with_craft(img_bgr)
    return stitch_images(line_imgs)

# ── Public API ────────────────────────────────────────────────────────────────
def run(crop_input: Path | list[Path],
        sentence_root: Path,
        original_stem: str,
        block_id_hint: str = None) -> Path | None:
    """
    Stage 3 entry point.

    Parameters
    ----------
    crop_input    : Path | list[Path] – block directory (Stage 2 output) OR
                                        list of specific line paths OR
                                        a single line PNG (fallback)
    sentence_root : Path              – root directory for sentence strip output
    original_stem : str               – stem of the original source image (e.g. "A")
    block_id_hint : str               – optional label for the output filename

    Returns
    -------
    Path | None  – path to the saved strip PNG, or None on failure
    """
    out_dir = sentence_root / original_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Case 0: crop_input is a list of paths ───────────────────────────────
    if isinstance(crop_input, list):
        if not crop_input:
            return None
        line_imgs = []
        for p in crop_input:
            img = cv2.imread(str(p))
            if img is not None:
                line_imgs.append(img)
        
        if not line_imgs:
            return None
            
        # If it's a list, we assume they are MAT crops, since this is a fallback.
        block_id = block_id_hint or "strip"
        stitched = stitch_images(line_imgs)
        if stitched is None:
            return None
            
        out_path = out_dir / f"{block_id}_strip.png"
        cv2.imwrite(str(out_path), stitched)
        print(f"  [Sentence] {block_id}: stitched {len(line_imgs)} line(s) from list → {out_path.name}")
        return out_path

    # Use internal name for following cases
    crop_path = crop_input
    # ── Case 1: crop_path is a block directory ───────────────────────────────
    if crop_path.is_dir():
        block_id   = crop_path.name
        
        line_imgs = _load_line_images_from_dir(crop_path)
        
        if not line_imgs:
            print(f"  [Sentence] {block_id}: no line images found in directory, skipping.")
            return None

        stitched = stitch_images(line_imgs)
        if stitched is None:
            return None

        out_path = out_dir / f"{block_id}_strip.png"
        cv2.imwrite(str(out_path), stitched)
        print(f"  [Sentence] {block_id}: stitched {len(line_imgs)} line(s) → {out_path.name}")
        return out_path

    # ── Case 2: crop_path is a single file (legacy / fallback) ──────────────
    img = cv2.imread(str(crop_path))
    if img is None:
        print(f"  [Sentence] ERROR: cannot read {crop_path}")
        return None

    h, w = img.shape[:2]
    if h < MIN_LINE_H or w < MIN_LINE_W:
        print(f"  [Sentence] {crop_path.name}: image too small, skipping.")
        return None

    stitched = stitch_images([img])
    if stitched is None:
        return None

    out_path = out_dir / f"{crop_path.stem}_strip.png"
    cv2.imwrite(str(out_path), stitched)
    print(f"  [Sentence] {crop_path.name}: saved single-line strip → {out_path.name}")
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stage3_Sentences.py <block_dir_or_line_png> <original_stem>")
        sys.exit(1)
    src = Path(sys.argv[1])
    out = src.parent.parent.parent / "Result" / "Sentences"
    out.mkdir(parents=True, exist_ok=True)
    run(src, out, sys.argv[2])


"""
Stage 4 – Binarization (title_extractor_p2.py)
────────────────────────────────────────────
Reads the sentence strips produced by Stage 3 and applies:
  1. Grayscale conversion.
  2. Scale to exactly 384px height (LANCZOS/CUBIC).
  3. Otsu/Adaptive binarization.
  4. Pre-MAT smoothing.
"""


import cv2
import numpy as np
from pathlib import Path

def binarize_strip(img_bgr: np.ndarray, is_dark_bg: bool) -> tuple[np.ndarray, np.ndarray]:
    # 1. Image Loading / Grayscale
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Scale Normalization (CRITICAL)
    target_h = 384
    h, w = grey.shape
    if h == 0 or w == 0:
        fallback = np.ones((target_h, 10, 3), dtype=np.uint8) * 255
        return fallback, fallback
        
    scale = target_h / h
    new_w = max(1, int(w * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(grey, (new_w, target_h), interpolation=interpolation)

    # 3. Binarization (Otsu Thresholding)
    _, binary_8u = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # We use the explicitly computed is_dark_bg from Stage 2 (CRAFT mask)
    # If the background is dark, we must invert to make the background white (255)
    if is_dark_bg:
        binary_8u = cv2.bitwise_not(binary_8u)
        
    # Boolean array where True means foreground text (black pixels)
    binary = binary_8u == 0
    
    # Remove specks and noise particles dynamically
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    
    areas = stats[1:, cv2.CC_STAT_AREA] # Exclude background label 0
    if len(areas) > 0:
        max_area = np.max(areas)
        # Dynamic threshold: 0.1% of the largest character's area.
        # Clamped between 5 pixels (absolute minimum noise) and 50 pixels (safe upper bound to avoid deleting periods/diacritics)
        min_area = max(5, min(50, max_area * 0.001))
        
        for l in range(1, num_labels):
            if stats[l, cv2.CC_STAT_AREA] < min_area:
                binary[labels == l] = False

    
    # --- MAT OPTIMIZATION ---
    binary_text = binary.astype(np.uint8) * 255
    binary_text = cv2.morphologyEx(binary_text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    binary_text = cv2.medianBlur(binary_text, 3)
    binary = binary_text == 255
    
    # Save the binarized state
    binarized_grey = np.where(binary, 0, 255).astype(np.uint8)
    binarized_bgr = cv2.cvtColor(binarized_grey, cv2.COLOR_GRAY2BGR)

    # Return the boolean mask for internal use if needed, and the bgr image
    return binary, binarized_bgr

def run(sentence_path: Path, binarized_root: Path, original_stem: str) -> Path | None:
    # Not used in word-level architecture, keep as stub
    pass

if __name__ == "__main__":
    pass


"""
Stage 5 – MAT Skeletonization (title_extractor_p3.py)
─────────────────────────────────────────────
Reads the binarized strips produced by Stage 4 and applies:
  1. MAT Skeletonization.
  2. Dilation (stroke thickening).
  3. Final RGB formatting.
"""


import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis, disk, dilation

# ─── VISUALISATION SETTINGS ───────────────────────────────────────────────────
EDGE_SAMPLE_GRID   = 10
GRADIENT_MAX_STEPS = 500
COLOUR_YELLOW      = (0, 220, 255)
SKELETON_DILATE    = 1
COLOUR_RED         = (0, 0, 220)

def trace_yellow_lines(binary_255: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Trace inward gradient lines from edge points toward stroke centres."""
    h, w = binary_255.shape
    grad_y, grad_x = np.gradient(dist)
    mag    = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    grad_x = grad_x / mag
    grad_y = grad_y / mag

    edges    = cv2.Canny(binary_255, 50, 150)
    edge_pts = np.argwhere(edges > 0)

    grid = {}
    for (r, c) in edge_pts:
        key = (r // EDGE_SAMPLE_GRID, c // EDGE_SAMPLE_GRID)
        if key not in grid:
            grid[key] = (r, c)
    sampled = list(grid.values())

    # Create a nice canvas: white background, light gray text
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[binary_255 == 0] = [220, 220, 220]

    for (r, c) in sampled:
        pts = [(c, r)]
        cr, cc = float(r), float(c)

        for _ in range(GRADIENT_MAX_STEPS):
            ri = int(np.clip(cr, 0, h - 1))
            ci = int(np.clip(cc, 0, w - 1))

            nr = float(np.clip(cr + grad_y[ri, ci], 0, h - 1))
            nc = float(np.clip(cc + grad_x[ri, ci], 0, w - 1))

            if binary_255[int(nr), int(nc)] == 255:
                break

            cr, cc = nr, nc
            pts.append((int(nc), int(nr)))

        if len(pts) > 2:
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], COLOUR_YELLOW, 1)

    return canvas

def overlay_skeleton(canvas_yellow: np.ndarray, letter_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute medial axis skeleton and overlay in red on the yellow canvas."""
    from skimage.morphology import skeletonize, disk, dilation
    
    # 4. Medial Axis Transform (MAT) Skeletonization (using skeletonize for consistency with p2_MeanAxis_v4)
    skeleton = skeletonize(letter_mask.astype(bool))
    skel_img = skeleton.astype(np.uint8) * 255

    canvas_red = canvas_yellow.copy()
    if SKELETON_DILATE > 0:
        k = np.ones((3, 3), np.uint8)
        skel_dilated = cv2.dilate(skel_img, k, iterations=SKELETON_DILATE)
        canvas_red[skel_dilated > 0] = COLOUR_RED
    else:
        canvas_red[skel_img > 0] = COLOUR_RED

    # 5. Stroke Thickening (Dilation) for OCR final output
    footprint = disk(1)
    thickened = dilation(skeleton, footprint)

    return thickened, canvas_red

def mat_process(binarized_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Extract binary image: text = 0 (black), background = 255 (white)
    grey = cv2.cvtColor(binarized_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_255 = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
    
    letter_mask = (binary_255 == 0).astype(np.uint8)
    
    # Compute distance transform
    dist = cv2.distanceTransform(letter_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Create debug visualization
    canvas_yellow = trace_yellow_lines(binary_255, dist)
    thickened, mat_debug = overlay_skeleton(canvas_yellow, letter_mask)

    # 6. Final Rendering & Formatting for OCR
    final_grey = np.where(thickened, 0, 255).astype(np.uint8)
    final_bgr = cv2.cvtColor(final_grey, cv2.COLOR_GRAY2BGR)

    return final_bgr, mat_debug

def run(binarized_path: Path, preprocessed_root: Path, original_stem: str) -> tuple[Path | None, Path | None, Path | None]:
    # Not used in word-level architecture, keep as stub
    pass

if __name__ == "__main__":
    pass


"""
title_extractor_p4.py  —  Standalone helper called by wrapper_recognize.py
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


import argparse
import sys
import os
import json
import shutil
import concurrent.futures
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
CUSTOM_LANG_RAW = "sin_raw"

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
                
                def recognize_mat():
                    predicted_words = []
                    if word_dir.exists():
                        word_files = sorted(word_dir.glob("*_mat.png"))
                        for word_file in word_files:
                            img = Image.open(word_file)
                            # Process each word individually with PSM 8
                            word_pred = pytesseract.image_to_string(img, lang=CUSTOM_LANG, config=tess_word_config).strip()
                            if word_pred:
                                predicted_words.append(word_pred)
                    return " ".join(predicted_words).strip()
                
                def recognize_raw():
                    if binarized_src.exists():
                        img = Image.open(binarized_src)
                        # Process binarized strip with PSM 13 (Raw line)
                        return pytesseract.image_to_string(img, lang=CUSTOM_LANG_RAW, config='--oem 1 --psm 13').strip()
                    return ""

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f_mat = executor.submit(recognize_mat)
                    f_raw = executor.submit(recognize_raw)
                    predicted = f_mat.result()
                    predicted_raw = f_raw.result()

                all_texts.append(predicted)
                print(f"     -> predicted (MAT): {predicted[:60] if predicted else '[empty]'}")
                print(f"     -> predicted (RAW): {predicted_raw[:60] if predicted_raw else '[empty]'}")

                split_results.append({
                    "strip_name":        strip_path.name,
                    "strip_stem":        strip_stem,
                    "predicted_text":    predicted,
                    "predicted_text_raw": predicted_raw,
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
                    "predicted_text_raw": "",
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
