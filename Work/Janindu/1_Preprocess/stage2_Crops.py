"""
Stage 2 – Crop & Line Detection  (stage2_Crops.py)
────────────────────────────────────────────────────
For each layout block returned by Stage 1:
  1. Crop the block from the corrected full image.
  2. Run PaddleOCR TextDetection (PP-OCRv5_server_det) on the block crop
     to get individual text-line polygons.
  3. Crop each text line, apply adaptive binarisation, and save as a PNG.

Output per source image
  <crop_root>/<stem>/<block_id>/line_001.png
                               line_002.png
                               …
  <crop_root>/<stem>/<block_id>_meta.json   ← line polygon metadata

The run() function now returns a list of (block_dir, block_meta) tuples
instead of raw crop paths so Stage 3 can iterate line-by-line.
For backward compatibility it also returns the list of individual line paths
(same shape as the old list[Path] return value) via a flat list.
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

# Import the LayoutBox dataclass from Stage 1
from stage1_LayoutDetection import LayoutBox

# ── Tunables ──────────────────────────────────────────────────────────────────
BLOCK_PAD    = 10   # px – expand each layout block crop
LINE_PAD     = 3    # px – expand each line crop
MIN_LINE_H   = 8    # px – discard line crops shorter than this
MIN_LINE_W   = 20   # px – discard line crops narrower than this

# ── Lazy model singleton ──────────────────────────────────────────────────────
_text_detector = None

def _get_text_detector():
    global _text_detector
    if _text_detector is None:
        from craft_text_detector import Craft
        import torch
        print("  [Crop] Loading CRAFT TextDetection model …")
        cuda = torch.cuda.is_available()
        _text_detector = Craft(output_dir=None, crop_type="poly", cuda=cuda)
        print("  [Crop] CRAFT TextDetection model ready.")
    return _text_detector


# ── Helpers ───────────────────────────────────────────────────────────────────
def _poly_to_bbox(poly: list | np.ndarray) -> tuple[int, int, int, int]:
    """Convert a polygon to an axis-aligned bounding box [x1,y1,x2,y2]."""
    poly = np.array(poly)
    return (int(np.min(poly[:, 0])), int(np.min(poly[:, 1])),
            int(np.max(poly[:, 0])), int(np.max(poly[:, 1])))

def perspective_warp(img: np.ndarray, points: np.ndarray, expand_pct: float = 0.02) -> np.ndarray:
    """Warp a 4-point polygon into a perfectly straightened rectangle, slightly expanded."""
    points = np.array(points, dtype=np.float32)
    
    # Expand boundaries by a tiny percent so letters don't get chopped
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
    
    # Pad with whitespace (10px on all sides)
    warped = cv2.copyMakeBorder(warped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    return warped


def _crop_region(img: np.ndarray, x1: int, y1: int,
                 x2: int, y2: int, pad: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    return img[max(0, y1 - pad):min(h, y2 + pad),
               max(0, x1 - pad):min(w, x2 + pad)]




# ── Public API ────────────────────────────────────────────────────────────────
def run(img_path: Path,
        corrected_img_path: Path | None,
        boxes: list[LayoutBox],
        crop_root: Path) -> list[Path]:
    """
    Stage 2 entry point.

    Parameters
    ----------
    img_path           : Path             – original source image path
    corrected_img_path : Path | None      – explicitly passed corrected image path from Stage 1
    boxes              : list[LayoutBox]  – layout blocks from Stage 1
    crop_root          : Path             – root directory for crop output

    Returns
    -------
    list[Path]  – flat list of all saved line-crop PNG paths
    """
    # Use the explicitly passed corrected image, or fallback to the original
    target_path = corrected_img_path if corrected_img_path and corrected_img_path.exists() else img_path
    
    if target_path == img_path:
        print(f"  [Crop] WARNING: using original image (corrected not provided or missing): {img_path.name}")
    else:
        print(f"  [Crop] Using corrected image: {target_path.name}")

    # Load via PIL (respects EXIF) then convert to OpenCV BGR
    pil_img  = Image.open(str(target_path)).convert("RGB")
    pil_img  = ImageOps.exif_transpose(pil_img)
    full_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    stem     = img_path.stem
    img_root = crop_root / stem
    img_root.mkdir(parents=True, exist_ok=True)

    detector  = _get_text_detector()
    all_blocks_dict: dict[str, list[dict]] = {}

    for box in boxes:
        block_id = box.label   # e.g. "body_000", "title_001"

        # ── Level 1: crop the layout block ──────────────────────────────────
        block_crop = _crop_region(full_img, box.x1, box.y1, box.x2, box.y2, pad=BLOCK_PAD)
        if block_crop.size == 0:
            continue
            
        # Save the original block crop
        block_out_name = f"{block_id}_block.png"
        cv2.imwrite(str(img_root / block_out_name), block_crop)

        # ── Level 2: run text-line detection on the block crop ───────────────
        det_result = detector.detect_text(block_crop)
        dt_polys = det_result.get("boxes", [])
        if len(dt_polys) == 0:
            print(f"  [Crop] {block_id}: no lines detected, skipping block.")
            continue
            
        # Draw CRAFT polygons on a copy of the block crop
        craft_viz = block_crop.copy()
        for poly in dt_polys:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(craft_viz, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        craft_out_name = f"{block_id}_craft.png"
        cv2.imwrite(str(img_root / craft_out_name), craft_viz)

        # Sort lines top-to-bottom
        dt_polys = sorted(dt_polys, key=lambda p: float(np.mean(np.array(p)[:, 1])))

        block_crop_gray = cv2.cvtColor(block_crop, cv2.COLOR_BGR2GRAY)
        # Determine background polarity for the entire block
        # Since text usually occupies less than 50% of the block area, the median represents the background.
        is_dark_bg = bool(np.median(block_crop_gray) < 127)
        
        meta_lines = []
        words = []
        for line_idx, poly in enumerate(dt_polys, start=1):
            lx1, ly1, lx2, ly2 = _poly_to_bbox(poly)
            
            # ── Vertical Filter ──
            pts = np.array(poly, dtype=np.float32)
            width_top = np.linalg.norm(pts[0] - pts[1])
            width_bottom = np.linalg.norm(pts[2] - pts[3])
            max_w = int(max(width_top, width_bottom))

            height_left = np.linalg.norm(pts[0] - pts[3])
            height_right = np.linalg.norm(pts[1] - pts[2])
            max_h = int(max(height_left, height_right))
            
            if max_w < max_h:
                print(f"  [Crop] {block_id}: skipping vertical word (w={max_w}, h={max_h})")
                continue

            # ── Level 3: warp the individual line using perspective transform ──
            line_crop = perspective_warp(block_crop, poly)
            lh, lw    = line_crop.shape[:2]
            if lh < MIN_LINE_H or lw < MIN_LINE_W:
                continue

            word_id = f"{block_id}_word_{line_idx:03d}"

            # Absolute bbox in full image
            abs_x1 = box.x1 + lx1
            abs_y1 = box.y1 + ly1
            abs_x2 = box.x1 + lx2
            abs_y2 = box.y1 + ly2

            meta_lines.append({
                "word_id":            word_id,
                "is_dark_bg":         is_dark_bg,
                "poly_in_block":      [list(map(int, p)) for p in poly],
                "bbox_in_full_image": [abs_x1, abs_y1, abs_x2, abs_y2],
            })
            
            words.append({
                "word_id": word_id,
                "img": line_crop,
                "is_dark_bg": is_dark_bg
            })

        # ── Save block metadata JSON ─────────────────────────────────────────
        meta = {
            "block_id":   block_id,
            "label":      box.label,
            "score":      box.confidence,
            "coordinate": [box.x1, box.y1, box.x2, box.y2],
            "words":      meta_lines,
        }
        (img_root / f"{block_id}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        
        all_blocks_dict[block_id] = words
        print(f"  [Crop] {block_id}: {len(words)} word(s) processed")

    return all_blocks_dict


if __name__ == "__main__":
    import sys
    print("Stage 2 is not meant to run standalone — use MainPreProcess.py.")
    sys.exit(1)
