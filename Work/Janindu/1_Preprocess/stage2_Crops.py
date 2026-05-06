"""
Stage 2 – Crop & Straighten (stage2_Crops.py)
─────────────────────────────────────────────
1. Uses Perspective Transform to straighten tilted/skewed text regions
   captured as 4-point polygons in Stage 1.
2. Post-processes each crop: denoise → Otsu binarise → morphological
   closing → invert → BLACK text on WHITE background.
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

# Import the LayoutBox dataclass from Stage 1
from stage1_LayoutDetection import LayoutBox

def _order_points(pts):
    """
    Orders 4 points: [top-left, top-right, bottom-right, bottom-left]
    Uses the sum/difference method which is robust to rotation.
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has the smallest sum, Bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has the smallest difference (y - x), 
    # Bottom-left has the largest difference (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def _straighten_crop(img: np.ndarray, polygon: list[list[int]]) -> np.ndarray:
    """
    Applies Perspective Transform to straighten a tilted text region.
    """
    pts = _order_points(polygon)
    (tl, tr, br, bl) = pts

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), borderMode=cv2.BORDER_REPLICATE)
    
    return warped


def _binarise_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a colour crop to a clean BLACK-on-WHITE image using Otsu.
    We avoid adaptive thresholding here because it creates noise dots in 
    uniform background areas.

    Pipeline:
      1. Mild bilateral filter  – softens noise while keeping stroke edges.
      2. Grayscale + Otsu       – global threshold for clean background.
      3. Morphological closing  – fills gaps inside strokes.
      4. Polarity fix           – ensure black ink on white background.
    """
    # Step 1: mild denoise
    denoised = cv2.bilateralFilter(crop_bgr, d=7, sigmaColor=50, sigmaSpace=50)

    # Step 2: grayscale + Otsu threshold
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: morphological closing – tiny 2×2 kernel fills ink gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # Step 4: polarity check — ensure text is black on white
    white_px = np.sum(closed == 255)
    black_px = np.sum(closed == 0)
    if white_px < black_px:
        closed = cv2.bitwise_not(closed)

    # Convert single-channel back to 3-channel BGR for downstream compatibility
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

def run(img_path: Path, boxes: list[LayoutBox], crop_root: Path) -> list[Path]:
    # Use PIL to load to ensure EXIF orientation is respected consistently with Stage 1
    pil_img = Image.open(str(img_path)).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if img is None:
        print(f"  [Crop] ERROR: cannot read {img_path}")
        return []

    stem = img_path.stem
    img_dir = crop_root / stem
    img_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[Path] = []

    for i, box in enumerate(boxes):
        if box.polygon:
            # Straighten using the polygon
            crop = _straighten_crop(img, box.polygon)
        else:
            # Fallback to simple crop
            crop = img[box.y1:box.y2, box.x1:box.x2]

        if crop.size == 0:
            continue

        # ── Post-process: binarise to black-on-white ───────────────────────
        crop = _binarise_crop(crop)

        out_name = f"{box.label}.png"
        out_path = img_dir / out_name

        cv2.imwrite(str(out_path), crop)
        crop_paths.append(out_path)
        print(f"  [Crop] Saved (binarised) → {out_path} ({crop.shape[1]}x{crop.shape[0]} px)")

    return crop_paths

if __name__ == "__main__":
    import sys
    print("Stage 2 is not meant to run standalone — use Main.py.")
    sys.exit(1)
