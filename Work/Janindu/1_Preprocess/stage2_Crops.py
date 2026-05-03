"""
Stage 2 – Crop & Straighten (stage2_Crops.py)
─────────────────────────────────────────────
Uses Perspective Transform to straighten tilted/skewed text regions 
captured as 4-point polygons in Stage 1.
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path

# Import the LayoutBox dataclass from Stage 1
from stage1_LayoutDetection import LayoutBox

def _order_points(pts):
    """
    Orders 4 points: [top-left, top-right, bottom-right, bottom-left]
    """
    pts = np.array(pts, dtype="float32")
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = np.linalg.norm(tl - rightMost, axis=1)
    (tr, br) = rightMost[np.argsort(D), :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

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

def run(img_path: Path, boxes: list[LayoutBox], crop_root: Path) -> list[Path]:
    img = cv2.imread(str(img_path))
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
            
        if crop.size == 0: continue

        out_name = f"{box.label}.png"
        out_path = img_dir / out_name

        cv2.imwrite(str(out_path), crop)
        crop_paths.append(out_path)
        print(f"  [Crop] Saved straightened -> {out_path} ({crop.shape[1]}x{crop.shape[0]} px)")

    return crop_paths

if __name__ == "__main__":
    import sys
    print("Stage 2 is not meant to run standalone — use Main.py.")
    sys.exit(1)
