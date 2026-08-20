"""
Stage 5 – MAT Skeletonization (stage5_MAT.py)
─────────────────────────────────────────────
Reads the binarized strips produced by Stage 4 and applies:
  1. MAT Skeletonization.
  2. Dilation (stroke thickening).
  3. Final RGB formatting.
"""

from __future__ import annotations

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
