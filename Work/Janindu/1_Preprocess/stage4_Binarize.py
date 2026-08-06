"""
Stage 4 – Binarization (stage4_Binarize.py)
────────────────────────────────────────────
Reads the sentence strips produced by Stage 3 and applies:
  1. Grayscale conversion.
  2. Scale to exactly 384px height (LANCZOS/CUBIC).
  3. Otsu/Adaptive binarization.
  4. Pre-MAT smoothing.
"""

from __future__ import annotations

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
