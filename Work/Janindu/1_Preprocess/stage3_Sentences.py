"""
Stage 3 – Sentence Strip Stitching
──────────────────────────────────────────────────────────
Applies Centroid Clustering (Coordinate-Based) (CCA) to split crops into lines.
Outputs a single combined strip image.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# ── Tunables ──────────────────────────────────────────────────────────────────
STRIP_HEIGHT = 512    # px – uniform height for the output strip
GAP_PX       = 40     # px – wider white gap between stitched lines
MIN_LINE_H   = 15     # px – ignore lines shorter than this
LINE_PAD_H   = 20     # px – horizontal padding for each line piece

def _split_lines(img: np.ndarray) -> list[np.ndarray]:
    """
    Uses connected components and clusters them by their Y-centroids to form lines (CCA).
    """
    h, w = img.shape[:2]
    if h < MIN_LINE_H:
        return [img]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    components = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 8:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_box = stats[i, cv2.CC_STAT_WIDTH]
            h_box = stats[i, cv2.CC_STAT_HEIGHT]
            cy = y + h_box / 2
            components.append({'x': x, 'y': y, 'w': w_box, 'h': h_box, 'cy': cy})
            
    if not components:
        return [img]
        
    components.sort(key=lambda c: c['cy'])
    
    clusters = []
    current_cluster = [components[0]]
    
    for comp in components[1:]:
        avg_h = sum(c['h'] for c in current_cluster) / len(current_cluster)
        last_cy_avg = sum(c['cy'] for c in current_cluster[-3:]) / min(3, len(current_cluster))
        
        # If cy difference is small enough, merge
        if abs(comp['cy'] - last_cy_avg) < avg_h * 0.7:
            current_cluster.append(comp)
        else:
            clusters.append(current_cluster)
            current_cluster = [comp]
            
    if current_cluster:
        clusters.append(current_cluster)
        
    line_crops = []
    for cluster in clusters:
        min_x = min(c['x'] for c in cluster)
        min_y = min(c['y'] for c in cluster)
        max_x = max(c['x'] + c['w'] for c in cluster)
        max_y = max(c['y'] + c['h'] for c in cluster)
        
        # Filter noise clusters
        if (max_y - min_y) < MIN_LINE_H or (max_x - min_x) < w * 0.02:
            continue
            
        pad_y = 4
        y1 = max(0, min_y - pad_y)
        y2 = min(h, max_y + pad_y)
        x1 = max(0, min_x - LINE_PAD_H)
        x2 = min(w, max_x + LINE_PAD_H)
        
        line_crops.append(img[y1:y2, x1:x2])
        
    return line_crops if line_crops else [img]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_h, 10, 3), dtype=np.uint8) * 255
    scale = target_h / h
    new_w = max(1, int(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(img, (new_w, target_h), interpolation=interp)

def _white_gap(height: int, width: int = None) -> np.ndarray:
    w = width if width else GAP_PX
    return np.ones((height, w, 3), dtype=np.uint8) * 255

def _sharpen_and_clean(strip: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    _, final_bw = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(final_bw == 255) < np.sum(final_bw == 0):
        final_bw = cv2.bitwise_not(final_bw)
    return cv2.cvtColor(final_bw, cv2.COLOR_GRAY2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def run(crop_path: Path, sentence_root: Path, original_stem: str) -> Path | None:
    """
    Applies Centroid Clustering (CCA) to the crop and saves a single strip.
    Returns the path to the saved strip.
    """
    img = cv2.imread(str(crop_path))
    if img is None:
        print(f"  [Sentence] ERROR: cannot read {crop_path}")
        return None

    line_crops = _split_lines(img)

    if not line_crops:
        print(f"  [Sentence] No text found in {crop_path.name}, skipping.")
        return None

    out_dir = sentence_root / original_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    resized = [_resize_to_height(lc, STRIP_HEIGHT) for lc in line_crops]
    
    pieces = []
    for i, s in enumerate(resized):
        pieces.append(s)
        if i < len(resized) - 1:
            pieces.append(_white_gap(STRIP_HEIGHT))
            
    if not pieces:
        return None

    strip = cv2.hconcat(pieces)
    strip = _sharpen_and_clean(strip)
    
    out_path = out_dir / f"{crop_path.stem}_strip.png"
    cv2.imwrite(str(out_path), strip)

    print(f"  [Sentence] {crop_path.name}: Generated CCA strip ({len(line_crops)} lines).")
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stage3_Sentences.py <crop_image_path> <original_stem>")
        sys.exit(1)
    src = Path(sys.argv[1])
    out = src.parent.parent.parent / "Result" / "Sentences"
    out.mkdir(parents=True, exist_ok=True)
    run(src, out, sys.argv[2])
