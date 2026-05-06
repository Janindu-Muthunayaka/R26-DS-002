"""
Stage 3 – Sentence Strip Stitching  (stage3_Sentences.py)
──────────────────────────────────────────────────────────
For every cropped image produced by Stage 2:

  1. Auto-rotate if the crop is vertical (taller than wide) → make horizontal.
  2. Use OpenCV horizontal projection to detect if there are multiple text
     lines (NO Surya model needed — instant, pure image processing).
  3. If multi-line: split into individual lines, resize each to 512 px height,
     concatenate left-to-right.
  4. If single-line: just resize the whole crop to 512 px height.
  5. Apply unsharp-mask sharpening + Otsu re-binarise for crisp black
     letters on a pure white background.

Saves to:  Result/Sentences/<original_stem>/<crop_stem>_strip.png

This is MUCH faster than the old approach because it does NOT use Surya
for line detection — it uses simple pixel-level projection analysis.
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# No _ensure_horizontal here anymore, it incorrectly rotated tall blocks of horizontal text.


def _split_lines(img: np.ndarray) -> list[np.ndarray]:
    """
    Split an image into individual text lines using horizontal smearing (RLSA).
    This is more robust than simple projection for sparse text.
    """
    h, w = img.shape[:2]
    if h < MIN_LINE_H:
        return [img]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 1. Clean noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned_bw = np.zeros_like(bw)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 8:
            cleaned_bw[labels == i] = 255

    # 2. Horizontal Smearing: connect characters into lines
    # We use a wide horizontal kernel to ensure words in a line are connected
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    smeared = cv2.dilate(cleaned_bw, kernel_h, iterations=1)
    
    # 3. Vertical dilation to catch dots/accents
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    smeared = cv2.dilate(smeared, kernel_v, iterations=1)

    # 4. Find bounding boxes of smeared lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(smeared, connectivity=8)
    
    line_regions = []
    for i in range(1, num_labels):
        x_box = stats[i, cv2.CC_STAT_LEFT]
        y_box = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Filter out noise components that aren't text-like
        if h_box >= MIN_LINE_H and w_box > w * 0.02:
            line_regions.append((y_box, y_box + h_box))

    # 5. Sort lines by Y coordinate
    line_regions.sort(key=lambda x: x[0])

    if not line_regions:
        return [img]

    # 6. Merge overlapping or very close regions
    merged = []
    if line_regions:
        curr_y1, curr_y2 = line_regions[0]
        for i in range(1, len(line_regions)):
            next_y1, next_y2 = line_regions[i]
            if next_y1 < curr_y2 + 5: # overlap or tiny gap
                curr_y2 = max(curr_y2, next_y2)
            else:
                merged.append((curr_y1, curr_y2))
                curr_y1, curr_y2 = next_y1, next_y2
        merged.append((curr_y1, curr_y2))

    # 7. Extract crops with horizontal trimming and padding
    line_crops = []
    y_pad = 4
    for (y1, y2) in merged:
        y1_c = max(0, y1 - y_pad)
        y2_c = min(h, y2 + y_pad)
        line_img = img[y1_c:y2_c, :]
        
        # Trim horizontal white space
        line_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        _, line_bw = cv2.threshold(line_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coords = cv2.findNonZero(line_bw)
        if coords is not None:
            bx, by, bw_line, bh_line = cv2.boundingRect(coords)
            x_left = max(0, bx - LINE_PAD_H)
            x_right = min(w, bx + bw_line + LINE_PAD_H)
            line_crops.append(line_img[:, x_left:x_right])
        else:
            line_crops.append(line_img)

    return line_crops


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize so height == target_h, keeping aspect ratio.

    Uses INTER_AREA for shrinking (avoids moiré on binary images) and
    INTER_LANCZOS4 for enlarging.
    """
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
    """
    Final polish on the stitched strip.

    1. Mild unsharp mask (σ=1.0, amount=1.5) — just enough to crisp up edges
       that softened during resize.
    2. Otsu threshold — cleans up all remaining anti-alias fringe and forces
       perfect 0 (black) and 255 (white). We avoid adaptive threshold because
       it creates dots in white spaces.
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Mild unsharp mask
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Re-binarise with Otsu
    _, final_bw = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure black-on-white
    if np.sum(final_bw == 255) < np.sum(final_bw == 0):
        final_bw = cv2.bitwise_not(final_bw)

    return cv2.cvtColor(final_bw, cv2.COLOR_GRAY2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run(crop_path: Path, sentence_root: Path, original_stem: str) -> Path | None:
    """
    Build a sentence strip from `crop_path` and save it under
    `sentence_root/<original_stem>/<crop_stem>_strip.png`.

    Steps:
      1. Rotate to horizontal if needed.
      2. Split into lines using projection (no AI model).
      3. Resize each line to 512px height.
      4. Concatenate left-to-right.

    Returns the output path, or None on failure.
    """
    img = cv2.imread(str(crop_path))
    if img is None:
        print(f"  [Sentence] ERROR: cannot read {crop_path}")
        return None

    # ── Step 1: Split into lines (pure OpenCV, no model) ─────────────────────
    line_crops = _split_lines(img)

    if not line_crops:
        print(f"  [Sentence] No text found in {crop_path.name}, skipping.")
        return None

    print(f"  [Sentence] {crop_path.name}: detected {len(line_crops)} line(s).")

    # ── Step 2: Resize each line to STRIP_HEIGHT ──────────────────────────────
    resized = [_resize_to_height(lc, STRIP_HEIGHT) for lc in line_crops]

    # ── Step 3: Concatenate left-to-right with gaps ───────────────────────────
    pieces: list[np.ndarray] = []
    for i, s in enumerate(resized):
        pieces.append(s)
        if i < len(resized) - 1:
            pieces.append(_white_gap(STRIP_HEIGHT))

    strip = cv2.hconcat(pieces)

    # ── Step 4: Soft sharpen + adaptive clean for natural black-on-white ──────
    strip = _sharpen_and_clean(strip)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = sentence_root / original_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{crop_path.stem}_strip.png"
    cv2.imwrite(str(out_path), strip)
    print(f"  [Sentence] Strip -> {out_path.name}  "
          f"({strip.shape[1]}x{strip.shape[0]} px, {len(line_crops)} lines)")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stage3_Sentences.py <crop_image_path> <original_stem>")
        sys.exit(1)
    src = Path(sys.argv[1])
    out = src.parent.parent.parent / "Result" / "Sentences"
    out.mkdir(parents=True, exist_ok=True)
    run(src, out, sys.argv[2])
