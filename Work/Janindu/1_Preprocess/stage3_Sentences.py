"""
Stage 3 – Sentence Strip Stitching  (stage3_Sentences.py)
──────────────────────────────────────────────────────────
For every cropped image produced by Stage 2:

  1. Auto-rotate if the crop is vertical (taller than wide) → make horizontal.
  2. Use OpenCV horizontal projection to detect if there are multiple text
     lines (NO Surya model needed — instant, pure image processing).
  3. If multi-line: split into individual lines, resize each to 512px height,
     concatenate left-to-right.
  4. If single-line: just resize the whole crop to 512px height.

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
GAP_PX       = 8      # px – white gap between stitched lines
MIN_LINE_H   = 8      # px – ignore lines shorter than this


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_horizontal(img: np.ndarray) -> np.ndarray:
    """
    If the image is taller than it is wide (vertical text), rotate it 90°
    clockwise so text reads left-to-right horizontally.
    """
    h, w = img.shape[:2]
    if h > w * 1.3:  # clearly vertical
        # Rotate 90° clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _split_lines(img: np.ndarray) -> list[np.ndarray]:
    """
    Split an image into individual text lines using horizontal projection.

    How it works:
      - Convert to grayscale → binary (inverted so text pixels = white).
      - Sum each row → horizontal projection profile.
      - Find gaps (rows with very few ink pixels) → line boundaries.
      - Crop each line region.

    Returns a list of line images (may be just one if single-line).
    """
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Horizontal projection: count ink pixels per row
    projection = np.sum(bw, axis=1) / 255.0  # normalise to pixel count

    # Threshold: a row is "text" if it has more than 1% of width in ink
    ink_threshold = w * 0.01
    is_text_row = projection > ink_threshold

    # Find contiguous text-row runs
    lines: list[tuple[int, int]] = []
    in_line = False
    start = 0

    for y in range(h):
        if is_text_row[y] and not in_line:
            start = y
            in_line = True
        elif not is_text_row[y] and in_line:
            if y - start >= MIN_LINE_H:
                lines.append((start, y))
            in_line = False

    # Close last line if it reaches the bottom
    if in_line and h - start >= MIN_LINE_H:
        lines.append((start, h))

    if not lines:
        # Couldn't split — return the whole image as one line
        return [img]

    # Add a small vertical padding to each line
    pad = 2
    crops = []
    for (y1, y2) in lines:
        y1_pad = max(0, y1 - pad)
        y2_pad = min(h, y2 + pad)
        crops.append(img[y1_pad:y2_pad, :])

    return crops


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize so height == target_h, keeping aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_h, 10, 3), dtype=np.uint8) * 255
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _white_gap(height: int, width: int = None) -> np.ndarray:
    w = width if width else GAP_PX
    return np.ones((height, w, 3), dtype=np.uint8) * 255


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

    # ── Step 1: Split into lines (pure OpenCV, no model) ──────────────────────
    line_crops = _split_lines(img)

    if not line_crops:
        print(f"  [Sentence] No text found in {crop_path.name}, skipping.")
        return None

    # ── Step 3: Resize each line to STRIP_HEIGHT ──────────────────────────────
    resized = [_resize_to_height(lc, STRIP_HEIGHT) for lc in line_crops]

    # ── Step 4: Concatenate left-to-right with gaps ───────────────────────────
    pieces: list[np.ndarray] = []
    for i, s in enumerate(resized):
        pieces.append(s)
        if i < len(resized) - 1:
            pieces.append(_white_gap(STRIP_HEIGHT))

    strip = cv2.hconcat(pieces)

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
