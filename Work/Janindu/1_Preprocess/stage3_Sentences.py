"""
Stage 3 – Sentence Strip Stitching  (stage3_Sentences.py)
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

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# ── Tunables ──────────────────────────────────────────────────────────────────
STRIP_HEIGHT = 384    # px – uniform height for the output strip
GAP_PX       = 40     # px – white gap between stitched line pieces
MIN_LINE_H   = 8      # px – skip lines shorter than this (degenerate guard)
MIN_LINE_W   = 20     # px – skip lines narrower than this


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
        stitched = _stitch(line_imgs)
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

        stitched = _stitch(line_imgs)
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

    stitched = _stitch([img])
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
