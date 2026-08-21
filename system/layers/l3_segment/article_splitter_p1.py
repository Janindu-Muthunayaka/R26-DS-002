"""
Stage 1 – Layout Detection & Unwarping  (article_splitter_p1.py)
────────────────────────────────────────────────────────────────────
Uses PaddleOCR PPStructureV3 to:
  1. Correct perspective / orientation of angled document photos.
  2. Detect semantic layout blocks (title, body text, header, footer …).
  3. Save an annotated layout visualisation and a corrected full-page image.

Returns a list of LayoutBox objects (same interface as the old Surya stage)
so that Stage 2 / Stage 3 require no structural changes.

PaddleOCR layout labels used (PP-DocLayout_plus-L, 23 classes):
  document_title, paragraph_title, text, header, footer, page_number,
  image_caption, aside_text  →  kept as text-bearing regions
  Everything else (image, table, formula …) → ignored
"""

from __future__ import annotations

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

# Ensure console can handle non-ASCII text
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Lazy model singleton ──────────────────────────────────────────────────────
_doc_preprocessor = None
_layout_detector = None

def _get_pipeline():
    global _doc_preprocessor, _layout_detector
    if _doc_preprocessor is None:
        from paddleocr import DocPreprocessor, LayoutDetection
        print("  [LayoutDet] Loading specialized models (Preprocessor + Layout) …")
        
        # Load ONLY the orientation and unwarping models
        _doc_preprocessor = DocPreprocessor(
            doc_unwarping_model_dir=r"C:\Users\JANINDU\.paddlex\official_models\UVDoc",
            doc_orientation_classify_model_dir=r"C:\Users\JANINDU\.paddlex\official_models\PP-LCNet_x1_0_doc_ori",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            device="gpu:0"
        )
        
        # Load ONLY the layout detection model
        _layout_detector = LayoutDetection(
            model_dir=r"C:\Users\JANINDU\.paddlex\official_models\PP-DocLayout_plus-L",
            threshold=SCORE_THRESHOLD,
            device="gpu:0"
        )
        print("  [LayoutDet] Models ready. (OCR components skipped for speed)")
    return _doc_preprocessor, _layout_detector


# ── Data class (same interface as the original Surya-based stage) ─────────────
@dataclass
class LayoutBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    polygon: list[list[int]] | None = None   # 4-point rectangle [[x,y],…]


# ── Tunables ──────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.20   # Lowered to 0.20 to be "more observant" of subtle regions

# Text-bearing label groups (from PP-DocLayout_plus-L)
# (Headers, footers, and page numbers are intentionally excluded)
TEXT_LABELS = {
    "text", "document_title", "paragraph_title",
    "image_caption", "aside_text",
}

# Colour map for the visualisation overlay (BGR)
_VIZ_COLOURS = {
    "text":             (0, 200, 0),
    "document_title":   (255, 50, 50),
    "paragraph_title":  (200, 130, 0),
    "header":           (0, 80, 255),
    "footer":           (0, 60, 200),
    "page_number":      (0, 180, 180),
    "image_caption":    (180, 0, 180),
    "aside_text":       (100, 200, 100),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _label_to_category(label: str) -> str:
    """Map a PaddleOCR layout label to one of: header / footer / title / body."""
    if label in ("header", "page_number"):
        return "header"
    if label == "footer":
        return "footer"
    if label in ("document_title", "paragraph_title"):
        return "title"
    return "body"
    
def _calculate_overlap_ratio(box1: LayoutBox, box2: LayoutBox) -> float:
    """
    Calculate the Intersection over Minimum Area (IoMin) of two LayoutBoxes.
    This effectively detects if one box is completely (or mostly) inside another.
    """
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    
    min_area = min(area1, area2)
    return intersection / min_area if min_area > 0 else 0.0

def _paddle_boxes_to_layout_boxes(paddle_boxes: list[dict]) -> list[LayoutBox]:
    """
    Convert PaddleOCR layout_det_res['boxes'] dicts to LayoutBox objects,
    applying specific filtering logic for body vs non-body blocks.
    """
    body_boxes: list[LayoutBox] = []
    non_body_boxes: list[LayoutBox] = []

    for i, b in enumerate(paddle_boxes):
        if b.get("score", 0) < SCORE_THRESHOLD:
            continue
        if b.get("label", "") not in TEXT_LABELS:
            continue

        x1, y1, x2, y2 = [int(c) for c in b["coordinate"]]
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        cat = _label_to_category(b['label'])

        box_obj = LayoutBox(
            label=f"{cat}_{i:03d}",
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=float(b["score"]),
            polygon=polygon,
        )

        if cat == "body":
            body_boxes.append(box_obj)
        else:
            non_body_boxes.append(box_obj)

    # ── Filtering Logic ──────────────────────────────────────────────────────
    if non_body_boxes:
        # User is strictly focusing on titles/headings
        final_boxes = non_body_boxes
    else:
        # Fallback Rule: If PaddleOCR failed to label any block as a title,
        # we guess the title by taking the 3 largest blocks on the page.
        if body_boxes:
            # Sort by area descending to find the most prominent blocks
            body_boxes.sort(key=lambda bx: (bx.x2 - bx.x1) * (bx.y2 - bx.y1), reverse=True)
            # Take up to the 3 largest blocks
            final_boxes = body_boxes[:3]
        else:
            final_boxes = []

    # Filter out nested/overlapping blocks (Overlap > 0.5)
    filtered_boxes = []
    # Sort by AREA descending, so we evaluate and keep the largest encompassing boxes first!
    final_boxes.sort(key=lambda bx: (bx.x2 - bx.x1) * (bx.y2 - bx.y1), reverse=True)
    
    for box in final_boxes:
        is_duplicate = False
        for kept_box in filtered_boxes:
            # If the current box is completely or mostly inside a larger kept box
            if _calculate_overlap_ratio(box, kept_box) > 0.6:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_boxes.append(box)

    # Sort in approximate reading order (row-by-row, then left-to-right)
    filtered_boxes.sort(key=lambda bx: (bx.y1 // 100, bx.x1))
    return filtered_boxes


def _save_layout_viz(img_bgr: np.ndarray,
                     boxes: list,
                     out_path: Path) -> None:
    """Draw labelled coloured rectangles on the corrected image and save."""
    viz = img_bgr.copy()
    for b in boxes:
        if isinstance(b, dict):
            if b.get("score", 0) < SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(c) for c in b["coordinate"]]
            label = b.get("label", "")
            score = b.get("score", 0)
        else:
            # LayoutBox object (filtered result)
            x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
            # Extract category name (removes the _001 suffix)
            label = b.label.split("_")[0]
            score = b.confidence

        colour = _VIZ_COLOURS.get(label, (128, 128, 128))
        cv2.rectangle(viz, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(viz, f"{label} {score:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imwrite(str(out_path), viz)


# ── Public API ────────────────────────────────────────────────────────────────
def run(img_path: Path, out_dir: Path) -> tuple[Path | None, Path | None, list[LayoutBox]]:
    """
    Stage 1 entry point.
    Creates a subfolder for the image and saves:
      A_corrected.png - the unwarped full-page image
      B_layout.png    - layout visualization with boxes
      C_refined.png   - the refined image (for now, same as A)
      <stem>.json     - metadata
    """
    if not img_path.exists():
        print(f"  [LayoutDet] ERROR: file not found {img_path}")
        return None, None, []

    # Create subfolder per image
    image_subfolder = out_dir / img_path.stem
    image_subfolder.mkdir(parents=True, exist_ok=True)

    # ── Run Models ───────────────────────────────────────────────────────────
    print(f"  [LayoutDet] Processing {img_path.name} (Unwarp + Layout) …")
    preprocessor, detector = _get_pipeline()
    
    # 1. Perspective Correction & Orientation
    pre_res = preprocessor.predict(str(img_path))[0]
    full_img = pre_res.get("output_img")
    
    corrected_img_path = image_subfolder / "A_corrected.png"
    if full_img is None:
        # Fallback if unwarping failed
        full_img = cv2.imread(str(img_path))
    
    cv2.imwrite(str(corrected_img_path), full_img)

    # 2. Layout Detection on the corrected image
    # Use 1280 resolution limit here too for memory safety
    layout_res = detector.predict(full_img)[0]
    paddle_boxes = layout_res.get("boxes", [])
    
    print(f"  [LayoutDet] Detected {len(paddle_boxes)} raw layout regions.")
    boxes = _paddle_boxes_to_layout_boxes(paddle_boxes)
    print(f"  [LayoutDet] Kept {len(boxes)} text-bearing blocks.")

    # ── Save B_layout (visualisation: RAW) ───────────────────────────────────
    layout_viz_path = image_subfolder / "B_layout.png"
    _save_layout_viz(full_img, paddle_boxes, layout_viz_path)

    # ── Save C_refined (visualisation: FILTERED) ─────────────────────────────
    refined_img_path = image_subfolder / "C_refined.png"
    _save_layout_viz(full_img, boxes, refined_img_path)

    # ── Save JSON metadata ────────────────────────────────────────────────────
    h_img, w_img = full_img.shape[:2]
    out_json = image_subfolder / f"{img_path.stem}.json"
    json_data = {
        "source_image":    str(img_path),
        "corrected_image": "A_corrected.png",
        "layout_image":    "B_layout.png",
        "refined_image":   "C_refined.png",
        "image_size":      {"width": w_img, "height": h_img},
        "boxes":           [asdict(b) for b in boxes],
    }
    out_json.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return layout_viz_path, corrected_img_path, boxes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage1_LayoutDetection.py <image_path>")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = src.parent / "Result" / "LayoutDetection"
    dst.mkdir(parents=True, exist_ok=True)
    run(src, dst)
