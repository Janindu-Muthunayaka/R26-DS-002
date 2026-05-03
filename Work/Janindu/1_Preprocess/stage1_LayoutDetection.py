"""
Stage 1 – Text Region Detection & Block Merging (stage1_LayoutDetection.py)
──────────────────────────────────────────────────────────────────────────
1. Uses Surya's DetectionPredictor to find text lines.
2. Extracts POLYGONS (4 points) to handle tilted/skewed text.
3. MERGES nearby lines into Paragraph Blocks to satisfy "capture the whole block".
4. Saves annotated image with RED outlines and JSON metadata.

JSON metadata includes both axis-aligned BBoxes and 4-point Polygons.
"""

from __future__ import annotations

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from PIL import Image, ImageDraw

# Ensure console can handle non-ASCII text
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Lazy model singleton ─────────────────────────────────────────────────────
_det_predictor = None

def _get_det_predictor():
    global _det_predictor
    if _det_predictor is None:
        from surya.detection import DetectionPredictor
        print("  [LayoutDet] Loading Surya detection model ...")
        _det_predictor = DetectionPredictor()
        print("  [LayoutDet] Model ready.")
    return _det_predictor

# ── Data class for detected boxes ─────────────────────────────────────────────
@dataclass
class LayoutBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    polygon: list[list[int]] = None  # List of 4 [x, y] points

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_AREA       = 200    
MIN_CONFIDENCE = 0.50   

# Merging Tunables
VERTICAL_GAP_THRESH   = 20  # Max vertical distance between lines to merge (pixels)
HORIZONTAL_OVERLAP_THRESH = 0.2 # Min horizontal overlap ratio to merge

def _compute_horizontal_overlap(box1, box2):
    """Check how much box1 and box2 overlap horizontally."""
    x_left = max(box1.x1, box2.x1)
    x_right = min(box1.x2, box2.x2)
    overlap = max(0, x_right - x_left)
    w1 = box1.x2 - box1.x1
    w2 = box2.x2 - box2.x1
    if min(w1, w2) == 0: return 0
    return overlap / min(w1, w2)

def _merge_boxes(boxes: list[LayoutBox]) -> list[LayoutBox]:
    """
    Groups individual lines into Paragraph blocks.
    """
    if not boxes: return []
    
    # Sort boxes by top y coordinate
    sorted_boxes = sorted(boxes, key=lambda b: b.y1)
    
    merged: list[list[LayoutBox]] = []
    
    for box in sorted_boxes:
        found_group = False
        # Try to find a group to join
        for group in merged:
            # Get the bottom-most y of the current group
            group_y2 = max(b.y2 for b in group)
            group_x1 = min(b.x1 for b in group)
            group_x2 = max(b.x2 for b in group)
            
            # Simple dummy box for overlap calculation
            group_box = LayoutBox("", group_x1, 0, group_x2, 0)
            
            # Check if this box is close to the bottom of the group
            v_gap = box.y1 - group_y2
            h_overlap = _compute_horizontal_overlap(box, group_box)
            
            if v_gap < VERTICAL_GAP_THRESH and h_overlap > HORIZONTAL_OVERLAP_THRESH:
                group.append(box)
                found_group = True
                break
        
        if not found_group:
            merged.append([box])
            
    # Convert groups back to LayoutBox blocks
    result: list[LayoutBox] = []
    for idx, group in enumerate(merged):
        x1 = min(b.x1 for b in group)
        y1 = min(b.y1 for b in group)
        x2 = max(b.x2 for b in group)
        y2 = max(b.y2 for b in group)
        
        # Calculate a combined polygon for the block
        # For simplicity, we'll take the minAreaRect of all points in the group
        all_points = []
        for b in group:
            if b.polygon:
                all_points.extend(b.polygon)
            else:
                all_points.extend([[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]])
        
        # Use minAreaRect to get a tilted box that fits the points
        pts = np.array(all_points, dtype=np.int32)
        rect = cv2.minAreaRect(pts)
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points).tolist() # 4 points [[x,y], ...]
        
        result.append(LayoutBox(
            label=f"Block_{idx:02d}",
            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            confidence=sum(b.confidence for b in group) / len(group),
            polygon=box_points
        ))
        
    return result

def run(img_path: Path, out_dir: Path) -> tuple[Path | None, list[LayoutBox]]:
    if not img_path.exists():
        print(f"  [LayoutDet] ERROR: file not found {img_path}")
        return None, []

    image = Image.open(str(img_path)).convert("RGB")
    det_predictor = _get_det_predictor()

    det_predictions = det_predictor([image])
    det_result = det_predictions[0]

    w_img, h_img = image.size
    line_boxes: list[LayoutBox] = []

    for i, bbox in enumerate(det_result.bboxes):
        x1, y1, x2, y2 = bbox.bbox
        conf = bbox.confidence
        poly = getattr(bbox, "polygon", None)

        if conf < MIN_CONFIDENCE: continue
        
        # Clamp coordinates
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

        if (x2 - x1) * (y2 - y1) < MIN_AREA: continue

        line_boxes.append(LayoutBox(
            label=f"Line_{i:02d}",
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=conf,
            polygon=poly
        ))

    # ── Merge lines into blocks ──────────────────────────────────────────────
    blocks = _merge_boxes(line_boxes)
    print(f"  [LayoutDet] Merged {len(line_boxes)} lines into {len(blocks)} blocks.")

    # ── Draw boxes ───────────────────────────────────────────────────────────
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for block in blocks:
        # Draw the tilted polygon in RED
        if block.polygon:
            # Draw lines between points
            p = block.polygon
            for i in range(4):
                draw.line([tuple(p[i]), tuple(p[(i+1)%4])], fill="red", width=3)
        else:
            draw.rectangle([block.x1, block.y1, block.x2, block.y2], outline="red", width=3)
            
        label = f"{block.label} ({block.confidence:.0%})"
        draw.text((block.x1 + 2, block.y1 - 14), label, fill="red")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_img = out_dir / f"{img_path.stem}_boxes.png"
    annotated.save(str(out_img))

    out_json = out_dir / f"{img_path.stem}_boxes.json"
    json_data = {
        "source_image": str(img_path),
        "image_size": {"width": w_img, "height": h_img},
        "boxes": [asdict(b) for b in blocks],
    }
    out_json.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return out_img, blocks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage1_LayoutDetection.py <image_path>")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = src.parent / "Result" / "LayoutDetection"
    dst.mkdir(parents=True, exist_ok=True)
    run(src, dst)
