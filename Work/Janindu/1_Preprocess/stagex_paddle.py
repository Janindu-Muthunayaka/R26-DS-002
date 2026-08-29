"""
Template for integrating PaddleOCR's PP-Structure Layout Recovery 
with the existing DocPreprocessor (Unwarping).

This script demonstrates how to leverage PP-Structure's native ability 
to correctly sequence multi-column layouts (Layout Recovery) instead of 
relying on naive left-to-right, row-by-row sorting.
"""

import cv2
from pathlib import Path
from dataclasses import dataclass

# In a real environment, you would import these:
# from paddleocr import PPStructure, DocPreprocessor

@dataclass
class LayoutBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

def get_pipeline():
    """
    Initialize the full pipeline, including Layout Recovery.
    """
    # 1. Orientation & Unwarping (Same as stage1_LayoutDetection.py)
    # preprocessor = DocPreprocessor(
    #     doc_unwarping_model_dir=r"C:\Users\JANINDU\.paddlex\official_models\UVDoc",
    #     doc_orientation_classify_model_dir=r"C:\Users\JANINDU\.paddlex\official_models\PP-LCNet_x1_0_doc_ori",
    #     use_doc_orientation_classify=True,
    #     use_doc_unwarping=True,
    #     device="gpu:0"
    # )
    
    # 2. PPStructure for Layout Analysis & Reading Order Recovery
    # Setting recovery=True is the key to enabling PaddleOCR's intelligent 
    # multi-column reading order logic.
    # structure_engine = PPStructure(
    #     recovery=True,
    #     layout_model_dir=r"C:\Users\JANINDU\.paddlex\official_models\PP-DocLayout_plus-L",
    #     device="gpu:0",
    #     show_log=True
    # )
    # return preprocessor, structure_engine
    pass

def run_ppstructure_template(img_path: Path) -> list[LayoutBox]:
    """
    Process an image to get properly sequenced LayoutBoxes.
    """
    img = cv2.imread(str(img_path))
    if img is None: 
        return []

    # preprocessor, structure_engine = get_pipeline()

    # ── Step 1: Unwarp image ──────────────────────────────────────────────────
    # pre_res = preprocessor.predict(str(img_path))[0]
    # corrected_img = pre_res.get("output_img", img)
    corrected_img = img # dummy

    # ── Step 2: Extract Layout and Reading Order ──────────────────────────────
    # The engine returns a list of layout elements ALREADY SORTED in the 
    # correct reading order. It traces down columns before moving right!
    # result = structure_engine(corrected_img)
    
    result = [] # Dummy empty output
    
    ordered_boxes = []
    for region in result:
        region_type = region.get('type') # e.g. 'title', 'text', 'figure'
        bbox = region.get('bbox')        # [x1, y1, x2, y2]
        
        # Keep only text elements
        if region_type in ['text', 'title', 'document_title', 'paragraph_title']:
            if bbox:
                x1, y1, x2, y2 = bbox
                ordered_boxes.append(LayoutBox(
                    label=region_type,
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)
                ))

    # ──────────────────────────────────────────────────────────────────────────
    # CRITICAL: Do NOT manually sort ordered_boxes by Y/X coordinates here!
    # 
    # Example of what NOT to do:
    # ordered_boxes.sort(key=lambda bx: (bx.y1 // 100, bx.x1)) 
    #
    # The result list is already in the exact sequence it needs to be read.
    # Doing a custom sort will destroy the multi-column layout recovery.
    # ──────────────────────────────────────────────────────────────────────────

    return ordered_boxes


if __name__ == "__main__":
    # Dummy execution for reference
    sample_path = Path("sample_input.png")
    ordered_layout = run_ppstructure_template(sample_path)
    print(f"Generated {len(ordered_layout)} layout regions in correct reading order.")
