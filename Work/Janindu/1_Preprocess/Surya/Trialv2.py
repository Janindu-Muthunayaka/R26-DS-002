import os
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# 1. Setup Paths
image_path = r"E:\Sliit\Research\Research Trials - Stage 3\Datasets\B.jpeg"
output_dir = r"E:\Sliit\Research\Research Trials - Stage 3\Code\Surya\result"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Initialize ONLY what we need - skip RecognitionPredictor entirely
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()

# 3. Load Image
image = Image.open(image_path).convert("RGB")

# 4. Run Detection ONLY - no OCR/recognition
det_predictions = det_predictor([image])

# 5. Draw Bounding Boxes
draw = ImageDraw.Draw(image)

for bbox in det_predictions[0].bboxes:
    # bbox has .bbox = [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox.bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    print(f"BBox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) | Confidence: {bbox.confidence:.2f}")

print(f"\nTotal text regions found: {len(det_predictions[0].bboxes)}")

# 6. Save Output
output_path = os.path.join(output_dir, "FileC_bbox_results.png")
image.save(output_path)
print(f"Result saved to {output_path}")