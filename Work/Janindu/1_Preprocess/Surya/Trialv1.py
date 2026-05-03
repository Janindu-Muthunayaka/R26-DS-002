import os
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# 1. Setup Paths
image_path = r"E:\Sliit\Research\Fonts\Surya\real.png"
output_dir = r"E:\Sliit\Research\Fonts\Surya\results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Initialize Predictors - FoundationPredictor is required
foundation_predictor = FoundationPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor)  # required arg
det_predictor = DetectionPredictor()

# 3. Load Image
image = Image.open(image_path).convert("RGB")

# 4. Run OCR
# task_names must be one of: 'ocr_with_boxes', 'ocr_without_boxes', etc.
# Languages are NOT a separate argument in this version - they are handled internally
predictions = rec_predictor(
    [image],
    task_names=["ocr_with_boxes"],   # valid task, not language codes
    det_predictor=det_predictor,
)

# 5. Draw and Print Results
draw = ImageDraw.Draw(image)
for line in predictions[0].text_lines:
    poly_points = [p for point in line.polygon for p in point]
    draw.polygon(poly_points, outline="red", width=3)
    print(f"Text: {line.text} | Confidence: {line.confidence:.2f}")

# 6. Save Output
output_path = os.path.join(output_dir, "FileC_results.png")
image.save(output_path)
print(f"Success! Result saved to {output_path}")