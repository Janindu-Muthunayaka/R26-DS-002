import os
from PIL import Image, ImageDraw
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# 1. Setup Paths
input_dir = r"E:\Sliit\Research\Fonts\Surya"
output_dir = r"E:\Sliit\Research\Fonts\Surya\results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Initialize Predictors once (reuse across all files)
foundation_predictor = FoundationPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor)
det_predictor = DetectionPredictor()

# 3. Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 4. Get all image files in the input folder (excluding the results subfolder)
image_files = [
    f for f in os.listdir(input_dir)
    if os.path.isfile(os.path.join(input_dir, f))
    and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
]

if not image_files:
    print("No image files found in the input directory.")
else:
    print(f"Found {len(image_files)} image(s) to process...\n")

# 5. Process each image
for filename in image_files:
    image_path = os.path.join(input_dir, filename)
    print(f"Processing: {filename}")

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Run OCR
        predictions = rec_predictor(
            [image],
            task_names=["ocr_with_boxes"],
            det_predictor=det_predictor,
        )

        # Draw bounding boxes and print results
        draw = ImageDraw.Draw(image)
        for line in predictions[0].text_lines:
            poly_points = [p for point in line.polygon for p in point]
            draw.polygon(poly_points, outline="red", width=3)
            print(f"  Text: {line.text} | Confidence: {line.confidence:.2f}")

        # Save output with same name + _results suffix
        stem, ext = os.path.splitext(filename)
        output_filename = f"{stem}_results.png"
        output_path = os.path.join(output_dir, output_filename)
        image.save(output_path)
        print(f"  Saved to: {output_path}\n")

    except Exception as e:
        print(f"  ERROR processing {filename}: {e}\n")

print("Done! All images processed.")