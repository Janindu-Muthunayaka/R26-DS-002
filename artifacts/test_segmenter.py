import sys
from pathlib import Path
import cv2

# Set up paths
sys.path.insert(0, str(Path("E:/Sliit/Research/Main Repository/R26-DS-002/system")))
from core import config
config.set_root("E:/Sliit/Research/Main Repository/R26-DS-002/system")

from paddleocr import LayoutDetection
from layers.l3_segment.segment import Segmenter

img_path = "E:/Sliit/Research/Main Repository/R26-DS-002/system/work/deb45262/f0_g22_s2245.jpg"
img = cv2.imread(img_path)
if img is None:
    print("Could not read image!")
    sys.exit(1)

layout = LayoutDetection(model_name='PP-DocLayout_plus-L', threshold=0.20, device="cpu", enable_mkldnn=False)
seg = Segmenter(yolo=None, layout=layout)

print("Running segmentation...")
articles = seg.run(img)

print(f"Segmenter found {len(articles)} articles:")
for art in articles:
    print(f"Article {art.index}: box={art.box}, regions_count={len(art.regions)}")
