# =============================================================================
# stage1_config.py  —  Configuration, Constants & PipelineConfig
#
# All tunable parameters, path constants, GPU configuration, and the
# PipelineConfig dataclass live here.  Every other stage module imports
# from this file — never the other way around.
# =============================================================================

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch
import numpy as np
import cv2
from typing import Optional

# =============================================================================
# OPTIMIZATION GRID DEFINITIONS (Shared by Optimizers and Inference)
# =============================================================================

SWEPT_GRID = {
    "smoothing_k":         [3, 5],
    "close_k":             [2, 3],
    "window_pad":          [4, 8, 12],
    "multi_seg_threshold": [99.0],
}

SWEPT_PARAMS = ["smoothing_k", "close_k", "window_pad", "multi_seg_threshold"]

PARAM_LIMITS = {
    "smoothing_k":         (1, 9),
    "close_k":             (1, 9),
    "window_pad":          (2, 40),
    "multi_seg_threshold": (80.0, 100.0),
}

# =============================================================================
# FEATURE EXTRACTION  (used by Optimizers and Inference)
# =============================================================================

FEATURE_KEYS = [
    "pixel_density", "foreground_fraction", "noise_coefficient",
    "mean_stroke_width", "aspect_ratio", "horizontal_projection_variance",
    "vertical_projection_variance", "edge_density", "local_contrast",
    "n_components", "mean_gap_px", "std_gap_px",
    "image_height_px", "image_width_px",
]

def extract_image_features(img_path: str) -> Optional[dict]:
    """Extract 14 statistical features from an image for parameter adaptation."""
    raw = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur       = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary  = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    ink_mask = (binary > 0)

    pixel_density        = float(ink_mask.sum()) / (h * w)
    foreground_fraction  = float((gray < 128).sum()) / (h * w)
    bg_pixels            = gray[~ink_mask]
    noise_coefficient    = float(bg_pixels.std()) if len(bg_pixels) > 0 else 0.0

    dist              = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_dists         = dist[ink_mask]
    mean_stroke_width = float(ink_dists.mean() * 2) if len(ink_dists) > 0 else 1.0
    aspect_ratio      = float(w) / float(h) if h > 0 else 1.0

    h_proj = ink_mask.sum(axis=0).astype(float)
    v_proj = ink_mask.sum(axis=1).astype(float)
    horizontal_projection_variance = float(h_proj.var())
    vertical_projection_variance   = float(v_proj.var())

    edges        = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).sum()) / (h * w)

    resized_32     = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    local_contrast = float(resized_32.astype(float).std())

    n_components, _ = cv2.connectedComponents(binary)
    n_components    = max(0, int(n_components) - 1)

    projection = ink_mask.sum(axis=0).astype(int)
    gap_lengths: list[int] = []
    in_gap    = False
    gap_start = 0
    for x, val in enumerate(projection):
        if val == 0 and not in_gap:
            gap_start = x
            in_gap    = True
        elif val > 0 and in_gap:
            gap_lengths.append(x - gap_start)
            in_gap = False
    mean_gap_px = float(np.mean(gap_lengths)) if gap_lengths else 0.0
    std_gap_px  = float(np.std(gap_lengths))  if gap_lengths else 0.0

    return {
        "pixel_density":                   round(pixel_density, 6),
        "foreground_fraction":             round(foreground_fraction, 6),
        "noise_coefficient":               round(noise_coefficient, 4),
        "mean_stroke_width":               round(mean_stroke_width, 4),
        "aspect_ratio":                    round(aspect_ratio, 4),
        "horizontal_projection_variance":  round(horizontal_projection_variance, 4),
        "vertical_projection_variance":    round(vertical_projection_variance, 4),
        "edge_density":                    round(edge_density, 6),
        "local_contrast":                  round(local_contrast, 4),
        "n_components":                    n_components,
        "mean_gap_px":                     round(mean_gap_px, 4),
        "std_gap_px":                      round(std_gap_px, 4),
        "image_height_px":                 h,
        "image_width_px":                  w,
    }


# =============================================================================
# ██████████████████████  PATH CONSTANTS  █████████████████████████████████████
# =============================================================================

INPUT_FOLDER   = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\InputFolder\Full30k\Images"
OUTPUT_FOLDER  = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\OutputFolder"
RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "Results")
INFO_FOLDER    = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\Information"
LABEL_CSV      = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\InputFolder\Full30k\Label_List.csv"
TESSERACT_CSV = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\InputFolder\Tessaract_Result_TextCleaned.csv"
MODEL_PATH    = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\0_Data\Model\final_model.pth"
CLASS_MAP     = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\0_Data\Model\class_mapping.json"
VARIANTS_PATH = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\Information\Variants.py"
CLASS_MAP_CSV = r"E:\Sliit\Research\Main Repository\R26-DS-002\Work\Janindu\2_Recogniton\Information\class_map.csv"
WORK_ROOT     = INFO_FOLDER

# =============================================================================
# ██████████████████████  SAMPLING DEFAULTS  ██████████████████████████████████
# =============================================================================

DEFAULT_FULLSET = False
DEFAULT_SAMPLE  = 100
DEFAULT_SEED    = 42

# =============================================================================
# ██████████████████████  STAGE-2 PREPROCESSING PARAMETERS (TUNABLE)  █████████
# =============================================================================

P_TARGET_HEIGHT    = 512
P_SMOOTHING_K      = 3
P_CLOSE_K          = 3
P_SKELETON_DIL     = 1
P_VALLEY_MIN_WIDTH = 2

# =============================================================================
# ██████████████████████  STAGE-3 SEGMENTATION PARAMETERS (TUNABLE)  █████████
# =============================================================================

# Minimum pixel area for a connected component to be treated as a valid blob.
# Components smaller than this are discarded as noise.
# Tune upward if stray ink specks are being counted as characters.
P_BLOB_MIN_AREA = 15      # Reduced from 30 to capture smaller components
P_RECT_THRESHOLD  = 1.3    # Width/Height ratio to trigger thin-valley splitting
P_THIN_RATIO      = 0.05   # Ink height ratio (ink/total_h) to treat as a 'thin' valley
P_MIN_SPLIT_DIST  = 12     # Min horizontal pixels between character splits

# =============================================================================
# ██████████████████████  STAGE-4 CLASSIFIER PARAMETERS (TUNABLE)  ████████████
# =============================================================================

P_CHAR_CANVAS_SIZE    = 384   # ← do NOT change unless model is retrained
P_WINDOW_PAD          = 12
P_MULTI_SEG_THRESHOLD = 97.0

# =============================================================================
# ██████████████████████  WORD-SPACER PARAMETERS (TUNABLE)  ███████████████████
# =============================================================================

P_WORD_SPACER_ENABLED = True
P_WORD_GAP_PX         = 50

# =============================================================================
# ██████████████████████  STAGE-4 MISC  ███████████████████████████████████████
# =============================================================================

TOP_K = 5   # alternative predictions to retain — not tunable

# =============================================================================
# ██████████████████████  ImageNet NORMALISATION  █████████████████████████████
# =============================================================================

_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# =============================================================================
# ██████████████████████  GPU / HARDWARE CONFIG  ██████████████████████████████
# =============================================================================

# Maximum crops sent to GPU in one forward pass.
INFER_BATCH_SIZE: int = 64

# Stage-2 parallel workers.
S1_WORKERS: int = max(1, math.ceil((os.cpu_count() or 8) * 0.75))

# PNG write pool size (I/O bound, kept small).
PNG_POOL_WORKERS: int = 4

# Enable CUDA benchmark mode for fixed-size inference loops.
CUDNN_BENCHMARK: bool = True

# =============================================================================
# ██████████████████████  DEVICE SELECTION  ███████████████████████████████████
# =============================================================================

def _select_device() -> torch.device:
    """
    Returns cuda:0 if available, else cpu.
    Applies cudnn.benchmark and prints a one-line device summary.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = False
        props = torch.cuda.get_device_properties(dev)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"  [Device] CUDA  - {props.name}  "
              f"VRAM {vram_gb:.1f} GB  "
              f"cudnn.benchmark={CUDNN_BENCHMARK}")
        if vram_gb < 4.0:
            print(f"  [Device] WARNING: < 4 GB VRAM detected. "
                  f"Consider reducing INFER_BATCH_SIZE (currently {INFER_BATCH_SIZE}).")
    else:
        dev = torch.device("cpu")
        print(f"  [Device] CPU   - CUDA not available")
    return dev


# Module-level device resolved once at import time.
_DEVICE: torch.device = _select_device()

# =============================================================================
# ██████████████████████  PIPELINE CONFIG DATACLASS  ██████████████████████████
# =============================================================================

@dataclass
class PipelineConfig:
    """Single object carrying every tunable parameter through the pipeline."""
    # Paths
    input_folder:        str   = INPUT_FOLDER
    output_folder:       str   = OUTPUT_FOLDER
    results_folder:      str   = RESULTS_FOLDER
    info_folder:         str   = INFO_FOLDER
    label_csv:           str   = LABEL_CSV
    model_path:          str   = MODEL_PATH
    class_map:           str   = CLASS_MAP
    variants_path:       str   = VARIANTS_PATH
    class_map_csv:       str   = CLASS_MAP_CSV
    work_root:           str   = WORK_ROOT

    # Sampling
    fullset:             bool  = DEFAULT_FULLSET
    sample:              int   = DEFAULT_SAMPLE
    seed:                int   = DEFAULT_SEED

    # Stage-2 preprocessing
    target_height:       int   = P_TARGET_HEIGHT
    smoothing_k:         int   = P_SMOOTHING_K
    close_k:             int   = P_CLOSE_K
    skeleton_dil:        int   = P_SKELETON_DIL
    valley_min_width:    int   = P_VALLEY_MIN_WIDTH

    # Stage-3 segmentation
    blob_min_area:       int   = P_BLOB_MIN_AREA
    rect_threshold:      float = P_RECT_THRESHOLD
    thin_ratio:          float = P_THIN_RATIO
    min_split_dist:      int   = P_MIN_SPLIT_DIST

    # Stage-4 classifier
    char_canvas_size:    int   = P_CHAR_CANVAS_SIZE
    window_pad:          int   = P_WINDOW_PAD
    multi_seg_threshold: float = P_MULTI_SEG_THRESHOLD
    variants_path:       str   = VARIANTS_PATH

    # Word spacer
    word_spacer_enabled: bool  = P_WORD_SPACER_ENABLED
    word_gap_px:         int   = P_WORD_GAP_PX

    def as_param_dict(self) -> dict:
        return {
            "target_height":       self.target_height,
            "smoothing_k":         self.smoothing_k,
            "close_k":             self.close_k,
            "skeleton_dil":        self.skeleton_dil,
            "valley_min_width":    self.valley_min_width,
            "blob_min_area":       self.blob_min_area,
            "rect_threshold":      self.rect_threshold,
            "thin_ratio":          self.thin_ratio,
            "min_split_dist":      self.min_split_dist,
            "window_pad":          self.window_pad,
            "multi_seg_threshold": self.multi_seg_threshold,
            "word_spacer_enabled": self.word_spacer_enabled,
            "word_gap_px":         self.word_gap_px,
        }