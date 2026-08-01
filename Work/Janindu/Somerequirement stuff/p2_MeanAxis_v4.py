"""
Character Mean Axis Extractor v3 - Batch Mode
Reads all *_Edges_Stg5-Denoise.png files from Edges_Output folder.

Pipeline:
  0. Polarity detection (three-way vote) → normalise to black text on white
  1. Distance transform → gradient field for yellow lines
  2. Yellow inward gradient lines
  3. Red medial axis skeleton overlaid on yellow

Outputs (saved to MeanAxis_Output):
  Stg1-Polarity  : original image with polarity sample zones drawn
  Stg2-Distance  : distance transform as colour heatmap (diagnostic)
  Stg3-Yellow    : inward gradient lines on letters
  Stg4-Red       : skeleton overlaid on yellow lines
"""

import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_FOLDER   = r"E:\Sliit\Research\Fonts\TestFontV4"
INPUT_FOLDER  = os.path.join(BASE_FOLDER, "Edges_Output")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "MeanAxis_Output")

# ─── POLARITY DETECTION ───────────────────────────────────────────────────────
# Fraction of smaller image dimension used for corner/border sampling.
# Range: 0.03–0.10
POLARITY_SAMPLE_FRACTION = 0.05

# ─── YELLOW LINE SETTINGS ─────────────────────────────────────────────────────
# Edge sampling grid size in pixels — one point sampled per grid cell.
# Smaller = denser lines (slower), larger = fewer lines (faster).
# Range: 5–20
EDGE_SAMPLE_GRID = 10
# Maximum steps each gradient line follows inward.
GRADIENT_MAX_STEPS = 500
# Yellow line colour (BGR)
COLOUR_YELLOW = (0, 220, 255)

# ─── SKELETON SETTINGS ────────────────────────────────────────────────────────
# Skeleton dilation iterations for visibility. 1 = thin, 2 = thicker.
SKELETON_DILATE = 1
# Red skeleton colour (BGR)
COLOUR_RED = (0, 0, 220)

# ─── VISUALISATION ────────────────────────────────────────────────────────────
COLOUR_CORNER = (0,   200, 255)   # yellow-orange — corner sample boxes
COLOUR_BORDER = (255, 130,   0)   # blue — border strip line
# ──────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_png(path, img):
    _, buf = cv2.imencode(".png", img)
    with open(path, "wb") as f:
        f.write(buf)


def detect_polarity(gray, h, w):
    """
    Three-way vote: corners + border strip + histogram.
    Returns True if background is dark (needs inversion).
    """
    patch      = max(5, int(min(h, w) * POLARITY_SAMPLE_FRACTION))
    votes_dark = 0

    # Signal 1: four corners
    corners = [
        gray[:patch, :patch],
        gray[:patch, w - patch:],
        gray[h - patch:, :patch],
        gray[h - patch:, w - patch:],
    ]
    if np.mean([c.mean() for c in corners]) < 128:
        votes_dark += 1

    # Signal 2: border strip
    border = np.concatenate([
        gray[:patch, :].flatten(),
        gray[-patch:, :].flatten(),
        gray[:, :patch].flatten(),
        gray[:, -patch:].flatten(),
    ])
    if np.mean(border) < 128:
        votes_dark += 1

    # Signal 3: histogram mass
    hist        = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_smooth = np.convolve(hist, np.ones(15) / 15, mode='same')
    if hist_smooth[:128].sum() > hist_smooth[128:].sum():
        votes_dark += 1

    return votes_dark >= 2


def draw_polarity_debug(img, h, w, bg_is_dark):
    """Draw polarity sample zones on a copy of the original image."""
    out   = img.copy()
    patch = max(5, int(min(h, w) * POLARITY_SAMPLE_FRACTION))

    # Corner rectangles
    cv2.rectangle(out, (0, 0),                  (patch, patch),       COLOUR_CORNER, 2)
    cv2.rectangle(out, (w - patch, 0),           (w, patch),          COLOUR_CORNER, 2)
    cv2.rectangle(out, (0, h - patch),           (patch, h),          COLOUR_CORNER, 2)
    cv2.rectangle(out, (w - patch, h - patch),   (w, h),              COLOUR_CORNER, 2)

    # Border strip inset rectangle
    cv2.rectangle(out, (patch, patch), (w - patch, h - patch),        COLOUR_BORDER, 1)

    label = f"bg={'DARK-inverted' if bg_is_dark else 'LIGHT-normal'}"
    cv2.putText(out, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA)
    return out


def distance_heatmap(dist, letter_mask):
    """
    Convert distance transform to a visible colour heatmap.
    Background pixels are forced black so heatmap only shows inside letters.
    """
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_u8   = np.uint8(dist_norm)
    heatmap   = cv2.applyColorMap(dist_u8, cv2.COLORMAP_JET)
    heatmap[letter_mask == 0] = (0, 0, 0)
    return heatmap


def trace_yellow_lines(binary, dist, h, w):
    """
    Trace inward gradient lines from edge points toward stroke centres.
    Returns BGR canvas with yellow lines, and sampled point count.
    """
    grad_y, grad_x = np.gradient(dist)
    mag    = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    grad_x = grad_x / mag
    grad_y = grad_y / mag

    edges    = cv2.Canny(binary, 50, 150)
    edge_pts = np.argwhere(edges > 0)

    grid = {}
    for (r, c) in edge_pts:
        key = (r // EDGE_SAMPLE_GRID, c // EDGE_SAMPLE_GRID)
        if key not in grid:
            grid[key] = (r, c)
    sampled = list(grid.values())

    canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for (r, c) in sampled:
        pts = [(c, r)]
        cr  = float(r)
        cc  = float(c)

        for _ in range(GRADIENT_MAX_STEPS):
            ri = int(np.clip(cr, 0, h - 1))
            ci = int(np.clip(cc, 0, w - 1))

            nr = float(np.clip(cr + grad_y[ri, ci], 0, h - 1))
            nc = float(np.clip(cc + grad_x[ri, ci], 0, w - 1))

            if binary[int(nr), int(nc)] == 255:
                break

            cr, cc = nr, nc
            pts.append((int(nc), int(nr)))

        if len(pts) > 2:
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], COLOUR_YELLOW, 1)

    return canvas, len(sampled)


def overlay_skeleton(canvas_yellow, letter_mask):
    """Compute medial axis skeleton and overlay in red on the yellow canvas."""
    skeleton = skeletonize(letter_mask.astype(bool))
    skel_img = skeleton.astype(np.uint8) * 255

    if SKELETON_DILATE > 0:
        k        = np.ones((3, 3), np.uint8)
        skel_img = cv2.dilate(skel_img, k, iterations=SKELETON_DILATE)

    canvas_red = canvas_yellow.copy()
    canvas_red[skel_img > 0] = COLOUR_RED
    return canvas_red


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

input_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.endswith("_Edges_Stg5-Denoise.png")
]

if not input_files:
    print(f"No '*_Edges_Stg5-Denoise.png' files found in:\n  {INPUT_FOLDER}")
else:
    print(f"Found {len(input_files)} file(s).\n")

for filename in input_files:
    stem = filename.replace("_Edges_Stg5-Denoise.png", "")

    out = {i: os.path.join(OUTPUT_FOLDER, f"{stem}_MeanAxis_Stg{i}-{n}.png")
           for i, n in {
               1: "Polarity",
               2: "Distance",
               3: "Yellow",
               4: "Red",
           }.items()}

    print(f"Processing: {filename}")

    src = cv2.imread(os.path.join(INPUT_FOLDER, filename))
    if src is None:
        print(f"  [SKIP] Cannot open: {filename}")
        continue

    h, w = src.shape[:2]
    gray  = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print(f"  Loaded: {w}x{h} px")

    # ── Polarity ──────────────────────────────────────────────────────────────
    bg_is_dark = detect_polarity(gray, h, w)
    print(f"  Polarity: {'dark bg — inverting' if bg_is_dark else 'light bg — normal'}")

    stg1 = draw_polarity_debug(src, h, w, bg_is_dark)
    save_png(out[1], stg1)
    print(f"  Stg1 saved -> {out[1]}")

    # Normalise: text = BLACK (0), background = WHITE (255)
    gray_norm = cv2.bitwise_not(gray) if bg_is_dark else gray
    _, binary = cv2.threshold(gray_norm, 127, 255, cv2.THRESH_BINARY)

    # ── Distance transform ────────────────────────────────────────────────────
    letter_mask = (binary == 0).astype(np.uint8)
    dist        = cv2.distanceTransform(
        letter_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    stg2 = distance_heatmap(dist, letter_mask)
    save_png(out[2], stg2)
    print(f"  Stg2 saved -> {out[2]}")

    # ── Yellow lines ──────────────────────────────────────────────────────────
    stg3, n_sampled = trace_yellow_lines(binary, dist, h, w)
    save_png(out[3], stg3)
    print(f"  Stg3 saved -> {out[3]}  ({n_sampled} edge points sampled)")

    # ── Red skeleton ──────────────────────────────────────────────────────────
    stg4 = overlay_skeleton(stg3, letter_mask)
    save_png(out[4], stg4)
    print(f"  Stg4 saved -> {out[4]}")
    print()

print("All done!")