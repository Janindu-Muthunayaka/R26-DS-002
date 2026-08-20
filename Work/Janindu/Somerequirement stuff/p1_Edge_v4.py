"""
Character Edge Extractor v12 - Batch Folder Mode
Base: v7 (Otsu threshold) with improved denoising and debug stages.

Pipeline:
  1. Otsu threshold → text mask
  2. Polarity detection (three-way vote) → normalise to black on white
  3. Connected components → classify by anchor-median size + proximity
  4. Final clean output

Outputs (saved to subfolder 'Edges_Output'):
  Stg1-Lined      : raw Otsu threshold, black on white
  Stg2-Polarity   : original image with sample zones drawn (debug polarity)
  Stg3-Components : every component in a random distinct colour + area label
  Stg4-Classify   : green=anchor, blue=kept via proximity, red=removed
  Stg5-Denoise    : final clean black on white output
"""

import cv2
import numpy as np
import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
INPUT_FOLDER     = r"E:\Sliit\Research\Fonts\TestFontV4"
OUTPUT_SUBFOLDER = "Edges_Output"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ─── POLARITY DETECTION ───────────────────────────────────────────────────────
# Size of corner/border sample as fraction of smaller image dimension.
# Range: 0.03–0.10
POLARITY_SAMPLE_FRACTION = 0.05

# ─── COMPONENT CLASSIFICATION ─────────────────────────────────────────────────
# Hard minimum component area as fraction of total image pixels.
# Anything below this is always noise regardless of other signals.
# Range: 0.000010–0.000100
HARD_MIN_FRACTION = 0.000030

# Number of largest components used to compute the anchor median.
ANCHOR_TOP_N = 10

# Component qualifies as anchor if area >= this fraction of anchor median.
# Higher = only the very largest survive as anchors.
# Range: 0.10–0.40
ANCHOR_FRACTION = 0.20

# Small component is kept if it is within this multiple of the median
# anchor height from the nearest anchor bounding box.
# Range: 0.3–1.5
PROXIMITY_FACTOR = 0.8

# Small component must also be at least this fraction of anchor median area
# to survive via proximity (stops truly tiny specks surviving near anchors).
# Range: 0.005–0.03
PROXIMITY_MIN_FRACTION = 0.010

# ─── VISUALISATION ────────────────────────────────────────────────────────────
COLOUR_ANCHOR    = (0,   180,   0)   # green
COLOUR_PROXIMITY = (200, 130,   0)   # blue
COLOUR_REMOVED   = (0,     0, 200)   # red
COLOUR_CORNER    = (0,   200, 255)   # yellow-orange — corner sample boxes
COLOUR_BORDER    = (255, 130,   0)   # blue — border strip sample line
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
    Three-way vote: corners + border strip + histogram peaks.
    Returns True if background is dark (image needs inversion).
    """
    patch = max(5, int(min(h, w) * POLARITY_SAMPLE_FRACTION))
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

    # Signal 3: histogram — more dark pixels than light = dark background
    hist        = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_smooth = np.convolve(hist, np.ones(15) / 15, mode='same')
    if hist_smooth[:128].sum() > hist_smooth[128:].sum():
        votes_dark += 1

    return votes_dark >= 2


def draw_polarity_debug(img, h, w):
    """
    Draw the polarity sample zones on a copy of the original image.
    Corners shown as rectangles, border strip shown as a line inset.
    """
    out   = img.copy()
    patch = max(5, int(min(h, w) * POLARITY_SAMPLE_FRACTION))

    # Corner rectangles
    cv2.rectangle(out, (0, 0),              (patch, patch),          COLOUR_CORNER, 2)
    cv2.rectangle(out, (w - patch, 0),      (w, patch),              COLOUR_CORNER, 2)
    cv2.rectangle(out, (0, h - patch),      (patch, h),              COLOUR_CORNER, 2)
    cv2.rectangle(out, (w - patch, h - patch), (w, h),               COLOUR_CORNER, 2)

    # Border strip inset rectangle
    cv2.rectangle(out, (patch, patch), (w - patch, h - patch),       COLOUR_BORDER, 1)

    return out


def otsu_threshold(gray):
    """
    Otsu threshold. Returns binary mask where text = WHITE.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure text (minority) is WHITE
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)
    return binary


def random_colour(seed):
    """Generate a visually distinct random colour from a seed integer."""
    rng = np.random.default_rng(seed * 6364136223846793005 % (2**32))
    # Avoid very dark or very light colours
    while True:
        c = tuple(int(x) for x in rng.integers(40, 220, size=3))
        brightness = 0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0]
        if 60 < brightness < 200:
            return c


def classify_components(labels, stats, num_labels, h, w):
    """
    Classify components into anchor, proximity-kept, and removed.
    Returns (anchor_set, proximity_set) — both are sets of label indices.
    """
    total_px = h * w
    hard_min = total_px * HARD_MIN_FRACTION

    # Filter out hard-minimum failures immediately
    valid = [(stats[l, cv2.CC_STAT_AREA], l)
             for l in range(1, num_labels)
             if stats[l, cv2.CC_STAT_AREA] >= hard_min]

    if not valid:
        return set(), set()

    valid.sort(reverse=True)

    # Anchor median from top-N
    top_areas     = [a for a, _ in valid[:ANCHOR_TOP_N]]
    anchor_median = float(np.median(top_areas))
    anchor_min    = anchor_median * ANCHOR_FRACTION
    proximity_min = anchor_median * PROXIMITY_MIN_FRACTION

    anchor_set = {l for a, l in valid if a >= anchor_min}

    # Proximity radius from median anchor height
    anchor_heights = [stats[l, cv2.CC_STAT_HEIGHT] for l in anchor_set]
    median_h       = float(np.median(anchor_heights)) if anchor_heights else 0
    proximity_px   = median_h * PROXIMITY_FACTOR

    # Anchor bounding boxes
    anchor_boxes = [
        (stats[l, cv2.CC_STAT_LEFT],
         stats[l, cv2.CC_STAT_TOP],
         stats[l, cv2.CC_STAT_LEFT] + stats[l, cv2.CC_STAT_WIDTH],
         stats[l, cv2.CC_STAT_TOP]  + stats[l, cv2.CC_STAT_HEIGHT])
        for l in anchor_set
    ]

    def dist_to_anchor(cx, cy):
        best = float('inf')
        for (ax1, ay1, ax2, ay2) in anchor_boxes:
            dx = max(ax1 - cx, 0, cx - ax2)
            dy = max(ay1 - cy, 0, cy - ay2)
            best = min(best, (dx * dx + dy * dy) ** 0.5)
        return best

    proximity_set = set()
    for a, l in valid:
        if l in anchor_set:
            continue
        if a < proximity_min:
            continue
        cx = stats[l, cv2.CC_STAT_LEFT] + stats[l, cv2.CC_STAT_WIDTH]  // 2
        cy = stats[l, cv2.CC_STAT_TOP]  + stats[l, cv2.CC_STAT_HEIGHT] // 2
        if dist_to_anchor(cx, cy) <= proximity_px:
            proximity_set.add(l)

    return anchor_set, proximity_set


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

output_folder = os.path.join(INPUT_FOLDER, OUTPUT_SUBFOLDER)
os.makedirs(output_folder, exist_ok=True)

all_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    and "_Edges_Stg" not in f
]

if not all_files:
    print("No valid images found.")
else:
    print(f"Found {len(all_files)} image(s).\n")

for filename in all_files:
    input_path = os.path.join(INPUT_FOLDER, filename)
    stem       = os.path.splitext(filename)[0]

    out = {i: os.path.join(output_folder, f"{stem}_Edges_Stg{i}-{n}.png")
           for i, n in {1: "Lined", 2: "Polarity",
                        3: "Components", 4: "Classify", 5: "Denoise"}.items()}

    print(f"Processing: {filename}")

    with open(input_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [SKIP] Cannot open: {filename}")
        continue

    h, w = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Loaded: {w}x{h} px  total_px={h*w}")

    # ── Polarity ──────────────────────────────────────────────────────────────
    bg_is_dark = detect_polarity(gray, h, w)
    print(f"  Polarity: {'dark bg — inverting' if bg_is_dark else 'light bg — normal'}")
    gray_norm = cv2.bitwise_not(gray) if bg_is_dark else gray

    # ── Stg1: Otsu threshold ──────────────────────────────────────────────────
    text_mask = otsu_threshold(gray_norm)

    stg1 = np.full((h, w, 3), 255, dtype=np.uint8)
    stg1[text_mask == 255] = (0, 0, 0)
    save_png(out[1], stg1)
    print(f"  Stg1 saved -> {out[1]}")

    # ── Stg2: Polarity debug ──────────────────────────────────────────────────
    stg2 = draw_polarity_debug(img, h, w)
    # Label result in corner
    label = f"bg={'DARK-inverted' if bg_is_dark else 'LIGHT-normal'}"
    cv2.putText(stg2, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA)
    save_png(out[2], stg2)
    print(f"  Stg2 saved -> {out[2]}")

    # ── Connected components ───────────────────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_mask, connectivity=8)
    print(f"  Components found: {num_labels - 1}")

    # ── Stg3: All components, random colours + area labels ────────────────────
    stg3 = np.full((h, w, 3), 245, dtype=np.uint8)  # light grey background
    total_px = h * w
    for l in range(1, num_labels):
        colour = random_colour(l)
        stg3[labels == l] = colour
        area = stats[l, cv2.CC_STAT_AREA]
        pct  = area / total_px * 100
        cx   = stats[l, cv2.CC_STAT_LEFT] + stats[l, cv2.CC_STAT_WIDTH]  // 2
        cy   = stats[l, cv2.CC_STAT_TOP]  + stats[l, cv2.CC_STAT_HEIGHT] // 2
        # Only label components large enough to be readable
        if area >= total_px * 0.0002:
            cv2.putText(stg3, f"{pct:.2f}%", (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    save_png(out[3], stg3)
    print(f"  Stg3 saved -> {out[3]}")

    # ── Classification ─────────────────────────────────────────────────────────
    anchor_set, proximity_set = classify_components(
        labels, stats, num_labels, h, w)

    keep_set = anchor_set | proximity_set
    removed  = (num_labels - 1) - len(keep_set)
    print(f"  Anchors={len(anchor_set)}  "
          f"Proximity-kept={len(proximity_set)}  "
          f"Removed={removed}")

    # Print anchor median info for tuning reference
    valid_areas = sorted([stats[l, cv2.CC_STAT_AREA]
                          for l in range(1, num_labels)], reverse=True)
    if valid_areas:
        top_areas     = valid_areas[:ANCHOR_TOP_N]
        anchor_median = float(np.median(top_areas))
        print(f"  Anchor median area: {anchor_median:.0f}px "
              f"({anchor_median/total_px*100:.3f}% of image)")
        print(f"  Anchor min cutoff:  {anchor_median*ANCHOR_FRACTION:.0f}px "
              f"({anchor_median*ANCHOR_FRACTION/total_px*100:.3f}% of image)")

    # ── Stg4: Classification map ───────────────────────────────────────────────
    stg4 = np.full((h, w, 3), 255, dtype=np.uint8)
    for l in range(1, num_labels):
        mask = labels == l
        if l in anchor_set:
            stg4[mask] = COLOUR_ANCHOR
        elif l in proximity_set:
            stg4[mask] = COLOUR_PROXIMITY
        else:
            stg4[mask] = COLOUR_REMOVED
    save_png(out[4], stg4)
    print(f"  Stg4 saved -> {out[4]}")

    # ── Stg5: Final clean output ───────────────────────────────────────────────
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    for l in keep_set:
        clean_mask[labels == l] = 255

    stg5 = np.full((h, w, 3), 255, dtype=np.uint8)
    stg5[clean_mask == 255] = (0, 0, 0)
    save_png(out[5], stg5)
    print(f"  Stg5 saved -> {out[5]}")
    print()

print("All done!")