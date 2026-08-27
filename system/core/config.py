"""
All tunable numbers in ONE place. Layers import from here — never hardcode.

Every constant carries its provenance. If a value has no measurement behind it,
that is stated rather than implied.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv(
    'SINHALA_ROOT', r'E:/RP/corpus/Sinhala_OCR_Correction_v2')).expanduser()


def _paths():
    global YOLO_WEIGHTS, MT5_PLAIN
    YOLO_WEIGHTS = [
        PROJECT_ROOT/'layout'/'runs'/'articles_full'/'weights'/'best.pt',
        PROJECT_ROOT/'layout'/'article_model_v1.pt',
    ]
    MT5_PLAIN = PROJECT_ROOT/'models'/'mt5_plain'


_paths()


def set_root(path) -> Path:
    """Re-point PROJECT_ROOT after import, and recompute dependent paths.

    Setting the SINHALA_ROOT environment variable alone is NOT enough once this
    module has been imported: `app/server.py` imports core.config at module
    level, so PROJECT_ROOT is already frozen by the time main() parses --root.
    Callers must use this function, and must call it BEFORE importing
    app.pipeline, which binds YOLO_WEIGHTS and MT5_PLAIN at its own import time.
    """
    global PROJECT_ROOT
    PROJECT_ROOT = Path(path).expanduser()
    os.environ['SINHALA_ROOT'] = str(PROJECT_ROOT)
    _paths()
    return PROJECT_ROOT


# ==========================================================================
# CAPTURE QUALITY  —  is this photograph good enough to OCR at all?
# ==========================================================================
# Metric: glyph_p75, the 75th percentile connected-component height in px,
# measured on the WHOLE captured frame. See core/imaging.glyph_p75 for the
# exact specification — the number is meaningless without it.
#
# MEASURED, 20 Aug 2026, over all 168 corpus pages
# (system/tools/reproduce_diagnostics.py):
#   * layout/page_diagnostics.csv marks a page OK iff glyph_p75 >= 25.
#     This rule holds on all 168 rows with ZERO disagreements.
#   * The estimator here reproduces that OK/MARGINAL verdict on 165/168
#     pages (98.2%). It does NOT reproduce the CSV's exact values —
#     median bias 0.0 px but only ~20% of pages agree within +-0.5 px.
#     Verdict-equivalent, not value-exact. Do not claim otherwise.
#
# WHY NOT p90 >= 22, which this file used until 20 Aug 2026:
#   That value came from the capture-resolution sweep's percentile table,
#   which measures glyph heights AT THE OCR OPTIMUM, i.e. AFTER downscaling.
#   It is a target for the resize step, not a minimum for a photograph, and
#   it was being applied to whole captured frames in l2_select.
#   Measured on the same 168 pages:
#       p75 >= 25  passes    8 pages   (4.8%)
#       p90 >= 22  passes  154 pages  (91.7%)
#       the two gates disagree on 146 pages
#   Measured p90/p75 ratio on this corpus: median 1.255 (IQR 1.200-1.351),
#   so p75 >= 25 corresponds to p90 >= ~31.4, not 22.
CAPTURE_MIN_GLYPH_P75 = 25.0

# The threshold the CAPTURE GATE actually warns on, and it is NOT 25.
#
# 25 above is a reproduction of the corpus diagnostics verdict on 168 pages
# and it stays exactly as it is - Chapter 4 cites it. But it answers a
# different question from the one the phone path asks. It asks "is this page
# good enough to OCR at all?"; the phone path asks "should the user move?".
#
# Measured 24 Aug 2026: the whole-article framing sits at glyph_p75 22, and
# reading it there is BETTER than the close framing, not worse - best mT5 CER
# 0.0497 against 0.0570. Warning "closer" at 22 would push the user back to a
# framing that clips a column off every capture. Telling a blind listener to
# move in the wrong direction is worse than saying nothing.
#
# So: 20 is where the advice changes, and it matches CLOSEUP_MIN_P75.
CAPTURE_WARN_BELOW_P75 = 20.0

# Below this the page is not marginal, it is unusable — do not spend a
# pipeline run on it. Adopted from the Android app's FAR_NEAR boundary
# (Guidance.kt) so the phone and the backend agree on one set of bands.
# NOT independently measured; it is a design choice, stated as one.
CAPTURE_REJECT_BELOW_P75 = 15.0

# Component filter for glyph_p75. Heights outside this range are not glyphs:
# below hmin they are noise and speckle, above hmax they are rules, borders
# and photo edges. hmin=6 was selected as the value that drives the median
# bias against page_diagnostics.csv to 0.0 px.
GLYPH_H_MIN, GLYPH_H_MAX = 6, 200

# ==========================================================================
# OCR  —  what size should Tesseract actually see?
# ==========================================================================
# NOTE: this is a DIFFERENT measurement from the capture gate above. It is
# applied to a single text-region crop, not to a whole frame, at a different
# pipeline stage. Keeping them separately named is deliberate; they were
# conflated under one constant until 20 Aug 2026.
#
# PENDING — the value below is carried over unchanged and is NOT yet
# reconciled with the capture metric. The capture-resolution sweep reports
# p50 14 / p75 18 / p90 22 "at the optimum", but this corpus measures
# p50 17 / p75 20 / p90 26 at NATIVE scale — larger than that optimum, not
# 2.5x larger as a 0.40x downscale would require. So the sweep's 1.00x
# baseline is not the native photograph, and until
# Pipeline_v11_Optimal_Capture.ipynb says what it was, this number must not
# be described as "downscale the captured photo by 0.40x".
OCR_TARGET_GLYPH_P90 = 24.0

# Scale is CAPPED AT 1.0. Upscaling cannot restore detail that was never
# captured: the sweep measured 2.0x -> CER 0.336 and 3.0x -> 0.659 against
# 0.175 at the optimum.
OCR_SCALE_MIN, OCR_SCALE_MAX = 0.15, 1.0
TESS_LANG   = 'sin'

# Tesseract page-segmentation mode is NOT one setting. It depends on what the
# region contains, and using one value for both is a real error:
#
#   psm 6  "a single uniform block of text" — correct for Pipeline v9, which
#          feeds one clean SINGLE-COLUMN region at a time.
#   psm 3  Tesseract's own column-aware layout analysis — needed when the
#          region contains several columns, as a close-up frame does.
#
# Measured 21 Aug 2026 on burst_20260820_105855_g27_s3615_4.jpg: psm 6 spliced
# column 1 and column 2 together mid-sentence ("චාන්දනී ද; ය [2 වි! සානායක ම,
# ටෙන්ඩරි පුවත්පත්..."). psm 3 preserved the column order.
TESS_CONFIG       = '--oem 1 --psm 6'    # single-column region (v9 path)
TESS_CONFIG_PAGE  = '--oem 1 --psm 3'    # multi-column crop (close-up path)

# ==========================================================================
# CLOSE-UP PATH
# ==========================================================================
# The article detector was trained on FULL and HALF page framings. The capture
# app's READY band puts the phone much closer than that: corpus half-pages
# measure glyph_p75 median 22, the app's own captures measure 33. At that
# distance the frame IS one article, and YOLO does not merely miss it — on
# burst_..._g27_s3615_4.jpg it returned one confident box covering the
# NEIGHBOURING article's headline at the bottom of the frame, and the pipeline
# read that instead.
#
# So the phone path does not use article segmentation at all. The trigger is
# the app's own READY threshold, which is what makes this principled rather
# than a tuned constant: the shutter only fires in the READY band, so every
# phone capture is a close-up by construction, and no corpus page reaches it.
# MEASURED 24 Aug 2026 and LOWERED from 28.0.
# 28 was adopted from the app's NEAR_READY band, a design choice, not a CER
# measurement - and it refused exactly the framing that holds a whole article.
# The distance test showed the article first fits at glyph_p75 22-25, and the
# accuracy test showed reading it there costs nothing: best mT5 CER 0.0497 at
# p75 22 against 0.0570 at p75 38 on the same 20 lines. 20 leaves a little
# margin below the measured 22 without reaching corpus pages (18-24 - the
# overlap is real, which is why the pitch/glyph sanity check in
# l3_segment/layout.py exists as a second gate).
CLOSEUP_MIN_P75 = 20.0

# Downscale applied to a close-up crop before OCR.
# SELECTED BY EYE on a 220-character sample across scales 1.0 / 0.6 / 0.4 —
# 0.4 read best on every word that differed. This is NOT a measured CER: there
# is no ground truth for that page. It agrees with the capture-resolution
# sweep's 0.40x optimum, which is corroboration, not proof.
CLOSEUP_OCR_SCALE = 0.40      # SUPERSEDED - see CLOSEUP_TARGET_GLYPH below

# The close-up path now scales to a TARGET GLYPH HEIGHT, not by a fixed factor.
#
# WHY THE FIXED FACTOR HAD TO GO. 0.40 was chosen on a frame whose glyph_p75
# was 33, giving Tesseract about 13 px. Applied unchanged to a glyph_p75 22
# frame it gives 8.8 px - below the ~11 px at which diacritics disappear - and
# measured mT5 CER 0.2193 against 0.0760 for the same frame scaled to 13.2 px.
# A factor of 2.9, from one constant, on the framing the system now wants.
#
# WHY 15 AND NOT AN ARGMIN. A six-point sweep (11, 13.2, 15, 17, 19, native)
# on two captures did NOT produce a single optimum: the wide capture was best
# at 17 px (0.0497) and the close one at 11 px (0.0365), and the run-to-run
# spread within one frame (0.0365-0.0936) is as large as the difference
# between frames. What the sweep DOES show is a cliff below 11 px and a flat
# region from 11 to 22. 15 is the middle of that flat region, safely clear of
# the cliff. It is a SAFE CHOICE, not a measured minimum - re-measure on more
# than one article before quoting it as one.
CLOSEUP_TARGET_GLYPH = 15.0

# Hard floor. Below this the diacritics go and no amount of correction gets
# them back; measured 0.2193 CER at 8.8 px. Nothing may scale under it.
CLOSEUP_MIN_GLYPH_PX = 11.0

# ==========================================================================
# CORRECTION
# ==========================================================================
# no_repeat_ngram_size=6 measured 0.0847 -> 0.0515 CER on identical inputs
# by suppressing seq2seq generation runaway on long lines.
MT5_NUM_BEAMS = 4
MT5_NO_REPEAT_NGRAM = 6
MT5_MAX_LENGTH = 128

# Sentences corrected per generate() call.
#
# MEASURED 21 Aug 2026 on the full 217-sentence locked test set, RTX 4060,
# transformers 5.1.0, via tools/verify_model.py:
#
#              exact match      CER        time
#   batch 1     199/217      0.061455     532 s
#   batch 8     199/217      0.061455      56 s
#
# Bit-identical output, 9.5x faster. Padded beam search was the risk — it can
# change results — and it measurably does not here. Raised on that evidence,
# not on the grounds that it is faster.
#
# Re-measure if the model, transformers version or GPU changes:
#     python tools/verify_model.py --root <root> --batch 8
MT5_BATCH = 8

# ==========================================================================
# FRAME SELECTION (Layer 2)
# ==========================================================================
BURST_KEEP      = 3
# NOT MEASURED on this decode path. The Android app measures ~1545 steady and
# ~3 moving and uses 600, but that is a different Laplacian implementation on
# a YUV analysis crop; absolute Laplacian variance is not comparable across
# implementations. Treat 45.0 as a placeholder until measured on server-side
# decoded JPEGs. The relative gate (SHARP_MIN_RATIO) is the one doing work.
MIN_SHARPNESS   = 45.0
SHARP_MIN_RATIO = 0.30     # drop frames far below the best

# ==========================================================================
# DETECTION (Layer 3)
# ==========================================================================
YOLO_CONF  = 0.40
YOLO_IMGSZ = 1024
MAX_ARTICLES = 8

# ==========================================================================
# SERVER
# ==========================================================================
HOST, PORT = '0.0.0.0', 8000
WORK_DIR = Path(os.getenv('SINHALA_WORK', './work')).resolve()
