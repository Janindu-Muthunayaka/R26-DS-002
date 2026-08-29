# Calibrating the on-device glyph-size estimator

**Project:** R26-DS-002 · Component 2 · Android auto-capture module
**Date:** 19 August 2026
**Device under test:** Samsung SM-A705FN (Galaxy A70), Android 11

---

## 1. The problem

The research acceptance band is stated in **base glyph height**: 22–30 px in the
captured photo, where glyph height is the p90 of connected-component heights.
Below ~22 px the dependent vowel signs (*pilla*) fall under ~11 px and stop being
recoverable.

The phone must decide, live, whether the page in front of it will satisfy that
band. It measures on the `ImageAnalysis` stream, not the capture stream, so two
questions had to be answered before guidance thresholds could be written:

1. What is the analysis→capture scale factor on this device?
2. Which measurable quantity on the analysis frame maps reliably onto base
   glyph height?

## 2. Measured stream geometry (SM-A705FN)

| Stream | Resolution | Aspect |
|---|---|---|
| `Preview` | 1440 × 1080 | 4:3 |
| `ImageAnalysis` | 1440 × 1080 | 4:3 |
| `ImageCapture` | 3264 × 2448 | 4:3 |

**Scale factor = 2.267**, identical in width and height. The equality is the
useful part: it confirms the two streams share a field of view, so the
relationship is a pure scale factor with no cropping term. CameraX overrode the
requested 1920 × 1080 (16:9) analysis size with 1440 × 1080, because its default
aspect-ratio strategy filters to 4:3 before applying the size strategy. Had it
honoured the 16:9 request, the width and height ratios would have differed
(2.27 vs 3.02) and a crop correction would have been required.

Other device facts that shaped the implementation:

- `rotationDegrees = 90` with the activity locked to portrait, so text lines run
  along the **columns** of the raw buffer, not its rows.
- `rowStride = 1472` against `width = 1440` — rows carry 32 bytes of padding, so
  the luminance plane must be indexed row by row.
- Analysis runs at 29.8 fps in good light, 20 fps indoors (exposure-limited, not
  compute-limited).

## 3. Three candidate estimators

Measured by reproducing the entire chain on synthetic Sinhala newspaper pages
(Noto Sans/Serif Sinhala, multi-column, baseline-aligned): render at capture
resolution → compute the research metric → downscale by 2.267 → centre-crop
640 × 480 → run the exact algorithm the Kotlin implements.

| Estimator | Definition |
|---|---|
| **Line-run height** | median height of runs of rows whose ink exceeds a threshold |
| **Line pitch** | mean baseline-to-baseline spacing (outlier-trimmed) |
| **p90 connected-component height** | the research metric itself, computed on the analysis crop |

### 3.1 Line-run height saturates in the operating band

Across base glyph heights of 17 → 29 px — the whole operating range — the
measured line-run height stayed pinned at 8.0 analysis px. Only above ~35 px did
it begin to track. The cause is the fixed ink-density threshold: after
downscaling, the anti-aliased ascender and descender rows fall below it, so the
run collapses onto the x-height core and stops responding to scale. An adaptive
threshold (fraction of the busiest row rather than of the crop width) restores
the response but leaves a 4–11% spread and 1 px quantisation.

**Rejected as the primary estimator.**

### 3.2 p90 connected components is exact on clean images and collapses under blur

On pristine renders, computing the research metric directly on the analysis crop
and scaling by 2.267 recovered true glyph height to within a few percent
(pred/true = 0.99–1.09 over G = 22–70 px) — with no calibration constant at all,
and no dependence on the publication's typography. That is the ideal property.

It does not survive contact with a real camera. Under mild blur the estimate
inflates by 16%; under realistic hand-held conditions (σ ≈ 1.0 px blur, sensor
noise, uneven illumination) by **52%**, with individual sizes reaching +147%.
Blur fuses adjacent components — vowel signs merge into their base glyphs and
neighbouring glyphs touch — and every merge increases the component bounding
box. Reducing the crop size makes it worse, not better.

**Rejected: accurate only under conditions the application never operates in.**

### 3.3 Line pitch is blur-invariant

| Condition | line pitch drift | p90 CC drift |
|---|---|---|
| pristine | 0.0% | 0.0% |
| mild blur | −0.04% | +16.3% |
| realistic hand-held | −0.21% | +52.3% |
| poor (blur + noise) | +0.92% | +39.7% |
| bad (heavy blur) | +0.24% | +15.7% |

Pitch measures **where lines start**, and blur does not move a baseline — it only
thickens the ink around it. Across the operating band the ratio pitch/glyph
held to ~2% within a given publication.

**Selected.**

![Estimator stability under degradation](estimator_stability.png)

## 4. The one constant that must be measured

Pitch is precise and blur-proof, but its absolute meaning depends on the
publication's leading, which is a typographic design choice:

| Setting | pitch ÷ glyph height |
|---|---|
| Noto Sans, leading 1.15 | 1.16 |
| Noto Sans, leading 1.30 | 1.33 |
| Noto Sans, leading 1.45 | 1.48 |
| Noto Serif, leading 1.30 | 1.40 |

So a single constant **R = pitch ÷ base glyph height** must be measured once per
publication and thereafter held fixed.

### Measured on real pages

Twelve Dinamina pages from `layout/raw_pages/` (3024 × 4032), with body-text
tiles sampled across each page. The denominator is `glyph_p75` taken from the
project's own `layout/page_diagnostics.csv`, so the measurement does not depend
on reimplementing the connected-component filter.

| page | pitch | `glyph_p75` | R |
|---|---|---|---|
| p01 | 38.00 | 30 | 1.27 |
| p03 | 27.09 | 19 | 1.43 |
| p04 | 25.62 | 20 | 1.28 |
| p05 | 24.05 | 20 | 1.20 |
| p06 | 33.50 | 19 | 1.76 (rejected) |
| p07 | 25.25 | 17 | 1.49 |
| p09 | 33.01 | 18 | 1.83 (rejected) |
| p10 | 24.83 | 20 | 1.24 |
| p11 | 28.33 | 21 | 1.35 |
| p13 | 25.62 | 20 | 1.28 |
| p14 | 27.75 | 18 | 1.54 |
| p15 | 24.40 | 21 | 1.16 |

p06 and p09 were rejected: pitch of 33 against glyphs of 18–19, where every
other page at that glyph size gives 24–28. Their sampling tiles landed on
headline or caption blocks, whose leading is far looser than body text. Excluding
them gives median 1.28, mean 1.32.

This gives **R = 1.30** for the research corpus — within 2% of the value
derived independently from synthetic renders, which cross-validates both.

### R does not transfer to the phone's own captures

Validated against captures from the SM-A705FN itself, by stamping the app's
live estimate into each filename and measuring `glyph_p75` on the resulting
JPEG:

| operating point | pitch (capture px) | measured p75 | implied R |
|---|---|---|---|
| further | 37.7 | 19.5 | 1.93 |
| closer | 52.7 | 31.5 | 1.67 |

Two findings. First, R on this device is far from the 1.30 measured on the
research pages: the phone's 8 MP output is downsampled and sharpened from a
32 MP sensor, components fragment more, and p75 reads lower for the same
physical text. **The constant is a property of the capture device, not only of
the typeface, and must be fitted per device.**

Second, the two operating points disagree, so the relationship is not a pure
ratio. Fitting both gives an affine form:

```
glyph_p75 ≈ 0.80 × pitch_capture − 10.7
```

p75 falls faster than the text shrinks, because at smaller scales glyph
components fragment and merge more readily. A single multiplicative constant
therefore cannot be correct across the whole range — it only has to be correct
near the decision threshold. At p75 = 25 the fitted line gives pitch ≈ 45,
hence:

**Adopted: R = 1.80**, calibrated at the acceptance threshold.

```
glyph_px = pitch_analysis × 2.267 ÷ 1.80
```

### Which percentile the research actually uses

The deployment design specifies p90 of connected-component heights.
`page_diagnostics.csv` records p50 and p75, and its OK/MARGINAL verdict is
driven by **p75 ≥ 25**. p75 is the correct target, and it is also the more
robust choice: an attempt to reproduce the project's p50 externally differed by
40–50% (8 vs 16, 9 vs 14) while p75 differed by under 15%, because the median
is highly sensitive to how many small components — isolated *pilla*,
punctuation, JPEG speckle — the filter admits, and the 75th percentile is not.

## 5. Guidance thresholds

Stated in base glyph pixels (p75) — the research's own units — so the
acceptance criterion in the app is literally the criterion in the thesis.

The ready band starts at **25**, not at the design document's widened lower
bound of 20. 20 sits below the project's own pass mark: `page_diagnostics.csv`
classifies every page with p75 ≤ 24 as MARGINAL. An app that said "hold steady"
at 21 would be authorising exactly the capture quality the corpus already
suffers from.

The ready band opens at **28**, three pixels above the pass mark. That margin
is deliberate and stated rather than hidden inside a conservative constant:
the estimate carries measurement noise, and a band that begins exactly at the
threshold would let half of its captures fall below it.

| State | Glyph p75 (capture px) | Pitch (analysis px), R = 1.80 | Spoken guidance |
|---|---|---|---|
| seek | no reading | — | "Point at the newspaper" |
| far | < 15 | < 11.9 | "Move much closer" |
| near | 15–28 | 11.9–22.2 | "A little closer" |
| **ready** | **28–40** | **22.2–31.8** | **"Hold steady"** |
| close | 40–50 | 31.8–39.7 | "Slightly back" |
| vclose | > 50 | > 39.7 | "Move back" |

### Validation

Captures taken by the auto-shutter after calibration, measured with the
project's own `glyph_p75` metric:

| app estimate | measured p75 | sharpness |
|---|---|---|
| 26 | 31 | 2275 |
| 27 | 31 | 1603 |
| 27 | 32 | 1388 |
| 27 | 32 | 976 |
| 26 | 29 | 702 |
| 29 | 34 | 453 |
| 29 | 33 | 365 |
| 29 | 32 | 308 |

Every frame clears the 25 px pass mark. For comparison, 7 of roughly 190 pages
in the existing ground-truth corpus do.

### The corpus is below its own threshold

Of the ~190 rows in `page_diagnostics.csv`, seven are OK. Full-page framings
sit at p75 ≈ 17–21; half-page framings reach 21–23 and still do not clear 25.
The entire ground-truth corpus was photographed below the resolution its own OCR
engine requires — which is what §12.7 of the research summary attributes the
residual CER to, now visible page by page rather than as an assertion.

This is the strongest available motivation for the auto-capture module: it is
not a convenience feature, it corrects a measured defect in capture practice.

### Coherence with the article detector

At p75 ≈ 25–30 the phone is roughly 1.4× closer than the full-page framings in
the corpus, putting about half a page in frame. Half-page framing is what the
YOLO11m article detector was trained and evaluated on (93.8% grouping accuracy
on half framings, 0 over-merge). The distance the guidance pushes the user to
and the distance the segmentation model handles best coincide — arrived at from
two independent constraints.

## 5a. Frame selection after the shutter

The shutter decides on a preview frame, but analysis is detached while the
burst runs, so nothing observes the ~2 s during which the photos are actually
taken. Measurement showed the frames degrading badly across that window:

| fired at sharpness | frame 1 | frame 2 | frame 3 |
|---|---|---|---|
| 4002 | 1024 | 273 | 229 |
| 893 | 722 | 535 | 203 |
| 858 | 1589 | 722 | 1504 |

Raising the shutter's sharpness threshold cannot fix this — it gates the wrong
moment. The burst therefore captures **five** frames and uploads the **three
sharpest**, scored on device by Laplacian variance of an eighth-scale decode
(~40 ms per frame). This is the smart-frame-selection pattern from the wearable
literature (GLIMPSE, 2026), applied for the same reason.

It reduces rather than eliminates bad frames: one capture survived selection at
sharpness 327 and yielded p75 6. The multi-frame consensus stage is what
outvotes that frame.

## 6. Cost

Measured on device: 7–12 ms per frame for crop + Otsu + projection profile,
against a 33 ms budget at 30 fps. The published design estimated 3 ms; the
difference is the rotation transpose of 307,200 pixels needed to correct
`rotationDegrees = 90`, which the estimate did not include. Reported as measured.

End-to-end timing, measured:

| Stage | Time |
|---|---|
| Burst of 5 frames | ~2.8 s |
| On-device selection | ~0.2 s |
| Upload of 3 frames (6.3 MB) + response | ~2.0 s |
| **Shutter to audio (stub backend)** | **~5 s** |

The 2 s upload is pure network and does not overlap pipeline work, which is the
concrete argument for returning title audio before body audio.

## 7. What this contributes to the report

The negative results are worth as much as the positive one. Computing the
research metric itself on-device is the obvious approach, and it is
quantitatively wrong under hand-held blur — a finding that only appears if the
degradation is simulated rather than assumed. The chosen estimator was selected
because it is invariant to the dominant nuisance parameter, not because it was
convenient.

Framing for Chapter 5: *capture requirements specified as a target glyph height,
enforced at capture time by an on-device estimator selected for invariance to
motion blur, rather than as a sensor-resolution specification.*

## 8. Reproducibility

- `calibrate.py` — renders pages, ports `GlyphMeasurement.kt` line for line,
  sweeps scale
- `calibrate2.py` — tests the p90 connected-component route and adaptive
  thresholding
- `calibrate3.py` — degradation robustness
- `plot_estimators.py` — the figure above

Run `python3 calibrate.py` etc. Requires Pillow with Raqm (for Sinhala shaping),
NumPy, SciPy, and `fonts-noto-core`.
