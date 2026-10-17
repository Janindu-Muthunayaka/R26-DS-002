# Corrections register — claims that are superseded or false

**R26-DS-002 · Component 2 · IT22259134 · 26 August 2026**

Every entry is a statement currently sitting in a project document that later
measurement has contradicted or replaced. **Check this list before copying any
number or sentence into Chapters 3, 4 or 5.**

Six of the nine are *more interesting* as corrections than they were as
original claims. Say them that way.

---

## 1. "Two independent constraints converged" — FALSE

**Where:** `Android_App_Build_Record.md` §10, *Coherence with the article
detector*.

> The distance the guidance pushes the user to and the distance the
> segmentation model handles best coincide — arrived at from two independent
> constraints, not designed in.

**What is actually true.** They conflict. On a real capture
(`burst_20260820_105855_g27_s3615_4.jpg`) YOLO did not miss the article filling
the frame — it returned **one confident box** over the *neighbouring* article's
headline. Because a box *was* returned, the whole-frame fallback in
`segment.py` never fired; the strip was rescaled by 0.19 and Tesseract returned
**zero characters**. The pipeline read the wrong article, badly, and reported
no error.

Framing loose enough for the detector puts `glyph_p75` below the project's own
25 px pass mark. The two ranges do not overlap.

**Replacement claim.**

> The article detector is evaluated on full and half page framings and is
> reported as Component 1 engineering support. The deployed reading path
> operates at close range, where the frame contains a single article, and
> therefore performs no article segmentation; article selection is performed by
> the user through aiming.

**Why the correction is better.** A real deployment trade-off that neither
result predicts alone is a finding. A coincidence is not.

---

## 2. Guidance thresholds READY 28–40 — SUPERSEDED

**Where:** `Android_Capture_Guidance_Calibration.md` §5, threshold table.

**Replaced by:** `Guidance_Recalibration_26Aug2026.md`.

The app's estimate and the server's `glyph_p75` were never the same quantity —
measured on 21 paired frames, the app **under-read by ~5 px**, so a READY band
of 28–40 landed captures at a true `p75` of 32–45. Band is now **20–26** in
true `glyph_p75`, with the affine correction applied in `MainActivity`.

§1–§4 of that document stand. Only the §5 table is wrong.

---

## 3. "Aim the guidance at glyph_p75 ≈ 28–30" — SUPERSEDED

**Where:** `Article_Boundaries_Measured.md` §4.

**What is actually true.** The whole article first fits at **`p75` 22–25**, and
reading it there is **better**, not worse: measured mT5 CER **0.0497** whole at
`p75` 22 against **0.0570** close and clipped at `p75` 38. At 28–30 a
four-column article still loses a column.

The direction of that recommendation was right; the number was two bands too
high, because it was derived from column arithmetic on one page rather than
from CER.

---

## 4. "No new constant was needed" — FALSE

**Where:** `Article_Boundaries_Measured.md` §5.

> `glyph_p75` separates the two cases cleanly on this sample — corpus 18–24,
> phone captures 30–36 — against the existing `CLOSEUP_MIN_P75 = 28`. No new
> constant was needed.

**What is actually true.** `CLOSEUP_MIN_P75` had to come down to **20** so that
whole-article framings are accepted — and lowering it let corpus full pages
into the close-up path, which a test caught. A **second gate was required**:
`COL_MAX_GLYPHS = 90`, a text band wider than 90 glyph-heights means no gutter
was found. Measured: phone close-ups 27–59, corpus pages 131 and 272.

---

## 5. "Is PITCH_PER_GLYPH = 1.80 too high?" — NOW ANSWERED

**Where:** `Article_Boundaries_Measured.md` §6, left open.

**Answer: yes, but not in the way the question assumed.** The error is an
**offset of about +5 px, near-constant from app 14 to 47** — not a wrong ratio.
Per-burst offsets measured: 5, 3, 3, 4, 8, 7, 5.

**A ratio cannot correct an offset.** Re-fitting `PITCH_PER_GLYPH` can only
tilt the line; it cannot shift it. That is why the constant appeared correct at
`p75` 25, where it was fitted, and was wrong everywhere else. Fixed with an
affine correction (`ESTIMATE_SLOPE`, `ESTIMATE_INTERCEPT`), verified by
re-measurement: mean offset now −1.06 px, then 0 after a second pass.

---

## 6. `CLOSEUP_OCR_SCALE = 0.40` — SUPERSEDED, and it was expensive

**Where:** `System_Explained.md` §4 calls it *"a judgement I made by eye on a
220-character sample"*.

**What is actually true.** A fixed factor makes the glyph height Tesseract sees
depend on how far away the user happened to stand. At `p75` 22 it gave
Tesseract **8.8 px** and measured mT5 CER **0.2193**, against **0.0760** for
the same frame at 13.2 px — **2.9×**, from one constant.

Replaced by `CLOSEUP_TARGET_GLYPH = 15.0` with a hard floor at 11 px. The floor
is the measured result; **the target is a safe choice, not an optimum** — a
sweep found 17 px best for one capture and 11 px for another, with the spread
*within* a single frame as large as the difference between frames. Report the
floor, not an argmin.

---

## 7. Multi-frame consensus "7.9% better" — TRUE, BUT

**Where:** consensus experiments, quoted as a benefit.

The number stands. What was not known is that `vote_lines()` voted **by line
index**, so from the first divergence onward it compared unrelated lines.
Measured on three frames of one static scene (105 / 106 / 101 lines):
**15 of 100 output lines were near-duplicates and 271 characters were lost.**

**The measured benefit was obtained despite the implementation, not because of
it.** After content-aligned medoid voting: 0 repeats, 3765 characters against
3494. Say this — it is a stronger paragraph than the bare 7.9%.

---

## 8. "The app's captures measure glyph_p75 33" — HISTORICAL

**Where:** `System_Explained.md` §3, `Session_Record_21Aug2026.md` §5.

True of the old build. Post-recalibration captures land at **20–26**. Wherever
this appears as a present-tense fact about the system, it now describes the
superseded configuration.

---

## 9. Two scoring bugs, and a third

**Where:** referenced in the working rules as "two scoring bugs".

There are now **three**. The third: `calibrate_guidance.py` treated the
filename stamp `g0` as a measurement of zero. It is a **null marker** —
`lastGlyph?.let { Math.round(it) } ?: 0` — meaning the estimate had failed its
regularity test when the shutter fired. Six such frames pulled a fit from
`r` 0.978 to 0.792 and its residual from 2.66 px to 8.51 px on the same data.

Worth one sentence in the methodology: **null markers and measurements must not
share a representation**, and every screening step must print what it dropped.

---

## What is NOT wrong

To be clear about the scope of this list, these stand as reported:

- **The correction result.** CER 0.1197 → 0.0757 (36.7%), WER 0.3358 → 0.1640
  (51.2%), n = 217, page-disjoint test set. This is the research contribution
  and it is properly powered.
- **The negative result.** The SinBERT-gated span corrector underperforming
  plain full-sequence mT5, reported carefully.
- **The corpus verdict reproduction.** 165/168 pages, 98.2%.
- **The library-version finding.** Same checkpoint, same inputs, CER 0.0730 vs
  0.0615 under a different transformers version.
- **The deskew reproducibility finding.** `cv2.minAreaRect`'s angle convention
  is version-dependent; replaced with a projection-profile search and verified
  identical across OpenCV 4.9/4.13.

---

## Known limitation to state, not to fix

**End-to-end reading accuracy is measured on one article, 684 characters.**
Every system-level CER figure rests on it.

This is a limitation of the *integration* evaluation, not of the research
result — Component 2 is evaluated on 217 sentences. State it in one sentence
under Limitations and move on:

> System-level accuracy figures are reported for a single article and are
> indicative rather than representative; the correction component itself is
> evaluated on a page-disjoint set of 217 sentences.
