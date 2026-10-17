# Recalibrating the capture guidance against the backend metric

**R26-DS-002 · Component 2 / Android · IT22259134 · 26 August 2026**

**Supersedes §5 of `Android_Capture_Guidance_Calibration.md`.** The measurements
in §1–§4 of that document stand; the threshold table in §5 does not.

---

## 1. The defect

`Guidance.kt` declared its thresholds to be *"base glyph pixels — p75 of
connected-component heights, measured in the captured photo"*, i.e. the
research's own metric. They were not. The app's estimate and the server's
`glyph_p75` are different quantities, and the gap was never closed.

`PITCH_PER_GLYPH = 1.80` was fitted at a single point, `p75 = 25`, and
validated over an app range of **26–29 — a three-pixel window**. Outside it,
nothing was known.

Consequence: a READY band of 28–40 in app units landed captures at a true
`glyph_p75` of roughly **32–45**. Two captures measured 33 and 38, and **both
had a column cut off the frame edge** — the whole article did not fit at the
distance the app was sending the user to.

## 2. The measurement

Every burst filename already carried the app's estimate at the instant of
firing (`burst_<stamp>_g<estimate>_s<sharpness>_<n>.jpg`), but `app/server.py`
renamed uploads to `f0/f1/f2.jpg` and destroyed the pairing on arrival. The
server now carries the stamp through as a suffix, so **every capture is a
calibration point**.

Ten bursts, one article, walking from too-far to too-close; app estimates 14 to
47; `glyph_p75` measured on each resulting JPEG by the project's own function.

| frames | fit | r | residual sd |
|---|---|---|---|
| all 30 | `0.725·app + 13.63` | 0.792 | 8.51 |
| `g0` dropped (24) | `1.088·app + 2.04` | 0.978 | 2.66 |
| **`g0` + blur dropped (21)** | **`1.057·app + 3.31`** | **0.983** | **2.32** |

### `g0` is a null marker, not a measurement

`MainActivity.captureBurst()` writes `lastGlyph?.let { Math.round(it) } ?: 0`.
A stamped `0` means the pitch estimate had **failed its regularity test** when
the shutter fired — not that the app measured zero. Six such frames (sharpness
8 and 46, i.e. motion blur, with `p75` scattered from 9.5 to 49) pulled the fit
from `r` 0.978 to 0.792 and the residual from 2.66 px to 8.51 px.

Feeding a null marker to a fitter as if it were data is a scoring bug. It is
the third in this project, and the tool now excludes such frames explicitly and
prints what it dropped.

### The error is an offset, not a scale

Per-burst offsets (measured `p75` − app estimate), from app 14 to 47:

```
5, 3, 3, 4, 8, 7, 5        mean 4.95, sd 2.36
```

**Near-constant across the whole range.** A single multiplicative constant
cannot correct an offset — re-tuning `PITCH_PER_GLYPH` can only tilt the line.
This is why the constant appeared to work at 25 and failed everywhere else.

### Independent corroboration

The eight paired captures in §5 of the superseded document predict a READY band
of **17–21** in app units. These 21 new frames, taken six days later, give
**17.7–21.5**. Two datasets, collected for different purposes, agreeing to
within a pixel.

## 3. What changed

**The correction is applied in `MainActivity`, not folded into the thresholds:**

```kotlin
val raw = pitch * captureRatio / PITCH_PER_GLYPH
glyph   = raw * ESTIMATE_SLOPE + ESTIMATE_INTERCEPT
```

So the number on screen, the number stamped into each filename, and the number
the server reports are now the same quantity. `Guidance.kt`'s stated units
become true rather than aspirational.

**Thresholds, now genuinely in `glyph_p75`, each anchored to a server constant:**

| | old (app units) | new (`p75`) | anchor |
|---|---|---|---|
| `FAR_NEAR` | 15 | **15** | `CAPTURE_REJECT_BELOW_P75` |
| `NEAR_READY` | 28 | **20** | `CAPTURE_WARN_BELOW_P75` = `CLOSEUP_MIN_P75` |
| `READY_CLOSE` | 40 | **26** | four-column ceiling |
| `CLOSE_VCLOSE` | 50 | **35** | — |

Simulating `GuidanceMapper` line for line with the new bounds: walking in from
far, READY is first announced at **22**, held from 19 to 27, and "slightly
back" begins at 28. No zone is skipped, and the 2 px hysteresis is comfortably
absorbed by the 6 px band.

## 4. The loop closed — measured with the corrected build itself

Twenty bursts taken **after** the rebuild, screened to the 17 whose three
frames of one static scene agreed to within 6 px:

| | mean(measured `p75` − app estimate) | 95% CI |
|---|---|---|
| before the correction | **+4.95** | — |
| all 20 post-correction bursts | −1.10 | [−2.18, −0.02] |
| **spread ≤ 6 px (17 bursts)** | **−1.06** | **[−1.74, −0.38]** |
| spread ≤ 2 px (15 bursts) | −0.87 | [−1.53, −0.21] |

**From ~5 px to ~1 px.** The residual is small but real — zero is outside the
interval at every screening cutoff — so `ESTIMATE_INTERCEPT` was moved
**3.31 → 2.25**. On a 6 px band 1 px is not noise: it takes captures from 91%
inside `p75` 20–26 to **96%**.

The app's own captures measured the app's own error. No test rig, no USB, no
special build.

### Two methodological points this run forced

**The burst is the unit, not the frame.** The shutter fires once per burst on a
median over a ring buffer, and the pipeline then votes the three uploaded
frames. Fitting per frame counts one decision three times: per frame sd 2.32 px,
per burst sd 1.85 px on identical data.

**A slope is meaningless once the app guides you into the band.** Post-
correction captures cluster within a few pixels of app estimate by design; the
fitted slope over that window came out 1.282 on data whose true offset was
−1.06 px. The question stops being *"what is the slope"* and becomes *"is the
offset zero"*, which a mean and a confidence interval answer at any cluster
width. The tool now decides which regime it is in from the offset, not the fit.

### Within-burst spread is a quality signal in its own right

Three of the twenty post-rebuild bursts had frames of **one static scene**
disagreeing badly:

| app | sharpness | median `p75` | spread |
|---|---|---|---|
| 22 | 737 | 28 | **14 px** |
| 23 | 2633 | 17 | **13 px** |
| 23 | 2550 | 19 | 8 px |

Sharpness alone does not catch this — the 2633 and 2550 frames are sharp. It is
movement across the ~2.8 s burst, and it is the reason the multi-frame consensus
stage exists. Worth a sentence in Chapter 4: frame selection by sharpness is
necessary but not sufficient, and the voter is what absorbs the remainder.

## 5. Verification

```
python tools\calibrate_guidance.py work --since-hours 1
```

Expect **"zero is inside the interval"** and **"No action."** The check costs
nothing and the data accumulates by itself.

`--since-hours` counts from **now**, not from the newest capture: both the pre-
and post-rebuild sessions happened on the same day, so anchoring to the newest
capture kept all 96 frames and fitted the two builds together — `1.231·app −
5.22`, which describes neither build.

## 6. For the report

Two things worth a paragraph each in Chapter 4 or 5.

**The calibration itself.** An on-device estimator specified in the research's
own units, but never validated in them — and the failure mode was silent: the
app confidently guided the user to a distance at which the article did not fit,
and a blind listener has no way to see that the instruction is wrong. Closing
the loop against the backend metric cost one line in the server and one in the
app, and the correction was then measured, applied, and re-measured with the
system's own output.

**The negative result inside it.** The obvious repair — re-fit
`PITCH_PER_GLYPH` — cannot work, because the residual is an offset and the
constant is a ratio. That is only visible once the relationship is measured
across a range rather than at a point, which is precisely what the original
calibration did not do.

## 7. Still open

- `ESTIMATE_INTERCEPT = 2.25` has not yet been confirmed by a capture. Rebuild,
  shoot 6–8 bursts, and expect "No action".
- **The `maxSpreadFraction` prediction did not bite.** 27 bursts fired normally
  at the new band, sharpness 737 to 6097. `AutoShutter` is left as it is.
- **End-to-end reading accuracy is still n = 1 article, 684 characters.** Every
  CER figure in Chapter 4 rests on it. Four or five more articles at the new
  band is half a day and is the highest-value work remaining.
- Articles wider than about four columns still need more than one capture — see
  `Large_Articles_Design.md`.
