# Project handoff — 26 August 2026

**R26-DS-002 · Interactive Sinhala newspaper reader for blind users**
**Madhusanka H.P.I · IT22259134 · SLIIT · deadline October 2026**

This file is written so a new session can continue without the earlier
conversation. Read it top to bottom before doing anything.

---

## 1. Scope — what is mine and what is not

> **What I own:** Component 2 (Sinhala post-OCR error correction — the
> research), plus the Android capture app and the integration scaffold.
> Components 3 and 4 (RAG, TTS) are Bumal's. Layer 4A (title extraction) is
> another member's. **Do not work on those** — they are stubs I integrate with,
> not code I write.

## 2. Working rules — follow these

> - **Never invent numbers.** Every figure in the handoff is measured. If you
>   need one that isn't there, say so rather than estimating.
> - **Test before you hand me code.** Two scoring bugs have already produced
>   plausible-looking wrong results in this project. If a harness cannot
>   reproduce a known number, it cannot be trusted on an unknown one.
> - **Tell me when I'm wrong.** I've asked for things that were measurably bad
>   ideas (upscaling images, more training data, full-page capture) and being
>   told plainly saved time.
> - **One step at a time.** Give me something I can run and verify before
>   moving on.
> - **Keep the honest framing.** My strongest contribution is a negative result
>   reported carefully. Don't help me overclaim.

**There are now THREE scoring bugs, not two.** The third is in §7.

## 3. Environment

| | |
|---|---|
| Repo | `E:\RP\R26-DS-002` (shared team repo, origin `github.com/Janindu-Muthunayaka/R26-DS-002.git`) |
| Backend | `E:\RP\R26-DS-002\system` |
| Models | `E:\RP\corpus\Sinhala_OCR_Correction_v2` |
| Android app | `F:\App\SinhalaReader` |
| My documents | `E:\RP\R26-DS-002\Work\Ishara\` |
| OS / GPU | Windows, RTX 4060 (CUDA 12.1) |
| Python | 3.12.5, venv with `--system-site-packages` |
| Libraries | cv2 **4.9.0**, numpy **1.26.4**, transformers 5.1.0, Tesseract 5.5.0 with `sin` |
| Phone | Samsung SM-A705FN (Galaxy A70), Android 11 |

**Library versions matter here** — see §7, the deskew finding. cv2 4.13 /
numpy 2.4 behave differently on `minAreaRect`.

Start the server with `python -m app.server`, **not** `python app\server.py`
(both work now, but the module form is the documented one):

```
cd E:\RP\R26-DS-002\system
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

---

## 4. How the system actually works

```
phone (F:\App)                      backend (system\)
──────────────                      ─────────────────
live pitch estimate  ──┐
guidance: "hold steady"│
auto-shutter           │
burst of 5, keep 3     ├── POST /capture ──▶ L2  frame selection
sharpest               │                     L3  layout analysis (close-up path)
                       │                     L4B OCR + mT5 correction
speak the returned  ◀──┘                     L6  TTS (stub, returns null)
text with on-device TTS
```

**Article segmentation is NOT used in the deployed phone path.** YOLO is
bypassed. The user segments by aiming the camera. This is a defensible design
for a blind reader — they choose what to read by aiming, as a sighted reader
chooses by looking — but it must be **stated**, not glossed. See
`Corrections_Register.md` entry 1.

**The close-up path** (`layers/l3_segment/layout.py`) works by projection:
deskew → glyph mask → vertical projection for column gutters → row profile →
autocorrelation for line pitch → blocks from wide white gaps. It **refuses**
full newspaper pages, deliberately; those are what YOLO is for.

---

## 5. Canonical numbers — do not re-derive, do not contradict

### The research result (Component 2) — properly powered

| | before | after | change |
|---|---|---|---|
| CER | 0.1197 | **0.0757** | −36.7% |
| WER | 0.3358 | **0.1640** | −51.2% |

n = **217** sentences, locked **page-disjoint** test set. Model: **mT5-small
(B3, plain full-sequence)**.

**The negative result:** a SinBERT-gated span corrector was built and
**underperformed** plain full-sequence mT5. This is the strongest contribution
and must be reported carefully, not buried.

### Capture and reading (integration) — n = 1 article, 684 characters

| what | value |
|---|---|
| whole article, `p75` 22 | mT5 CER **0.0497**, WER 0.1429 |
| close but clipped, `p75` 38 | mT5 CER **0.0570** |
| fixed `CLOSEUP_OCR_SCALE = 0.40` at `p75` 22 (8.8 px) | mT5 CER **0.2193** |
| same frame at 13.2 px | mT5 CER **0.0760** — a **2.9×** difference from one constant |
| text read, whole vs clipped | **3499** chars vs 1599 — **2.2×** |
| corpus verdict reproduction | **165/168 = 98.2%** |
| library-version finding | same checkpoint, CER 0.0730 vs **0.0615** |

**Reading the whole article is not a compromise — it is better.**

### Guidance calibration (26 Aug)

| | value |
|---|---|
| app estimate vs server `glyph_p75`, before | `1.057·app + 3.31`, r 0.983 — **under-read by ~5 px** |
| after the affine correction | mean offset **−1.06 px**, then **0** after a second pass |
| per-burst residual sd | **1.85 px** (per frame 2.32 — the burst is the unit) |
| captures landing in `p75` 20–26 | **96%** |
| four-column ceiling | 2448 px frame ÷ ~560 px per column at `p75` 22 ≈ **4.4 columns** |

### The consensus voter bug

Three frames of one static scene produced 105 / 106 / 101 lines. Index-based
voting gave **15 of 100 output lines near-duplicate** and **lost 271
characters**. Content-aligned medoid voting: **0 repeats, 3765 chars vs 3494**.

---

## 6. Current state — what works

- **115 tests pass.** `cd E:\RP\R26-DS-002\system && python -m pytest tests -q`
- A 2–4 column article is read **whole**, at the right distance, with the
  guidance now aimed there.
- A clipped article is **detected** and the user is told which edge —
  *"part of this article is off the right of the frame — move a little to the
  right"* — instead of a fragment being read in silence.
- OCR receives a constant ~15 px glyph at any distance, with a hard floor at
  11 px.
- Full newspaper pages cannot enter the close-up path (two gates).
- Every capture is automatically a calibration point (§8).

**What does not work:** articles wider than about **four columns** cannot be
captured in one frame at any distance. Detection is solved; capture is not.
The two-capture continuation is designed and two thirds built — see
`Large_Articles_Design.md`.

---

## 7. What changed today (26 August)

### 7.1 A regression I caused, and fixed

`capture_verdict()` was switched from `CAPTURE_MIN_GLYPH_P75` (25) to
`CAPTURE_WARN_BELOW_P75` (20) — after I had explicitly said the 25 would stay.
That silently changed a number Chapter 4 cites: corpus agreement collapsed from
**98.2% to 29.2%**.

**Fix:** two functions, two questions.

```python
capture_verdict(p75, warn_below=CAPTURE_MIN_GLYPH_P75)   # is this page OCR-able? -> 25
guidance_verdict(p75)                                     # should the USER move? -> 20
```

`capture_verdict` is now byte-identical in behaviour to the pre-24-Aug original
across 804 inputs including message strings. `layers/l2_select/select.py` (the
phone path) calls `guidance_verdict`; `layers/l3_segment/segment.py` (the page
gate) still calls `capture_verdict`.

### 7.2 The guidance was aimed at the wrong distance — now fixed

`Guidance.kt` claimed its thresholds were in `glyph_p75`. **They were not.**
`PITCH_PER_GLYPH = 1.80` was fitted at a single point (`p75` 25) and validated
over an app range of 26–29 — a three-pixel window. Outside it, the app
under-read by ~5 px, so a READY band of 28–40 landed captures at a true `p75`
of **32–45**, which is why every capture lost a column.

**The error is an OFFSET, not a scale** — per-burst offsets 5, 3, 3, 4, 8, 7, 5
across app 14 to 47. A ratio cannot correct an offset; re-fitting
`PITCH_PER_GLYPH` can only tilt the line.

**Fixed in `MainActivity.kt`:**

```kotlin
val raw = it * captureRatio / PITCH_PER_GLYPH
raw * ESTIMATE_SLOPE + ESTIMATE_INTERCEPT      // 1.057f, 2.25f
```

**`Guidance.kt`, now genuinely in `glyph_p75`:**

| | old | new | anchored to |
|---|---|---|---|
| `FAR_NEAR` | 15 | **15** | `CAPTURE_REJECT_BELOW_P75` |
| `NEAR_READY` | 28 | **20** | `CAPTURE_WARN_BELOW_P75` = `CLOSEUP_MIN_P75` |
| `READY_CLOSE` | 40 | **26** | four-column ceiling |
| `CLOSE_VCLOSE` | 50 | **35** | — |

Backups: `Guidance.kt.bak_26aug`, `MainActivity.kt.bak_26aug`.

`GuidanceMapper` was ported to Python and simulated before the edit: walking in
from far, READY is announced at **22**, held 19–27, "slightly back" at 28. No
zone skipped; the 2 px hysteresis fits inside the 6 px band.

### 7.3 The server now preserves the phone's stamp

The app names every burst frame `burst_<stamp>_g<estimate>_s<sharpness>_<n>.jpg`.
The server was renaming uploads to `f0/f1/f2.jpg` and **destroying that pairing
on arrival**. One line in `app/server.py`:

```python
p = sess / f'f{i}{_stamp_of(f.filename)}.jpg'      # -> f0_g23_s2969.jpg
```

Frame order is unchanged; unstamped uploads (debug page, `tests/test_api.py`)
are named exactly as before. **Every capture is now a calibration point, for
free.**

### 7.4 The third scoring bug

`calibrate_guidance.py` treated the stamp `g0` as a measurement of zero. It is
a **null marker** — `lastGlyph?.let { Math.round(it) } ?: 0` — meaning the
estimate had failed its regularity test when the shutter fired.

Six such frames pulled a fit from **r 0.978 to 0.792** and its residual from
**2.66 px to 8.51 px** on the same 30 frames.

Two further methodology points that came out of the same work:

- **The burst is the unit, not the frame.** The shutter fires once per burst on
  a median over a ring buffer, and the pipeline then votes three frames.
  Per-frame sd 2.32 px, per-burst sd 1.85 px on identical data.
- **A slope is meaningless once the app guides you into the band.** Captures
  cluster within a few px by design; a fit over that window gave slope 1.282 on
  data whose true offset was −1.06 px. The question becomes *"is the offset
  zero"*, answered by a mean and a CI at any cluster width.

### 7.5 Also confirmed today

- **`AutoShutter` needs no change.** Prediction on the record before the test:
  `maxSpreadFraction = 0.06` is a *fraction*, so moving the band down tightens
  the drift tolerance. It **did not bite** — 27 bursts fired normally,
  sharpness 737 to 6097.
- **Within-burst spread is its own quality signal.** Three of twenty bursts had
  frames of one static scene disagreeing by 8–14 px, at sharpness up to 2633.
  Sharpness alone does not catch it; the multi-frame voter is what absorbs it.

---

## 8. Files that matter

### Changed today

| file | what |
|---|---|
| `system/core/imaging.py` | `capture_verdict(p75, warn_below=...)` + new `guidance_verdict(p75)` |
| `system/layers/l2_select/select.py` | now calls `guidance_verdict` |
| `system/app/server.py` | carries the phone's `_g<n>_s<n>` stamp through |
| `system/tests/test_closeup_scale.py` | updated + a new test pinning the two gates apart |
| `F:\App\...\Guidance.kt` | thresholds 15 / 20 / 26 / 35, in real `glyph_p75` |
| `F:\App\...\MainActivity.kt` | `ESTIMATE_SLOPE = 1.057f`, `ESTIMATE_INTERCEPT = 2.25f` |

### Tools

| tool | use |
|---|---|
| `tools/calibrate_guidance.py` | fits app estimate vs server `glyph_p75`, checks the identity |
| `tools/eval_articles.py` | end-to-end CER over several articles; also settles psm 3 vs psm 6 |
| `tools/compare_framing.py` | framing/scale CER comparison on one column |
| `tools/diagnose_article.py` | layout diagnostics on a capture |
| `tools/run_pipeline.py` | run the whole pipeline on a capture folder |

All have `--self-test` where a scorer is involved. **Run it before trusting
output.**

### Documents

| document | what |
|---|---|
| **`Corrections_Register.md`** | **nine claims in the project docs that later measurement contradicted — read before writing any chapter** |
| `Guidance_Recalibration_26Aug2026.md` | supersedes §5 of `Android_Capture_Guidance_Calibration.md` |
| `Large_Articles_Design.md` | the >4-column problem and the two-capture design |
| `Article_Reading_Fixed.md` | the target-glyph sweep and the 11 px cliff |
| `Session_Record_24Aug2026.md` | the voter bug, adaptive scaling, layout cropping |
| `System_Explained.md` | how the whole system fits together |

---

## 9. Commands

```
cd E:\RP\R26-DS-002\system
```

**Tests** — expect **115 passed**:
```
python -m pytest tests -q
```

**Server:**
```
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

**Check the app is still calibrated** — expect *"zero is inside the interval"*
and *"No action."*:
```
python tools\calibrate_guidance.py work --since-hours 1
```

**Run a capture through the pipeline:**
```
python tools\run_pipeline.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --chars 4000 work\<jobid>
```

**Layout diagnostics on a capture:**
```
python tools\diagnose_article.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 work\<jobid>
```

---

## 10. What is left, in priority order

### 1 · Chapters 3, 4 and 5 — THE ONLY REAL RISK

Six weeks to October. The code is finished enough. **A thesis with an
unfinished chapter fails; a thesis without one more experiment does not.**

Chapter 4 has six measured results, every one with numbers behind it:

1. The framing/resolution trade-off — the first end-to-end CER this system has
2. The fixed-scale defect — 0.2193 → 0.0760, **2.9× from one constant**
3. The 11 px diacritic cliff, confirmed independently of the corpus work
4. The OpenCV version reproducibility finding, fixed and verified across two majors
5. The guidance calibration closing on itself — ~5 px error measured, corrected, re-measured to 0
6. The consensus voter bug — 15% of lines duplicated, 271 characters lost

Plus the negative result, which is the strongest contribution.

**Start here. Use `Corrections_Register.md` as the checklist.**

### 2 · Deliberately NOT doing: the 5-article end-to-end evaluation

`tools/eval_articles.py` exists and works if it is wanted. It was **deprioritised
on purpose**: it does not make the system work, it only measures how well it
already does, and the research contribution is already evaluated at n = 217
sentences. State the limitation in one sentence instead:

> System-level accuracy figures are reported for a single article and are
> indicative rather than representative; the correction component itself is
> evaluated on a page-disjoint set of 217 sentences.

### 3 · Optional, after the chapters

- **Two-capture continuation** for 5–8 column articles — one day.
  `layout.overlap()` and `layout.join()` exist and are tested; what is missing
  is session state and reading order. `Large_Articles_Design.md` step C.
- **Per-column OCR with psm 6** — measured once: 693 chars vs psm 3's 561 on
  one crop, and psm 3 returned **zero** on one 277×445 crop. `eval_articles.py`
  scores both paths paired if this is ever worth settling.
- **Count Latin-fragment leakage** from mT5 (`with`, `ikon`, `One`, `ush`,
  `kinni`, `high`) across the 217-sentence test set — an hour, and it gives the
  error taxonomy a named category it currently lacks.
- **Full mosaic capture** — write up as future work, do not build it.

---

## 11. Things that will bite a new session

- **Do not change `CAPTURE_MIN_GLYPH_P75 = 25`, or the default of
  `capture_verdict()`.** It reproduces the 168-page corpus verdict that
  Chapter 4 cites. The phone path has its own function.
- **Do not use `cv2.minAreaRect` for deskew.** Its angle convention is
  version-dependent — measured +0.90/+0.86/+0.75 on one machine where another
  gave 0.00/−1.38/−1.00 on the same frames, doubling the tilt on one of them.
  Use the projection-profile search in `layout.deskew_angle()`.
- **psm 3 on a single-column crop can return zero characters.** Use
  `TESS_CONFIG` (psm 6) for single columns, `TESS_CONFIG_PAGE` (psm 3) only for
  multi-column crops.
- **`glyph_p75` and `glyph_p90` are not interchangeable.** p75 is the capture
  gate on a whole frame; p90 is the OCR resize target on a region crop. They
  were conflated once and the backend accepted 154 of 168 corpus pages while
  the app accepted 8.
- **`--since-hours` in `calibrate_guidance.py` counts from now**, not from the
  newest capture — both sessions on 26 Aug were the same day, and anchoring to
  the newest capture fitted two different builds together as `1.231·app − 5.22`,
  which described neither.
- **117 files may show as modified in git.** It is CRLF churn. Check with
  `git diff --ignore-cr-at-eol --stat` before believing it.
- **Model files are large.** Git LFS is installed; 10 GiB free storage and
  bandwidth on GitHub Free, billed to the repo **owner** (Janindu, not me).

---

## 12. The framing to keep

This is a **research prototype with honest numbers**, and that is what a
final-year project should be. It is not an industrial system, and claiming so
is the one thing that could undermine a set of results that are otherwise
carefully measured.

What it does **not** have, and should be stated as limitations:

- evaluation on more than one article end to end
- offline operation, battery and latency budgets
- error recovery when the network drops
- **any testing with an actual blind user** — a capture flow nobody can see
  cannot be validated by a sighted developer at a desk

Say those plainly and the measured limits become findings. The negative result,
the three scoring bugs caught and reported, the library-version reproducibility
finding, and a calibration loop that measured and corrected its own error are
all stronger material than a claim of completeness would be.
