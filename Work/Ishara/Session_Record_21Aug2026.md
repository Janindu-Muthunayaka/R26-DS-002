# Session record — 20–21 August 2026

**Project:** R26-DS-002 · Component 2 · IT22259134
**Scope of this session:** resolve the p90/p75 conflict, and get the backend
reading a real phone capture end to end.

Every number below was measured during the session on the student's own
machine or corpus. Where something is a judgement rather than a measurement it
says so.

---

## 1. What now works that did not before

A real phone capture goes end to end: EXIF corrected on arrival → quality
gated → close-up path → column-aware OCR → sentence-level mT5 correction →
assembled document. **7.4 s** on the RTX 4060 for a 3-frame burst, of which
~1.8 s is correction after batching.

Test suite: **45 passing** (was 11), including three that run against the
nine real captures in `F:\App\backend\inbox`.

---

## 2. D1 — the glyph metric — RESOLVED

**The pass mark is confirmed from data.** `resolution == OK` iff
`glyph_p75 >= 25` holds on **all 168 corpus rows, zero disagreements**, 7 pages
OK. (The corpus is 168 pages — 107 full, 61 half — not the "roughly 190" the
handoff and build record state.)

**The CSV's estimator was NOT recovered.** Closest specification — whole page,
global Otsu, 8-connectivity, `6 <= height <= 200` — gets median bias to 0.0 px
but only ~20% of pages agree within ±0.5 px. Tested and rejected: deskewed
images instead of raw (no difference at all), and a ±1 bounding-box convention
(cannot fix p50 and p75 simultaneously).

**But the same specification reproduces the CSV's *verdict* on 165/168 pages —
98.2%**, with all three disagreements within 3 px of the threshold. So the
value is not portable and the decision is. Write it that way: do not claim the
estimator was reproduced.

**The two gates were never the same requirement.** On the same 168 pages:

| Gate | Passes |
|---|---|
| `p75 >= 25` (CSV, Android app) | **8 of 168 — 4.8%** |
| `p90 >= 22` (old `core/config.py`) | **154 of 168 — 91.7%** |
| Disagree | **146 pages** |

Measured `p90/p75` on this corpus: **median 1.255**, IQR 1.200–1.351. So
`p75 >= 25` is `p90 >= 31.4`, not 22. The scaffold was ~7× more permissive
than the phone.

Cause: `MIN_BASE_GLYPH = 22` came from the capture-resolution sweep's
percentile table, which measures glyphs **after downscaling to the OCR
optimum**. It is a resize target, not a capture minimum — and it was being
applied to whole captured frames.

**Applied**: `CAPTURE_MIN_GLYPH_P75 = 25.0` split from
`OCR_TARGET_GLYPH_P90 = 24.0`; `glyph_p75` / `glyph_p90` are separate
functions with the specification in the docstring; `schemas.py` gained
`glyph_p75` additively so nothing a teammate depends on broke.

**Still open under D1:** §4.2's images are not the corpus pages at 0.40×.
§4.2 reports p50 14 at the optimum; this corpus measures p50 **17** at
*native* scale. So the U-curve's 1.00× baseline is not the native photograph,
and the 0.40× figure must not be described as "downscale the captured photo by
0.40×" until `Pipeline_v11_Optimal_Capture.ipynb` says what it was.

Full per-page table: `Work/Ishara/glyph_metrics_recomputed.csv`.

---

## 3. The conflation was in three places, not one

| File | What it did | Verdict |
|---|---|---|
| `l2_select/select.py` | `glyph_p90` on a whole frame → `capture_verdict` | wrong |
| `l4b_body/body.py` | `glyph_p90` on a region crop → `rescale_to_optimum` | correct |
| `l3_segment/segment.py` | `glyph_p90` on an article crop → `capture_verdict` | wrong |

The third mattered most: `l5_assemble` **drops** articles on that verdict, so
it decided what got read aloud.

`tests/test_metric_hygiene.py` guards it by reading the source. The first
version banned arguments containing "p90" and **did not catch the real bug** —
the original line was `capture_verdict(p)`. It is now a positive requirement:
the argument must be named for the percentile it holds, so `capture_verdict(p)`
fails too. A test asserts the guard fires on the historical line verbatim.

---

## 4. Bugs found and fixed

**`server.py --root` was silently ignored.** `core.config` is imported at
module level, so `PROJECT_ROOT` was frozen before `main()` set the environment
variable. `config.set_root()` now recomputes dependent paths and is called
before `app.pipeline` is imported, which binds `YOLO_WEIGHTS`/`MT5_PLAIN` at
its own import.

**EXIF orientation was left to library defaults.** Corrected note: OpenCV
*does* apply EXIF in `imread` and (on current versions) `imdecode`; **PIL does
not** on `Image.open`, and the notebooks feed PIL images to the detector.
`imread_upright` / `imdecode_upright` now apply it explicitly, so the result is
upright on any OpenCV. The test compares against an `np.rot90` reference built
independently of PIL, so it cannot pass by agreeing with the code under test.

**Correction was truncating ~85% of every article.** `strong_dedup()` joins
the article into one line; `correct()` split on `'\n'`, got one unit, and made
a single 128-token call. Research Summary §12.5 records this being found and
fixed once in v9; it returned because dedup destroys the line structure before
correction sees it. `textutils.sentences()` now splits on sentence terminators.
Measured on a real capture: **raw 2072 chars / 14 sentences → mT5 2067 chars
(100%)**, previously ~300.

`run_pipeline.py` now prints `raw N chars / M sentences -> mT5 K chars (X%)`
and warns below 60%, because this bug has no symptom — the output is good
Sinhala that simply stops.

---

## 5. The phone path — why YOLO could not be used

The article detector was trained on **full and half page** framings. The
capture app's READY band puts the phone much closer: corpus half-pages measure
`glyph_p75` median **22**, the app's captures **33**.

On `burst_20260820_105855_g27_s3615_4.jpg` the detector did not merely miss the
article filling the frame — it returned **one confident box**,
`(0,2692)-(2172,3264)`, covering the **neighbouring** article's headline along
the bottom edge. Because a box *was* returned, `segment.py`'s `if not boxes`
whole-frame fallback never triggered. That strip was then rescaled by 0.19 and
Tesseract returned **0 characters**. The pipeline read the wrong article,
badly, and reported no error.

**This contradicts a claim in the build record.** §10 says the guidance
distance and the detector's best framing coincide — "two independent
constraints converged". Measured, they conflict: framing loose enough for the
detector puts p75 below the 25 pass mark. **Correct this in Chapter 4.** It is
a more interesting finding than the convergence claim was.

**`layers/l3_segment/closeup.py`** is the phone path. Text lines are found by
*shape* (wider than tall, bounded height, not a speck), so no page mask is
needed — a page mask was tried and failed, because a global Otsu calls a thumb
ink and 36% of the frame came back as "ink", welding every contour into one.

Measured across the nine captures: **53–73 text lines per accepted frame**,
median line height **39–54 px**, and the text bbox trimmed **315–617 px** of
thumb from the left on six frames and from the right on three.

Trigger: `CLOSEUP_MIN_P75 = 28.0`, which is the app's own `NEAR_READY`. The
shutter only fires in the READY band, so every phone capture qualifies by
construction and no corpus page does.

**Known limitation, in the code and covered by a test that asserts it is still
true:** a finger that *overlaps* the text merges with it and lines are lost.
The synthetic test uses a realistic 80 px gap matching the real captures rather
than being tuned until it passed.

---

## 6. Tesseract page-segmentation mode was wrong for this path

`TESS_CONFIG` was `--psm 6`, "a single uniform block of text". Correct for
Pipeline v9, which feeds one clean single-column region at a time; wrong for a
multi-column close-up frame.

Measured on a real capture, psm 6 spliced adjacent columns together
mid-sentence: `චාන්දනී ද; ය [2 වි! සානායක ම, ටෙන්ඩරි පුවත්පත්...`. psm 3
preserved the column order.

Now two constants: `TESS_CONFIG` (psm 6, single-column) and
`TESS_CONFIG_PAGE` (psm 3, close-up).

`CLOSEUP_OCR_SCALE = 0.40`, chosen across 1.0 / 0.6 / 0.4 — 0.4 read better on
every word that differed (`විගණනයක්`, `බෙදා දීම`, `සිදුව`, `ජනතාව`).
**This is a judgement on a 220-character sample, not a measured CER** — there
is no ground truth for that page. It agrees with the sweep's 0.40× optimum,
which is corroboration, not proof.

---

## 7. Reproducibility finding — handle carefully

`tools/verify_model.py` re-runs the 217 locked test sentences and compares
output **strings** against `results/per_sentence_results.json`, so it does not
depend on how CER was defined.

| | exact match | CER (plain Levenshtein) | time |
|---|---|---|---|
| stored B3 strings | — | 0.073034 | — |
| regenerated, batch 1 | 199/217 | **0.061455** | 532 s |
| regenerated, batch 8 | 199/217 | **0.061455** | 56 s |

Two conclusions.

**Batching is free.** Bit-identical output, 9.5× faster. `MT5_BATCH = 8` was
raised on that evidence. Correction on a real capture: 16.6 s → ~1.8 s.

**The regenerated outputs score better than the stored ones**, 0.0615 against
0.0730 under the same metric — entirely attributable to transformers 5.1.0,
since batch 1 and batch 8 agree exactly.

> **Do not use 0.0615 as a result.** The canonical 0.0757 / 36.7% was measured
> under a locked protocol with fixed baselines and a paired bootstrap. This is
> a re-run of one system under a different library. Claiming the better number
> would require re-running the entire evaluation that way — B1, the gated
> system, and the significance tests. Until then the canonical number stands
> and this is a reproducibility observation worth one sentence in Chapter 4.

`verify_model.py` reports REGRESSION / VERIFIED / DIFFERENT-AND-BETTER /
DIFFERS-BUT-NOT-WORSE. An earlier version used `abs(delta)` and called an
improvement a failure.

---

## 8. Findings about the captures themselves

**`burst_20260820_105901_g26_s3995_4.jpg`**: sharpness **83.6** against 175 and
487 for its two siblings, measured p75 **11** against 30 and 32. A blurred
frame that survived on-device selection — build record open item 7, reproduced.

Worth noting *which* gate caught it: `MIN_SHARPNESS = 45` would have let it
through; the **p75 gate rejected it**. That is an argument for the glyph gate.

A live example of over-correction for Chapter 4: on a real capture mT5 changed
`සාක්ෂි මත පදනම් වූ` (correct) to `පදනම්කර වූ`. That is the 3.3% rate in the
wild.

---

## 9. Open, with the evidence attached

1. **mT5 emits Latin fragments** into Sinhala text — `හාත්kumba`,
   `පුවත්පත්ikon`, `කොන්ත්‍රාත්prices`. A guard (strip Latin runs absent from
   the input) is easy, but it changes the **corrector**, which is the research
   contribution. Measure on the locked test set before adopting.
2. **The cut-off column at the frame edge produces fragments** which mT5 turns
   into confident nonsense. Edge columns measured 187–242 px against 744–769 px
   for full ones, so they are separable on width. Fix belongs in the crop.
3. **Headlines are dropped** — the line finder rejects them on height. Titles
   are Layer 4A, which is another member's stub.
4. **A passage repeats in the raw OCR**; `strong_dedup`'s window is 8 words and
   the repeats are far apart.
5. **§4.2's 1.00× baseline** — read it out of `Pipeline_v11_Optimal_Capture.ipynb`.
6. **D5, the v1 CER figure** — 0.0274 vs 0.0238. No v1 artefact survives; every
   notebook in the corpus is v2. Both values are typed prose with no surviving
   computation. Recommendation: quote the mechanism (the gate removed ~57% of
   the data; the unbiased baseline is 0.1197) and describe the inflation as
   roughly five-fold, rather than quoting a precise figure as measured.
7. **`MIN_SHARPNESS = 45.0`** is the one constant in `config.py` with no
   provenance. Labelled NOT MEASURED rather than left looking measured.

---

## 10. Next, in order

| # | Step | Who |
|---|---|---|
| 9.4 | `/capture` returns title + body as JSON; add `web/reader.html` | me |
| 9.5 | `ReaderApi.kt` parses JSON instead of expecting audio bytes | I write, you build |
| 9.6 | MainActivity speaks title then body via `GuidanceSpeaker` | I write, you build |
| — | **Install a Sinhala TTS voice on the A70** | **you** |
| — | `adb reverse tcp:8000 tcp:8000` for the viva | you |

**The contract mismatch that would have killed the demo:** `ReaderApi.kt`
declares the response as `ResponseBody` — raw audio. `stub_server.py` matches
it. The real `server.py` returns JSON. The app works against the stub and
breaks against the real backend. And `l6_speech.speak()` returns `None`, so
there is no audio to return — that is Bumal's layer.

The route out is in the build record §14: return **text**, let the phone speak
it with its own TTS. It removes the Bumal dependency from the demo path
entirely, gives better Sinhala than an offline server engine, and drops the
audio download from the latency budget.
