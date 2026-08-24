# How the system works, end to end

**R26-DS-002 · Component 2 · IT22259134 · 21 August 2026**

Written to answer three questions: what happens to the data, where the research
sits inside it, and whether article segmentation is actually being used.

Every number here was measured on your machine. Where something is an estimate
or a judgement it says so.

---

## 1. The data path, in order

### On the phone — before the shutter

```
CameraX ImageAnalysis      1440x1080 YUV, ~30 fps daylight / 20 indoor
        │
        ├─ centreCropUpright(640x480)   rotation 90 + rowStride 1472 handled
        ├─ Otsu threshold               adapts to lighting
        ├─ line runs -> baseline pitch  outlier-trimmed, 3 validity tests
        ├─ glyph = pitch x 2.267 / 1.80     analysis px -> capture px -> p75
        │
        ├─> GuidanceMapper -> TTS       "a little closer" / "hold steady"
        └─> AutoShutter
```

The phone never measures glyph height directly. It measures **line pitch** and
converts. That is the estimator finding: connected-component height drifts
+52.3% under hand-held blur, line pitch drifts under 1%, because blur thickens
ink but does not move a baseline.

**AutoShutter fires** when three conditions hold for 4 consecutive frames —
guidance is READY, Laplacian sharpness ≥ 600, and the glyph estimate spread is
under 6%. Any failure resets to zero, because one lucky frame is not evidence
the camera is steady.

### On the phone — the burst

```
5 frames captured sequentially (~2.8 s)
        │
        ├─ jpegSharpness() on an eighth-scale decode, ~40 ms each
        └─ keep the 3 sharpest, delete the rest
                │
                └─> POST /capture   multipart, field name "frames", ~6.3 MB, ~2 s
```

Five and keep three because sharpness **degrades across a burst** — the user
relaxes once they think it is over. Measured: a burst that fired at sharpness
4002 delivered frames at 1024, 273, 229. Raising the shutter threshold cannot
fix that; it gates the wrong moment.

### On the server

```
POST /capture
  │
  1. imdecode_upright(bytes)         EXIF orientation applied EXPLICITLY
  │                                  (PIL ignores it; OpenCV's behaviour is
  │                                   version-dependent — neither is trusted)
  2. L2  select()                    per frame: sharpness, glyph_p75, verdict
  │                                  keeps only COMPARABLE-quality frames
  3.     closeup.analyse(ref)        p75 >= 28 ? -> close-up path
  │        ├─ text_lines()           lines by SHAPE (a thumb fails the tests)
  │        ├─ text_bbox()            crop away hand, margin, neighbour article
  │        └─ title_lines()          LOCATE the headline (not read — see §3)
  4. L4A title.extract()             STUB (another member's layer) -> unchanged
  5. L4B read_page()                 per frame: crop, scale 0.40, Tesseract psm 3
  │        └─ vote_lines()           medoid consensus across the 3 frames
  │        └─ strong_dedup()
  6. L4B correct()                   sentences() -> mT5, batch 8, beams 4,
  │                                  no_repeat_ngram 6      <-- THE RESEARCH
  7. L5  assemble()                  drop rejects, collect warnings, reindex
  8. L6  speak()                     STUB (Bumal's layer) -> returns None
  9.     flatten -> JSON             {ok, title, body, warnings, ...}
```

### Back on the phone

```
JSONObject(reply)
  ├─ warnings > 0 ?  say "some parts were skipped"
  ├─ title  (empty until Layer 4A lands)
  └─ body -> GuidanceSpeaker.readAloud()
               ├─ chunk into ~300-character utterances
               ├─ QUEUE_FLUSH first, QUEUE_ADD after
               └─ UtteranceProgressListener -> endCycle()
```

Tap once while it is reading to stop. Guidance stays silent throughout because
`busy` is true.

### Measured timings

| Stage | Time |
|---|---|
| burst of 5 | ~2.8 s |
| on-device selection | ~0.2 s |
| upload, 3 frames | ~2.0 s |
| server: select | 0.59 s |
| server: close-up analyse | 0.07 s |
| server: OCR (3 frames) | 4.48 s |
| server: mT5 correction, batch 1 | **16.58 s** |
| server: mT5 correction, batch 8 | **~1.75 s** (estimated from the 9.5× measured on the 217-sentence benchmark — NOT yet measured on this path) |

So shutter-to-speech is roughly **12 seconds**, not the ~5 s measured against
the stub backend. Re-run `run_pipeline.py` to get the real post-batching figure
before quoting it.

This is exactly why the title-first design in the build record matters: a
Sinhala headline takes three or four seconds to speak, which covers most of the
body's processing. That design is still worth implementing — but it needs
Layer 4A.

---

## 2. Where your research actually sits

Only **step 6** is Component 2. Everything else is engineering that exists so
step 6 receives usable text.

| Research finding | Where it is used |
|---|---|
| **mT5 post-OCR correction, CER 0.1197 → 0.0757** | step 6, the corrector itself |
| `no_repeat_ngram_size=6` (CER 0.0847 → 0.0515) | `MT5_NO_REPEAT_NGRAM` |
| Correct **sentence-by-sentence**, not per article | `textutils.sentences()` |
| Capture must reach a **target glyph height**, not a megapixel count | `CAPTURE_MIN_GLYPH_P75 = 25`, and the phone's guidance bands |
| **Downscaling** to the optimum beats native resolution | `CLOSEUP_OCR_SCALE = 0.40` |
| **Never upscale** (2× → CER 0.336, 3× → 0.659) | `OCR_SCALE_MAX = 1.0`, enforced by a test |
| Consensus needs **comparable-quality** views | L2 keeps only frames within 30% of the best |
| Multi-frame **medoid voting** | `vote_lines()` |
| Diacritics die below ~11 px | why the scale is floored, and why the p75 gate exists |

**The system runs B3 — plain mT5.** The gated architecture is not in it. That
is the correct outcome of your own result and you should say so plainly: the
negative finding is that gating loses, so the delivered system is the baseline.

One live example from your own capture, worth using in Chapter 4:

- OCR produced `කක න ~~ ~නගෙනෆ` as a junk prefix — mT5 removed it entirely.
  That is the NOISE_ARTIFACT category, which B3 fixes at 92.6%.
- OCR produced `'බබදා දීම`; mT5 gave `බෙදා දීම`. Correct.
- OCR produced `සාක්ෂි මත පදනම් වූ`, which was already correct; mT5 changed it
  to `පදනම්කර වූ`. That is an over-correction — your measured 3.3% rate,
  caught in the wild.

---

## 3. Is article segmentation working? No — and that matters

**Be careful here, because the honest answer is not the flattering one.**

Your YOLO11m article detector works, and it is well evaluated: mAP50 0.96,
95.6% correct grouping, **zero over-merge**, on full and half page framings.

**But the phone path does not use it at all.**

The detector was trained on full and half pages. Your capture app's READY band
puts the phone much closer — corpus half-pages measure `glyph_p75` median 22,
your captures measure 33. At that distance the frame *is* one article.

On a real capture, YOLO did not simply miss the article filling the frame. It
returned **one confident box** over the *neighbouring* article's headline at the
bottom edge, and the pipeline read that instead — producing zero characters and
no error message. That is why the close-up path exists and why it bypasses
segmentation entirely.

**So who segments the article? The user does, by pointing at it.**

That is a defensible design for a blind reader — they choose what to read by
aiming, exactly as a sighted reader chooses by looking. But it must be *stated*,
not glossed over:

> The article detector is evaluated on full and half page framings and is
> reported as Component 1 engineering support. The deployed reading path
> operates at close range, where the frame contains a single article, and
> therefore performs no article segmentation; article selection is performed by
> the user through aiming.

**And this contradicts a claim currently in your build record.** §10 says the
guidance distance and the detector's best framing "coincide", from two
independent constraints. They do not. Framing loose enough for the detector puts
p75 below your own 25 px pass mark. **Correct that in Chapter 4** — the conflict
is a more interesting finding than the coincidence was, because it is a real
deployment trade-off nobody would predict from either result alone.

---

## 4. What to improve — ranked, with a gate

**The gate: if Chapters 3, 4 and 5 are not drafted, do none of this.** October
is close, the research is complete, and every item below is optional. A thesis
with an unfinished chapter fails; a thesis without one more experiment does not.

### Worth doing, in this order

**1 · Ground-truth 15–20 phone captures and measure end-to-end CER.**
*Cost: a day of transcription. Value: high.*
Right now `CLOSEUP_OCR_SCALE = 0.40` is a judgement I made by eye on a
220-character sample, and the whole deployed path has never been measured. This
one experiment gives you three things: a real end-to-end number on the system
you actually demonstrate, independent confirmation of the 0.40× optimum on a
different device and different pages, and it removes the only "by eye" constant
in the system. It is the single highest-value remaining measurement.

**2 · The transformers version finding.**
*Cost: 2 hours — the harness already exists. Value: high, and it is a
methodological contribution.*
The same checkpoint, same inputs, scores CER 0.0730 under the stored run and
**0.0615** under transformers 5.1.0 — a 16% relative difference from a library
version alone, with batch size ruled out as the cause. That belongs alongside
your alignment-filtering finding as contribution #5 or #6:

> Post-OCR correction results are sensitive to the inference library version.
> The same fine-tuned mT5 checkpoint, evaluated on an identical locked test set
> with identical generation settings, differed by 16% relative CER between
> transformers 4.x and 5.1.0. Post-OCR benchmarks should pin and report the
> inference stack, not only the model.

That is the same *kind* of finding as contribution #4 — a way benchmarks in this
field can mislead — and it is cheap because `verify_model.py` already produces
it. **Do not replace your headline number with 0.0615.** Report the canonical
0.0757 under the locked protocol and this as a separate observation.

**3 · Measure psm 3 vs psm 6 on the corpus.**
*Cost: an hour. Value: medium.*
I found by eye that psm 6 splices adjacent columns together mid-sentence and psm
3 does not. On corpus pages you have ground truth, so this becomes a measured
result rather than an observation — and it is a genuinely useful practical
finding for anyone OCR-ing multi-column Sinhala newsprint.

**4 · Quantify the framing conflict.**
*Cost: a few hours. Value: medium, but it strengthens §3 above.*
For each corpus page you have both `glyph_p75` and the detector's grouping
outcome. Plot one against the other and the trade-off becomes a figure instead
of a paragraph: the range where the detector works well and the range where the
resolution requirement is met barely overlap.

**5 · Count the Latin-fragment leakage.**
*Cost: an hour. Value: low but easy.*
mT5 emits Latin tokens into Sinhala text — `හාත්kumba`, `පුවත්පත්ikon`,
`කොන්ත්‍රාත්prices`. Counting how many of the 217 test outputs contain Latin
characters absent from their input gives you a named over-correction mode that
is not currently in your taxonomy.

### Do not do

- **Retraining or more data.** 38 → 230 pages moved CER by 0.0006. Measured.
- **Surya.** Attempted, failed on dependencies. It is future work, and the
  deployability failure is itself a legitimate finding.
- **Custom hardware.** A phone in a head mount demonstrates everything.
- **Reviving the gated architecture.** It loses in all ten categories and the
  recall-ceiling argument closes the obvious objection. It is your strongest
  contribution *as a negative result*.

---

## 5. Still open in the system

| Item | Owner | Blocking the demo? |
|---|---|---|
| Title OCR — headline is located but not read | Layer 4A, another member | no, but the demo is weaker |
| Sinhala TTS quality (eSpeak is robotic) | Component 4, Bumal | no |
| Cut-off column at the frame edge yields fragments | me | no |
| Latin fragments from mT5 | me — needs measuring first | no |
| A repeated passage `strong_dedup` cannot span | me | no |
| §4.2's 1.00× baseline unidentified | you — read `Pipeline_v11` | affects Chapter 4 wording |
| D5, the v1 CER figure (0.0274 vs 0.0238) | you | affects contribution #4 |
