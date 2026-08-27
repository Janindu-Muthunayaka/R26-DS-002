# How the system works, end to end

**R26-DS-002 · Component 2 · IT22259134 · updated 27 August 2026**
*(first written 21 August; every section below has changed since.)*

Written to answer three questions: what happens to the data, where the research
sits inside it, and whether article segmentation is actually being used.

Every number here was measured on your machine. Where something is an estimate
or a judgement it says so. **Where a number in the 21 August version has been
superseded, the old one is shown struck through rather than deleted** — the
correction is usually more interesting than the original claim.

---

## 0. What changed since 21 August, in one place

| | 21 Aug | now |
|---|---|---|
| close-up gate | `p75 >= 28` | layout tried first, `LAYOUT_MIN_P75 = 12` |
| OCR scale | fixed `0.40` | `CLOSEUP_TARGET_GLYPH = 15`, floor 11 |
| headline | located, **not read** | located, attached, **read** (`sin_raw`, psm 11) |
| article boundary | white gaps only | **headline is a hard boundary** |
| YOLO in the phone path | bypassed, one frame's evidence | **off, measured on 70 captures** |
| garbage output | read aloud | **readability gate** stops it |
| follow-up questions | impossible | `POST /ask`, 9 local commands + RAG |
| Components 3 / 4 | not connected | 3 built and running; 4 still a stub |
| tests | 115 | **245** |

---

## 1. The data path, in order

### On the phone — before the shutter

```
CameraX ImageAnalysis      1440x1080 YUV, ~30 fps daylight / 20 indoor
        │
        ├─ centreCropUpright(640x480)   rotation 90 + rowStride 1472 handled
        ├─ Otsu threshold               adapts to lighting
        ├─ line runs -> baseline pitch  outlier-trimmed, 3 validity tests
        ├─ glyph = pitch x ratio / 1.80
        ├─         x 1.057 + 2.25       <-- ADDED 26 Aug: the affine correction
        │
        ├─> GuidanceMapper -> TTS       "a little closer" / "hold steady"
        └─> AutoShutter
```

The phone never measures glyph height directly. It measures **line pitch** and
converts. That is the estimator finding: connected-component height drifts
+52.3% under hand-held blur, line pitch drifts under 1%, because blur thickens
ink but does not move a baseline.

**The affine correction is new and it mattered.** The app's estimate and the
server's `glyph_p75` were never the same quantity: measured on 21 paired
frames, the app **under-read by about 5 px** across app estimates 14 to 47. The
error is an **offset, not a scale**, so re-fitting `PITCH_PER_GLYPH` could only
tilt the line and never shift it. Fixed with `ESTIMATE_SLOPE = 1.057` and
`ESTIMATE_INTERCEPT = 2.25`; re-measured mean offset −1.06 px, then 0 after a
second pass.

Guidance thresholds are therefore now genuinely in `glyph_p75`:
~~READY 28–40~~ → **FAR/NEAR 15, NEAR/READY 20, READY/CLOSE 26, CLOSE 35**.

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
                └─> POST /capture   multipart, field "frames", ~6.3 MB, ~2 s
```

Five and keep three because sharpness **degrades across a burst** — the user
relaxes once they think it is over. Measured: a burst that fired at sharpness
4002 delivered frames at 1024, 273, 229. Raising the shutter threshold cannot
fix that; it gates the wrong moment.

Every frame is named `burst_<stamp>_g<estimate>_s<sharpness>_<n>.jpg`, and
since 26 Aug the server **keeps that stamp**, so every capture ever taken is a
calibration point for free.

### On the server

```
POST /capture
  │
  1. imdecode_upright(bytes)        EXIF orientation applied EXPLICITLY
  2. L2  select()                   sharpness, glyph_p75, guidance_verdict
  │                                 keeps only COMPARABLE-quality frames
  3. L3  layout.analyse()           <-- TRIED FIRST NOW, not gated on p75
  │       ├─ deskew()               projection-profile search, NOT minAreaRect
  │       ├─ glyph_mask()
  │       ├─ column_bands()         gutters; a page with none is refused here
  │       ├─ clipped_bands()        drop any column the frame edge sliced
  │       ├─ blocks()               white gaps
  │       ├─ split_at_headlines()   <-- NEW: a headline is a HARD boundary
  │       └─ current_block()        the one spanning the frame centre
  │   L3  headline_for_block()      <-- NEW: which headline belongs to it
  4. L4A title.extract()            <-- NOW REAL: sin_raw, psm 11, scaled
  5. L4B read_page()                crop, scale to a TARGET GLYPH, psm 3
  │       ├─ vote_lines()           content-aligned medoid across 3 frames
  │       └─ strong_dedup()
  6. L4B correct()                  sentences() -> mT5, batch 8, beams 4,
  │                                 no_repeat_ngram 6      <-- THE RESEARCH
  7. L4C polish()                   optional LLM repair — OFF by default
  8. L5  assemble()                 drop rejects, collect warnings, reindex
  9.     quality gate               <-- NEW: refuse to read garbage aloud
 10.     SESSIONS[job] = Document   <-- NEW: the system's one memory
 11.     flatten -> JSON            {ok, title, body, warnings, quality, ...}
```

Three of those steps did not exist on 21 August and each fixes something the
system was getting wrong. They are §3, §4 and §5.

### The second door — `POST /ask`

```
POST /ask {job, text}
  │
  L0 voice.interpret()      9 local command words, else -> a real question
  L6 generate.answer()      local intents answered from session;
                            anything else -> Component 3 (RAG)
  -> {speakable, route, intent, sources, warnings}
```

The phone speaks `speakable`. Volume-down starts a question — a physical key,
because a blind user cannot find a target they cannot see.

### Back on the phone

```
JSONObject(reply)
  ├─ warnings > 0 ?  say "some parts were skipped"
  ├─ title  (now populated when a headline could be attached)
  └─ body -> GuidanceSpeaker.readAloud()
               ├─ chunk into ~300-character utterances
               ├─ QUEUE_FLUSH first, QUEUE_ADD after
               └─ UtteranceProgressListener -> endCycle()
```

Tap once while it is reading to stop. Volume-down to ask about it. Guidance
stays silent throughout because `busy` is true.

### Measured timings

| Stage | Time |
|---|---|
| burst of 5 | ~2.8 s |
| on-device selection | ~0.2 s |
| upload, 3 frames | ~2.0 s |
| server: select | 0.59 s |
| server: layout + close-up analyse | 0.07 s |
| server: title OCR (when a headline is attached) | not measured |
| server: OCR (3 frames) | 4.48 s |
| server: mT5 correction, batch 1 | **16.58 s** |
| server: mT5 correction, batch 8 | **~1.75 s** — still ESTIMATED from the 9.5× measured on the 217-sentence benchmark, **not** measured on this path |

Shutter-to-speech is roughly **12 seconds**. That figure has not been
re-measured since batching landed. Run `run_pipeline.py` before quoting it.

The title-first design in the build record now has something to work with: a
Sinhala headline takes three or four seconds to speak, which covers most of the
body's processing. Worth implementing, and no longer blocked on another member.

---

## 2. Where your research actually sits

Only **step 6** is Component 2. Everything else is engineering that exists so
step 6 receives usable text.

| Research finding | Where it is used |
|---|---|
| **mT5 post-OCR correction, CER 0.1197 → 0.0757** | step 6, the corrector itself |
| `no_repeat_ngram_size=6` (CER 0.0847 → 0.0515) | `MT5_NO_REPEAT_NGRAM` |
| Correct **sentence-by-sentence**, not per article | `textutils.sentences()` |
| Capture must reach a **target glyph height**, not a megapixel count | `CAPTURE_MIN_GLYPH_P75 = 25` (page gate) and `CAPTURE_WARN_BELOW_P75 = 20` (phone gate) — **two functions now, two questions** |
| **Downscaling** to the optimum beats native resolution | ~~`CLOSEUP_OCR_SCALE = 0.40`~~ → `CLOSEUP_TARGET_GLYPH = 15.0`, floor 11 |
| **Never upscale** (2× → CER 0.336, 3× → 0.659) | `OCR_SCALE_MAX = 1.0`, enforced by a test |
| Consensus needs **comparable-quality** views | L2 keeps only frames within 30% of the best |
| Multi-frame **medoid voting** | `vote_lines()` — content-aligned since 24 Aug |
| Diacritics die below ~11 px | `CLOSEUP_MIN_GLYPH_PX`, and why the gates exist |

**The system runs B3 — plain mT5.** The gated architecture is not in it. That
is the correct outcome of your own result and you should say so plainly: the
negative finding is that gating loses, so the delivered system is the baseline.

### The fixed-scale defect, and that it happened twice

~~`CLOSEUP_OCR_SCALE = 0.40`~~ was a judgement made by eye on a 220-character
sample. A fixed factor makes the glyph height Tesseract sees depend on how far
away the user happened to stand. At `p75` 22 it gave Tesseract **8.8 px** and
measured mT5 CER **0.2193**, against **0.0760** for the same frame at 13.2 px —
**2.9×, from one constant.**

**The same defect appeared again on 27 Aug, in the new headline path.** Headline
bands are 250–325 px; at native scale Tesseract *collapsed* on one of three
captures (`'දිමුදු ිළ ුකී ි දු ී'`) and recovered at any downscale into 40–90 px
(`'ණෑගල නගර සම ඔණනයක් ලබා දෛ'`). `TITLE_TARGET_BAND_PX = 90`.

Two independent occurrences of one mistake is a methodology point worth a
sentence: **scale to a measured target, never by a fixed factor** — and both
are "a safe choice inside a flat region", not an argmin.

### Live examples from your own captures

- OCR produced `කක න ~~ ~නගෙනෆ` as a junk prefix — mT5 removed it entirely.
  That is the NOISE_ARTIFACT category, which B3 fixes at 92.6%.
- OCR produced `'බබදා දීම`; mT5 gave `බෙදා දීම`. Correct.
- OCR produced `සාක්ෂි මත පදනම් වූ`, already correct; mT5 changed it to
  `පදනම්කර වූ`. An over-correction — your measured 3.3% rate, caught in the wild.

---

## 3. Is article segmentation working? Now yes — and the detector is why not

**The 21 August answer was "no, and the user segments by aiming."** That was
half right. The other half was that the code was not identifying an article at
all, and its own docstring said so:

> `blocks()`: It does NOT distinguish "next article" from "sub-heading" — the
> geometry is the same. **This narrows the crop; it does not identify a story.**

### What was actually happening — 70 real captures, `system/work`

| path | share | what it read |
|---|---|---|
| layout → column crop | 48 (69%) | the article… mostly |
| **not close-up → YOLO** | **20 (29%)** | whatever the detector returned |
| layout refused → text bbox | 2 (3%) | **every text line in the frame** |

And **13 of the 48** on the "good" path had a headline band sitting *inside* the
crop — two articles merged into one read.

Three causes:

1. **`row_profile()` sums ink across all columns.** A white gap in column 1 is
   filled by text in column 2, so the profile can never see a horizontal
   article boundary.
2. **The gate was `glyph_p75 >= 20`.** Real captures have a median of 22 with
   the lower quartile *below* 20, so nearly a third fell through to YOLO.
3. **The fallback was `text_bbox()`** — literally the bounding box of all
   visible text.

### The fixes

**A headline is a hard boundary** (`layout.split_at_headlines`). Above it is the
previous story; the headline and what follows are the next one. A headline off
to one side is ignored — it heads a story in another column.

This works because the headline threshold is now **measured**, on nine captures:

| | of the median line height |
|---|---|
| tallest **body** line | **1.28× – 1.70×** |
| tallest **headline** band | **5.91× – 8.71×** |

Nothing lands between them. ~~The old constant was 1.6~~ — inside the body
range, so the tallest body line of every capture was being called a headline.
`TITLE_MIN_LINE_RATIO = 3.0` sits in the empty gap.

**Layout is tried first; the p75 gate was refusing frames it handles.**
Re-measured with the gate lowered, layout succeeds on **16 of the 20** frames
that were going to YOLO. Safe, and checked before changing: with the p75 gate
**off entirely**, **12 of 12 corpus full pages are still refused** by the gutter
gate. `LAYOUT_MIN_P75 = 12`.

| | before | after |
|---|---|---|
| proper article crop | 48 (69%) | **57 (81%)** |
| falls through to YOLO | 20 (29%) | **11 (16%)** |
| **crops spanning two articles** | **13 of 48** | **0 of 57** |

### The detector — settled, on seventy frames

Your YOLO11m article detector is well evaluated on what it was trained for:
mAP50 0.96, 95.6% correct grouping, zero over-merge, on **full and half page**
framings.

`tools/probe_yolo.py` compared its most confident box against the article the
layout path chose, on all 70 real captures:

| verdict | frames |
|---|---|
| **DISAGREE — a different story** | **35 (69% of the 51 comparable)** |
| partial | 5 (10%) |
| agree | 11 (22%) |
| detector returned nothing | 8 |
| layout refused, nothing to compare | 11 |

`Corrections_Register.md` entry 1 recorded this on **one** frame. It is now
measured on seventy and it holds.

**Which side is wrong?** Disagreement alone does not say — but the two are not
equally evidenced. The layout crop: **0 of 57 span an article boundary**, and
two were confirmed by eye against the photograph. The detector: one confident
box and a 69% disagreement rate, at a range it was never trained for.

### So the phone path no longer runs it

`SEGMENT_MODE = 'off'`. When layout cannot identify an article the system says
*"ලිපිය හඳුනාගත නොහැකි විය. ටිකක් ළං වී නැවත උත්සාහ කරන්න."* and reads
**nothing**.

**A wrong article read confidently to someone who cannot check it is worse than
no article** — the same reasoning that makes a wrong headline worse than none.
And "move closer" is the instruction that actually fixes the frame, so the next
capture is right instead of this one being wrong.

The detector is not deleted. It is Component 1's contribution, it is evaluated
on the framings it was trained for, `SINHALA_SEGMENT_MODE=yolo` restores the old
behaviour, and a test asserts it cannot come back silently.

**The cost, stated: 11 captures in 70 (16%) now get "move closer" instead of a
read.** One in six asks the user to try again. What reduces that is better
guidance, not a better guess.

### For Chapter 4

> An article detector trained on full-page framings disagrees with
> column-projection segmentation on 69% of close-range captures; the deployed
> system therefore declines to segment rather than segment wrongly.

That is the same *shape* as your headline negative result — a component
measured, found unsuitable for a specific condition, and reported. Two negative
results honestly reported is a stronger methodology story than either alone.

**It also settles the build-record contradiction.** §10 claimed the guidance
distance and the detector's best framing "coincide", from two independent
constraints. They do not, and now there is a number: 69%.

---

## 4. The headline is read now

Locating it is `closeup.headline_for_block()`; reading it is Layer 4A.

**Association refuses rather than guesses.** Three tests must all pass — gap,
x-overlap with the article's own columns, and a single contiguous row group.
These pages carry a masthead, a page number and a section strip
(`ප්‍රාදේශීය පුවත්`) above the real headline, all headline-sized. Reading those
aloud as the headline would be worse than silence. **A headline is attached on
8 of 9 captures**; the ninth is not a close-up at all.

**Reading it uses Janindu's `sin_raw`, psm 11**, measured on the located regions:

| | result |
|---|---|
| `sin_raw` psm 11 | `'කුරුණෑගල නගර විගණනයක් ලබා'` — near-correct |
| `sin_raw` psm 6 | `'අරුණමුලු ප්ගර විශිණිනියක් ලබා'` |
| `sin_raw` psm 7 | `'දදු'` — collapses |
| `sin_custom` psm 11 | `'දදකදීද්ථීදල, එද්්ට දඉීීීීර්ද'` — garbage |

`sin_custom` is the **MAT** model, trained on skeletonised glyphs and garbage on
raw pixels — exactly what `l4a_title/README.md` warns about, which is why that
warning is there. The full MAT pipeline is not used.

**Known limits, stated:** coloured (red) headline words are lost to the
grayscale Otsu threshold — `අකුමිකතා` is missing from every reading of that
capture. A clipped headline reads partially; that is a capture problem. The
headline is **not** run through mT5: Component 2 is trained on body sentences,
and headline fragments have not been measured.

`SINHALA_TITLE_MODE = stub` by default until you re-measure on the pinned
environment. `python tools\measure_headline.py --ocr` reproduces all of it.

---

## 5. Two guards that did not exist on 21 August

### The readability gate — `core/quality.py`, always on, no model

A bad capture **does not fail**. Tesseract returns something, mT5 corrects that
something, and the phone reads it aloud in the same confident voice it uses for
real news. A sighted developer sees garbage on a screen; a blind user hears
fluent nonsense and cannot tell it from the article.

Four surface statistics, calibrated on this project's own OCR outputs
(`tools/calibrate_quality.py`). It separates **short** from **shattered**: a
six-word news brief is read with a warning; fragments, Latin where Sinhala
belongs, or undecodable bytes are replaced by a request for another photograph.

**A measurement bug it exposed in itself**, worth a methodology sentence: the
first version scored ground truth identically to the worst OCR, because Sinhala
dependent vowel signs are combining marks and `str.isalnum()` is False for every
one of them — `ක්‍රියාත්මක` measured as 5 characters. *A token-length measure
that ignores combining marks is not measuring Sinhala.*

Honest calibration result: on these files the continuous measures barely
separate anything, and `n_words` does the work — one real capture in
`tools/out/cer` returned **zero characters** (psm 3 on a single-column crop) and
the system would previously have corrected nothing and read nothing.

### Layer 4C — LLM post-editing, built and OFF

Handing corrupt Sinhala to a general-purpose model is the most dangerous thing
in the repository: it returns a fluent sentence with names and numbers that were
never on the page, and it would make Chapter 4's CER measure a different model.

It writes to `body_polished`, **never** to `body`. Four guards — character
similarity ≥ 0.75, length ratio 0.70–1.30, Sinhala ratio must not fall, word
count must not grow > 20% — and a rewrite failing any one is discarded. Every
touched article, **including a rejection**, carries a warning to the phone.
`tests/test_polish.py` asserts no evaluation tool can enable it.

`auto` deliberately **refuses** unreadable text: that is where a repair is most
wanted and invention most likely.

---

## 6. What to improve — ranked, with a gate

**The gate: if Chapters 3, 4 and 5 are not drafted, do none of this.**

**1 · Ground-truth 15–20 phone captures and measure end-to-end CER.**
*A day. Value: still the highest.* The deployed path has never been measured
end to end. It also re-measures `CLOSEUP_TARGET_GLYPH` on a second device and
different pages.

**2 · The transformers version finding.** *2 hours, harness exists.* Same
checkpoint, same inputs: CER 0.0730 vs **0.0615** under transformers 5.1.0 — 16%
relative from a library version, batch size ruled out.

> Post-OCR correction results are sensitive to the inference library version.
> Post-OCR benchmarks should pin and report the inference stack, not only the
> model.

**Do not replace your headline number with 0.0615.** Report 0.0757 under the
locked protocol and this as a separate observation.

**3 · Measure psm 3 vs psm 6 on the corpus.** *An hour.* Still open, still
worth it — on corpus pages you have ground truth.

**4 · ~~Quantify the framing conflict~~ — DONE.** It is the 69% in §3. What is
left is turning it into a figure: `glyph_p75` against agreement, per capture.

**5 · Count the Latin-fragment leakage.** *An hour.* `හාත්kumba`,
`පුවත්පත්ikon`, `කොන්ත්‍රාත්prices` — a named over-correction mode not yet in
the taxonomy.

**6 · NEW — re-run everything on the pinned environment.** The 27 Aug
measurements (headline separation, the 70-capture census, the YOLO probe) were
made under **cv2 5.0.0**, not the pinned 4.9.0. The functions used are
version-stable — unlike `minAreaRect`, which is why `deskew_angle` uses a
projection profile — but nothing from 27 Aug should enter a chapter until it has
been reproduced under 4.9.0.

### Do not do

- **Retraining or more data.** 38 → 230 pages moved CER by 0.0006. Measured.
- **Surya.** Failed on dependencies; the deployability failure is itself a
  legitimate finding.
- **Custom hardware.** A phone in a head mount demonstrates everything.
- **Reviving the gated architecture.** Your strongest contribution *as a
  negative result*.
- **Turning the article detector back on in the phone path** without new
  evidence. 69% disagreement is the evidence against.

---

## 7. Still open in the system

| Item | Owner | Blocking the demo? |
|---|---|---|
| 16% of captures cannot be segmented → "move closer" | me | no — but it is one capture in six |
| Two stories side by side, no headline between them in view | me | no capture in 70 shows it; a gap, not a measured failure |
| Coloured (red) headline words lost to grayscale threshold | me | no |
| Title OCR not re-measured on the pinned cv2 | me | affects Chapter 4 wording |
| Component 4 (voice/intent/personalisation) still a stub | Bumal | no — 9 local commands work offline |
| Sinhala strings not checked by a native speaker | me + team | **yes, before the viva** |
| Android app written but never built or run on the device | me | **yes** |
| RAG answer quality never judged by a Sinhala reader | Nadee + me | no |
| Cut-off column at the frame edge yields fragments | me | no |
| Latin fragments from mT5 | me — needs measuring first | no |
| A repeated passage `strong_dedup` cannot span | me | no |
| §4.2's 1.00× baseline unidentified | you — read `Pipeline_v11` | affects Chapter 4 wording |
| D5, the v1 CER figure (0.0274 vs 0.0238) | you | affects contribution #4 |

---

## 8. Running it

```
cd E:\RP\R26-DS-002\system
python -m pytest tests -q          # expect 245 passed
python tools\run_all.py            # reader + svc-rag
```

`http://127.0.0.1:8000/debug` is the whole system without a phone: upload
frames, see raw OCR beside corrected text, then ask a question.

Reproduce the measurements in this document:

```
python tools\measure_headline.py --ocr     # the headline constants
python tools\calibrate_quality.py          # the readability thresholds
python tools\probe_yolo.py --root ...      # the 69% disagreement
```
