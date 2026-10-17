# Android Auto-Capture Reader — Complete Build Record

**Project:** R26-DS-002 · Component 2 · Android capture module
**Student:** Madhusanka H.P.I (IT22259134), SLIIT
**Device under test:** Samsung SM-A705FN (Galaxy A70), Android 11
**Build date:** 19–20 August 2026

---

## 0. What this document is

A complete record of building the Android capture application: what was built,
in what order, every measurement taken on the device, every finding, every
mistake and its fix, and the constants that came out of it.

It is written so that the app could be rebuilt from nothing using only this
document, and so that every number in it can be defended at a viva.

Companion document: `claude/Android_Capture_Guidance_Calibration.md` — the
calibration study in detail, with the estimator-comparison figure.

---

## 1. Why the app exists

The research established a hard capture requirement: base glyphs must reach a
certain pixel height in the captured photo or the Sinhala dependent vowel signs
(*pilla*) fall below recoverability, and no amount of post-OCR correction can
restore evidence that is absent from the image.

Two facts make this an interaction problem rather than a specification problem:

1. **A blind user cannot judge that distance visually.** Asking them to press a
   shutter and then telling them the photo was unusable is the wrong interaction.
2. **The system already knows the threshold.** So it should ask for a photo it
   can read, and take it the moment the conditions are met.

The project's own data shows this is not hypothetical. In
`layout/page_diagnostics.csv`, of roughly 190 photographed pages, **seven are
marked OK**; every full-page framing sits at `glyph_p75` 17–21 and every
half-page framing at 21–23, against a pass mark of 25. The entire ground-truth
corpus was photographed below the resolution its own OCR engine requires. That
is the defect this module corrects.

---

## 2. Environment

| Item | Value |
|---|---|
| IDE | Android Studio, Kotlin, **Empty Views Activity** template |
| Package | `lk.sliit.r26ds002.sinhalareader` |
| minSdk / targetSdk / compileSdk | 26 / 35 / 35 |
| Java / Kotlin JVM target | 17 |
| CameraX | 1.4.2 (`camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view`) |
| Networking | Retrofit 2.11.0, OkHttp logging-interceptor 4.12.0, coroutines 1.8.1 |
| View binding | enabled |
| Backend | FastAPI + uvicorn on the laptop (RTX 4060) |

**Views, not Compose.** `PreviewView` is a classic Android `View`; using it from
Compose means wrapping it in `AndroidView` interop and pushing every update from
the analysis thread through recomposition. For a 30 fps measurement loop feeding
a text readout that is extra machinery between the developer and the thing being
demonstrated.

---

## 3. Architecture

```
┌─ Android (Kotlin, CameraX) ────────────────────────────────┐
│                                                             │
│  Preview ───────────────► screen (sighted helper / debug)   │
│                                                             │
│  ImageAnalysis ─────────► GlyphMeasurement (every frame)    │
│     1440×1080 YUV         │  crop → Otsu → line runs        │
│     KEEP_ONLY_LATEST      │  → pitch → glyph height         │
│                           │  → Laplacian sharpness          │
│                           │                                 │
│                           ├─► GuidanceMapper → TTS          │
│                           └─► AutoShutter                   │
│                                    │                        │
│  ImageCapture ◄────────────────────┘ 5 frames, keep 3       │
│     3264×2448 JPEG                                          │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │ multipart POST /capture
          ▼
┌─ FastAPI backend (laptop) ─────────────────────────────────┐
│  exif_transpose → YOLO articles → PaddleOCR layout →        │
│  Tesseract per column → consensus → mT5 → TTS               │
│  title audio returned first, body follows                   │
└─────────┬───────────────────────────────────────────────────┘
          │ audio
          ▼
    MediaPlayer playback
```

### Source files

| File | Responsibility |
|---|---|
| `MainActivity.kt` | camera binding, orchestration, UI, capture/upload/playback cycle |
| `GlyphMeasurement.kt` | pure measurement: crop, Otsu, line runs, pitch, sharpness |
| `GlyphAnalyzer.kt` | `ImageAnalysis.Analyzer`; produces one `FrameState` per frame |
| `Guidance.kt` | thresholds, `GuidanceState` enum, `GuidanceMapper` state machine |
| `GuidanceSpeaker.kt` | TextToSpeech wrapper with throttling and language fallback |
| `AutoShutter.kt` | the fire/don't-fire decision |
| `ReaderApi.kt` | Retrofit interface and OkHttp client |
| `res/layout/activity_main.xml` | PreviewView, crop overlay, guidance text, diagnostics |
| `res/xml/network_security_config.xml` | permits cleartext HTTP on the research LAN |

`GlyphMeasurement.kt` is deliberately free functions over primitive arrays, so
it can be unit-tested on the JVM without a device.

---

## 4. Build order

Each step is independently testable and each one proves something specific.
Nothing was stacked on an unverified layer.

| # | Step | Proves | Status |
|---|---|---|---|
| 1 | Empty project + CameraX preview | CameraX binds, camera shows | done |
| 2 | ImageAnalysis + resolution logging | analysis/capture ratio, rotation, strides | done |
| 3 | Glyph measurement on screen | the estimator responds to distance | done |
| 4 | On-screen guidance states | thresholds and anti-chatter behave | done |
| 5 | Throttled TextToSpeech | speech is usable, not chatter | done |
| 6 | Auto-shutter | fires only when in range and steady | done |
| 7 | Burst capture | multiple usable frames per trigger | done |
| 8 | Upload to placeholder `/capture` | full round trip incl. playback | done |
| 9 | Real pipeline behind the endpoint | end-to-end reading | in progress |

Steps 1–6 need no backend at all. Steps 3–4 alone make a convincing viva
demonstration: a phone showing a live glyph-height reading and telling the user
where to stand is the capture finding made visible.

---

## 5. Measured device characteristics

Logged at startup by `logGeometry()`:

```
preview  = 1440x1080  (4:3)
analysis = 1440x1080  (4:3)
capture  = 3264x2448  (4:3)
ratio: width=2.267 height=2.267
rot 90   rowStride 1472 / pixelStride 1
fps 29.8 (daylight) · 20.2 (indoor, exposure-limited)
```

Five things follow from that block, and all five shaped the code.

**The width and height ratios are identical.** That proves the analysis and
capture streams share a field of view, so the relationship between them is a
pure scale factor with no cropping term. It was not guaranteed — see the next
point.

**CameraX ignored the requested 1920×1080.** Its default aspect-ratio strategy
filters candidate sizes to 4:3 *before* applying the resolution strategy, so it
returned 1440×1080. That accident is fortunate: it matched the capture's 4:3.
Had a 16:9 analysis stream been granted, the ratios would have read 2.27 and
3.02 and a cropping correction would have been required.

**`rotationDegrees = 90`.** The activity is locked to portrait; the sensor
delivers a landscape buffer. A horizontal line of text in the upright image is
therefore a band of **columns** in the raw buffer. The projection profile cannot
simply sum rows.

**`rowStride` 1472 against width 1440.** Every row carries 32 bytes of padding.
Code that assumed a contiguous 1440-byte row would shear the image progressively
down the frame.

**Frame rate is exposure-limited, not compute-limited.** 20 fps indoors, 30 in
daylight, with an analyzer doing no work. Worth knowing so the measurement code
is not blamed for it later.

---

## 6. The measurement: what was tried, what works

The research acceptance criterion is stated in **base glyph height** —
specifically `glyph_p75`, the 75th percentile of connected-component heights
measured on the captured photo. (The deployment design says p90;
`page_diagnostics.csv` uses p50 and p75, and its OK/MARGINAL verdict is driven
by p75. The CSV is authoritative.)

The phone measures on the analysis stream, so something measurable there must
map onto that. Three candidates were tested by reproducing the whole chain on
synthetic Sinhala newspaper pages — render at capture resolution, compute the
research metric, downscale by 2.267, centre-crop 640×480, run a line-for-line
Python port of the Kotlin.

### 6.1 Line-run height — rejected

Median height of runs of rows whose ink exceeds a threshold.

Across base glyph heights of 17 → 29 px — **the entire operating range** — the
measured value stayed pinned at 8.0 analysis px and never moved. The cause is a
fixed ink-density threshold: after downscaling, anti-aliased ascender and
descender rows fall below it, so the run collapses onto the x-height core and
stops responding to scale. An adaptive threshold (a fraction of the busiest rows
rather than of the crop width) restores the response but leaves 4–11% spread and
1 px quantisation.

Retained as a displayed diagnostic only.

### 6.2 Connected-component height on the analysis crop — rejected

The obvious approach: compute the research metric itself on-device, scale by
2.267, and need no calibration constant at all.

On pristine renders it works beautifully — `predicted / true` = 0.99–1.09 across
G = 22–70 px, with no dependence on the publication's typography.

It does not survive contact with a camera:

| condition | p90 CC drift | line pitch drift |
|---|---|---|
| pristine | 0.0% | 0.0% |
| mild blur | +16.3% | −0.04% |
| realistic hand-held | **+52.3%** | −0.21% |
| poor (blur + noise) | +39.7% | +0.92% |
| bad (heavy blur) | +15.7% | +0.24% |

Blur fuses adjacent components — vowel signs merge into their base glyphs,
neighbouring glyphs touch — and every merge grows the bounding box. Individual
sizes reached +147%. Reducing the crop makes it worse.

This is the deployment design's "Option B — OpenCV parity" route. It would have
cost 40–100 MB of APK to be quantitatively wrong under exactly the conditions
the application operates in.

### 6.3 Line pitch — selected

Mean baseline-to-baseline spacing, outlier-trimmed.

Pitch measures **where lines start**, and blur does not move a baseline — it
only thickens the ink around it. Drift under every degradation condition stayed
under 1%. It is also measured across ~25 lines, so averaging gives genuine
sub-pixel resolution rather than the 1 px steps a single run length allows.

### 6.4 The algorithm as implemented

```
centreCropUpright(image, 640, 480)     // handles rot 90 and rowStride
  → otsuThreshold(crop)                // adapts to lighting
  → lineRuns(crop, thr)                // adaptive ink threshold
  → meanPitch(starts)                  // with validity tests
  → glyph = pitch × captureRatio ÷ PITCH_PER_GLYPH
```

`centreCropUpright` reads the **source** sequentially — one bulk `get` per
buffer row — and scatters into the destination, because strided writes into a
small heap array are far cheaper than strided reads from a direct ByteBuffer.

`lineRuns` uses an ink threshold relative to the 90th-percentile row ink of that
crop, not a fixed fraction of crop width. That is what fixed the saturation in
6.1 and it also handles crops that are partly white space.

`meanPitch` applies three validity tests before returning a value:

1. **enough lines** to average over (≥ 6)
2. **enough gaps survive** the ±40% trim (≥ 50%) — a mixture of two spacings
   leaves almost nothing, which is exactly what should be rejected
3. **coefficient of variation** of the surviving gaps is small (≤ 0.30)

Without those tests the estimator happily returned a pitch of 67 px from seven
irregularly spaced runs — a headline block — which read as a glyph height of
116 px and threw the guidance into "move back" at arm's length. The observed
symptom was eighteen state changes in eight seconds.

The rejection reason is surfaced in the UI (`ok` / `few-lines` / `irregular` /
`uneven` / `range`) and logged as a periodic census, so the filter can be tuned
against data rather than guesswork.

---

## 7. Calibration

### 7.1 The constant

`R = pitch ÷ glyph_p75`, in the same units.

**On the research corpus.** Twelve Dinamina pages from `layout/raw_pages/`
(3024×4032), with `glyph_p75` taken from the project's own
`page_diagnostics.csv` so the measurement does not depend on reimplementing the
component filter. Excluding two pages whose sampling tiles landed on headline
blocks: median 1.28, mean 1.32. Synthetic renders across four font/leading
combinations gave 1.33 independently.

**On the phone's own captures.** The app's live estimate was stamped into each
filename (`burst_<stamp>_g27_s3615_4.jpg`) so prediction and result could be
paired:

| operating point | pitch (capture px) | measured p75 | implied R |
|---|---|---|---|
| further | 37.7 | 19.5 | 1.93 |
| closer | 52.7 | 31.5 | 1.67 |

Two findings.

**R is a property of the capture device, not only of the typeface.** The A70's
8 MP output is downsampled and sharpened from a 32 MP sensor; components
fragment more and p75 reads lower for the same physical text. 1.30 does not
transfer.

**The relationship is affine, not proportional.** Fitting both points gives

```
glyph_p75 ≈ 0.80 × pitch_capture − 10.7
```

p75 falls faster than the text shrinks, because at smaller scales components
fragment and merge more readily. A single multiplicative constant therefore
cannot be right across the whole range — it only has to be right near the
decision threshold. At p75 = 25 the fitted line gives pitch ≈ 45, hence:

**Adopted: `PITCH_PER_GLYPH = 1.80`**, calibrated at the acceptance threshold.

### 7.2 How to re-calibrate for a different phone

1. Set `PITCH_PER_GLYPH` to any starting value.
2. Confirm the filename carries `_g<estimate>_`.
3. Take 5–6 bursts at the ready distance on well-lit body text.
4. Run `check_captures.py`; it prints `R_implied = R_used × app_g / p75` per
   frame and the median over sharp frames.
5. Set the constant to that median, restricted to frames above ~1000 sharpness.

---

## 8. Guidance

Thresholds are stated in **base glyph pixels** — the research's own units — so
the acceptance criterion in the app is literally the criterion in the thesis.

| State | glyph p75 | pitch (analysis px) @ R=1.80 | English | Sinhala |
|---|---|---|---|---|
| seek | no reading | — | Point at the newspaper | පුවත්පත දෙසට යොමු කරන්න |
| far | < 15 | < 11.9 | Move much closer | තවත් ළං වන්න |
| near | 15–28 | 11.9–22.2 | A little closer | ටිකක් ළං වන්න |
| **ready** | **28–40** | **22.2–31.8** | **Hold steady** | **නිශ්චලව තබාගන්න** |
| close | 40–50 | 31.8–39.7 | Slightly back | ටිකක් ඈත් වන්න |
| vclose | > 50 | > 39.7 | Move back | ඈත් වන්න |

**The ready band opens at 28, not 25.** The pass mark is 25; three pixels is
deliberate margin. Keeping the margin explicit — rather than hidden inside a
deliberately conservative constant — means the number on screen means what it
says and the safety allowance can be defended as a design decision.

### Three mechanisms stop the guidance being useless

**Hysteresis (2 px).** A state is left only when the reading passes the boundary
by an extra margin. Without it a hand held at exactly 28.0 crosses the boundary
several times a second and the app alternates between "a little closer" and
"hold steady" — actively misleading, because the user keeps moving in response
to contradictory instructions.

**Text-present gating.** A missing estimate has two completely different causes
and conflating them produces the worst possible instruction:

- *text present, estimate rejected* → the page is in frame and this frame's
  reading is noisy. The camera has not moved, so the last instruction is still
  correct: **hold it, say nothing**.
- *no text at all* → the page has left the frame. Only this earns "point at the
  newspaper".

Before this distinction existed the app told a user pointing straight at a
newspaper to point at the newspaper — sending them searching for something they
had already found. The gate is `lineCount ≥ 8`.

**Seek grace (20 frames ≈ 0.7 s).** A single blurred frame or a hand passing
over the page must not reset guidance.

### Speech

`GuidanceSpeaker` is called on **every** frame with the current state, not only
on change. It decides internally whether to speak:

- `QUEUE_FLUSH` always, so a stale instruction is cut off rather than queued
  behind the current position
- speak on state change, with a 1.2 s minimum interval; if a change is
  suppressed by that floor the next frame retries it — the naive "speak only
  when changed" version drops the instruction permanently and the user waits for
  a cue that never comes
- re-announce a non-READY state after 6 s of silence, because to a blind user
  silence is ambiguous between "still too far" and "the app has stopped"

Sinhala is used when a Sinhala voice is present, English otherwise. **On the
test device no Sinhala voice is installed** (`tts en` in the readout) — install
one via Settings → General management → Text-to-speech before the demo.

---

## 9. Auto-shutter and frame selection

### 9.1 The trigger

Three conditions, all of which must hold for **4 consecutive frames**. Any
failure resets the counter to zero — not decrements it. One lucky frame is not
evidence the camera is steady.

1. **In range** — guidance is READY
2. **Sharp** — Laplacian variance ≥ 600
3. **Not drifting** — glyph estimate spread across the window < 6%

Condition 3 is the calibration-free steadiness test: unlike sharpness it needs
no device- or content-specific threshold, and it catches slow drift that
individual frames are too sharp to reveal.

Plus a 4 s cooldown, so one good moment yields one capture rather than a burst
of near-identical ones.

The blocking condition is displayed every frame (`not-ready`, `blurry`,
`drifting`, `settling 2/4`, `cooldown`), which makes the shutter debuggable
without a debugger.

**The sharpness threshold must be measured per device.** The absolute scale of a
Laplacian variance depends entirely on the implementation. On this device: ~1545
holding steady, ~3 while moving. The design document's suggested 80 was from a
different implementation and would have accepted almost anything.

### 9.2 Selection after the shutter — the important part

The shutter decides on a preview frame, but analysis is **detached** while the
burst runs, so nothing observes the ~2 s during which the photos are actually
taken. Measured:

| fired at sharpness | frame 1 | frame 2 | frame 3 |
|---|---|---|---|
| 4002 | 1024 | 273 | 229 |
| 893 | 722 | 535 | 203 |
| 858 | 1589 | 722 | 1504 |

The shutter fired on a genuinely crisp frame and delivered smeared photos.
Sharpness degrades across the burst in every case, because the user relaxes once
they think it is over.

**Raising the shutter threshold cannot fix this — it gates the wrong moment.**

The burst therefore captures **five** frames and uploads the **three sharpest**,
scored on device by Laplacian variance of an eighth-scale decode (~40 ms per
frame; blur is a global property, so a quarter of the pixels ranks frames just
as well). This is the smart-frame-selection pattern from the wearable literature
(GLIMPSE, 2026), applied for the same reason.

The user is told "Hold still" at the start of the burst — without an audio cue
a blind user has no way to know the two seconds have begun.

It reduces rather than eliminates bad frames: one capture survived selection at
sharpness 327 and yielded p75 6. The multi-frame consensus stage is what
outvotes that frame.

### 9.3 Why three frames, sequentially

Three gives the consensus a majority to vote on. Frames of the same article
taken moments apart at the same distance are, by construction, views of
**comparable quality** — the condition the consensus experiments identified as
necessary for voting to help.

Sequential, never concurrent: firing several `takePicture` calls at once drops
frames on most devices, because the capture pipeline has a limited number of
in-flight requests and silently discards the surplus. You get three callbacks
and two files.

---

## 10. Validation

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

**Every frame clears the 25 px pass mark.** Seven of roughly 190 pages in the
existing ground-truth corpus do.

### Coherence with the article detector

At p75 ≈ 25–30 the phone sits roughly 1.4× closer than the full-page framings in
the corpus, putting about half a page in frame. Half-page framing is what the
YOLO11m article detector was trained and evaluated on (93.8% grouping accuracy
on half framings, zero over-merge). The distance the guidance pushes the user to
and the distance the segmentation model handles best coincide — arrived at from
two independent constraints, not designed in.

---

## 11. Measured performance

| Stage | Time |
|---|---|
| Per-frame analysis (crop + Otsu + profile + sharpness) | 7–12 ms |
| Frame budget at 30 fps | 33 ms |
| Burst of 5 frames | ~2.8 s |
| On-device selection (5 decodes) | ~0.2 s |
| Upload of 3 frames (6.3 MB) + stub response | ~2.0 s |
| **Shutter → audio (stub backend)** | **~5 s** |

The design document estimated 3 ms per frame; the difference is the rotation
transpose of 307,200 pixels needed to correct `rotationDegrees = 90`, which the
estimate did not include. Reported as measured, not as estimated.

The 2 s upload is pure network and does not overlap pipeline work. That is the
concrete argument for returning title audio before body audio: a Sinhala
headline takes three to four seconds to speak, which is roughly how long body
OCR and correction need, so the user perceives almost no wait.

---

## 12. Constants reference

Every tunable, with where its value came from.

| Constant | Value | File | Provenance |
|---|---|---|---|
| `PITCH_PER_GLYPH` | 1.80 | MainActivity | fitted on-device at the acceptance threshold |
| `NEAR_READY` | 28 | Guidance | pass mark 25 + 3 px stated margin |
| `READY_CLOSE` | 40 | Guidance | field-of-view limit, not a quality limit |
| `FAR_NEAR` / `CLOSE_VCLOSE` | 15 / 50 | Guidance | coarse guidance zones |
| `hysteresisPx` | 2.0 | Guidance | observed boundary noise |
| `seekGraceFrames` | 20 | Guidance | ≈0.7 s at 30 fps |
| `TEXT_PRESENT_LINES` | 8 | MainActivity | separates "no page" from "noisy frame" |
| `minSharpness` | 600 | AutoShutter | device-measured: 1545 steady, 3 moving |
| `stableFrames` | 4 | AutoShutter | design document |
| `maxSpreadFraction` | 0.06 | AutoShutter | drift tolerance |
| `cooldownMs` | 4000 | AutoShutter | one capture per positioning |
| `BURST_CAPTURE` / `BURST_KEEP` | 5 / 3 | MainActivity | measured intra-burst degradation |
| `cropW` × `cropH` | 640 × 480 | GlyphAnalyzer | ~25 lines, spans two columns |
| `inkFracOfPeak` | 0.35 | GlyphMeasurement | adaptive; fixed fraction saturated |
| `maxCv` | 0.30 | GlyphMeasurement | two-column baselines are never perfectly aligned |
| `minKeptFraction` | 0.5 | GlyphMeasurement | rejects mixed line spacings |
| smoothing window | 7 frames, **median** | MainActivity | median, not mean — see below |

**Median, not mean, everywhere.** One merged pair of text lines produces an
outlier roughly double the true value. A mean over the window would drag the
estimate up ~20% and could push the state machine into a false "ready". A median
discards it entirely.

---

## 13. Pitfalls encountered — symptom, cause, fix

Every one of these cost real time. They are recorded so they cost nobody else
any.

### Android / CameraX

| Symptom | Cause | Fix |
|---|---|---|
| No `activity_main.xml`, a `ui.theme` package present | "Empty Activity" (Compose) chosen instead of "Empty Views Activity" | recreate with the Views template |
| Camera works ~half a second then freezes | `imageProxy.close()` not called | close in a `finally` block, always |
| Image sheared progressively down the frame | assumed contiguous rows | index rows explicitly; `rowStride` is 1472 vs width 1440 |
| Text lines not detected at all | `rotationDegrees = 90`; lines run along buffer **columns** | rotate the crop to upright before profiling |
| `resolutionInfo` is null | read before `bindToLifecycle` | read after binding; sizes are negotiated at bind time |
| Analysis returned 1440×1080 despite requesting 1920×1080 | default aspect-ratio strategy filters to 4:3 first | accept it — it matched the capture aspect and removed a crop term |
| `Unresolved reference: addLast` | `ArrayDeque` API differs by resolution; Kotlin 2.x is deprecating `removeFirst`/`addLast` | fixed-size `FloatArray` ring buffer — also allocation-free in the hot path |
| `Type mismatch: PitchEstimate but Float? expected` | file merged rather than replaced | replace whole files when a signature changes |
| Captures sideways in some viewers | CameraX writes rotation to EXIF; it does not rotate pixels | `setTargetRotation`, and `ImageOps.exif_transpose()` server-side |
| Files not visible in the phone's file manager | Android 11 hides `Android/data` from file managers | Device Explorer, or `adb pull` |
| Capture folder does not exist at all | `getExternalFilesDir()` creates it on first call — so the shutter never fired | read the `shutter` reason on screen |

### TextToSpeech

| Symptom | Cause | Fix |
|---|---|---|
| TTS init succeeds but nothing is ever spoken | API 30+ package visibility hides the TTS engine | `<queries><intent><action android:name="android.intent.action.TTS_SERVICE"/></intent></queries>` |
| Occasional crash configuring language in the init callback | callback can fire before the constructor assigns the field | configure lazily on first `announce()` |
| Guidance in English | no Sinhala voice installed on the device | install one; the app falls back correctly meanwhile |

### Networking

| Symptom | Cause | Fix |
|---|---|---|
| `failed to connect to /192.168.1.100 from /172.20.10.6` | the example IP was never changed | read the real IPv4 from `ipconfig`; the phone's own address in the error tells you the subnet |
| `CLEARTEXT communication not permitted` | Android blocks plain HTTP from API 28 | `network_security_config.xml` + `usesCleartextTraffic` |
| Browser cannot load `:8000/docs` from either device | uvicorn not running, or bound to 127.0.0.1 | `--host 0.0.0.0`; check the startup line says `0.0.0.0` |
| Ping works but port 8000 refused | Windows Firewall allows ICMP, not that TCP port | `netsh advfirewall firewall add rule name="uvicorn8000" dir=in action=allow protocol=TCP localport=8000` |
| `/docs` loads but renders blank | Swagger UI pulls CSS/JS from a CDN; the hotspot has no internet | cosmetic; use a plain JSON route instead |
| PowerShell rejects `"%LOCALAPPDATA%\...\adb.exe" devices` | cmd syntax in PowerShell | `& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" devices` |

**The reliable fallback for any network problem**, and the recommended path for
the viva itself:

```
adb reverse tcp:8000 tcp:8000
BACKEND_BASE_URL = "http://127.0.0.1:8000/"
```

Traffic goes down the USB cable. No WiFi, no firewall, no hotspot isolation. Re-run after each unplug.

### Measurement

| Symptom | Cause | Fix |
|---|---|---|
| Reading stuck at 8 px across the whole operating band | fixed ink threshold loses ascender rows after downscaling | adaptive threshold relative to the crop's own ink peak |
| Glyph readings of 116 px at arm's length | pitch computed from a headline block's irregular spacing | gap-regularity validity tests |
| Eighteen guidance changes in eight seconds | outliers passing through the median window in bursts | validity tests + hysteresis |
| "Point at the newspaper" while pointing at it | rejected estimate treated as "no page" | `textPresent` gating |
| p75 varying 8 → 25 within one burst | blur, not distance — smeared ink fragments after Otsu | frame selection |
| `lines 0` with the crop over text, in dim light | low contrast: gaps never fall below the ink threshold, whole crop merges into one run > `maxRun` | **open** |

---

## 14. Backend

### Stub (for bring-up)

`stub_server.py` stores the burst and returns a fixed audio file. Purpose: prove
the round trip before any model is involved. A dashboard version renders
received bursts with thumbnails so captures can be inspected without pulling
files off the phone.

```powershell
pip install fastapi uvicorn python-multipart pillow
python -m uvicorn stub_server:app --host 0.0.0.0 --port 8000
```

### Real pipeline (step 9)

Two files: `pipeline.py` (stages, each independently replaceable, with a `MOCK`
flag) and `server.py` (orchestration and the API).

`MOCK = True` lets the whole API be exercised before any model is loaded, so the
Android side can be built and debugged against real timings and real failure
modes rather than against nothing. Each stage is then replaced one at a time
with the notebook code; the server never changes.

Stages, matching §12.3 of the research summary:

```
exif_transpose → YOLO11m article boxes → border filter
→ PaddleOCR layout on the FULL page → containment assign → de-overlap
→ column order (L→R, each column T→B) → Tesseract per single-column region
→ de-duplicate → multi-frame consensus (medoid per line) → mT5 sentence-by-sentence
→ TTS
```

**API shape — title first:**

| Route | Returns |
|---|---|
| `POST /capture` | `{job, title, title_audio}` as soon as the title is read |
| `GET /result/{job}` | `{status, title, body, body_audio}`; status pending → ready |
| `GET /audio/{job}/{title\|body}` | the audio file |

The response carries **text as well as audio**. If a Sinhala voice is installed
on the phone, the app can speak the returned text with its own TTS — lower
latency, less data, and better Sinhala than an offline server-side engine.

Two things the pipeline must do that the notebooks did not:

- **`exif_transpose` on arrival.** PIL ignores EXIF by default, so without it the
  detector receives pages turned 90°, and deskew corrects degrees, not quarter
  turns.
- **Load models once at startup.** Loading YOLO and mT5 per request adds seconds
  to every capture.

---

## 15. Not done / known open items

1. **`lines 0` in dim light.** With low contrast the inter-line gaps never drop
   below the adaptive ink threshold, the crop merges into a single run longer
   than `maxRun`, and it is discarded. Candidate fixes: adaptive `maxRun`,
   narrowing the crop to a single column, or a contrast pre-check.
2. **mDNS / NSD discovery.** The backend address is a compile-time constant.
   `NsdManager` on Android (no new dependency) plus `zeroconf` on the laptop
   would let the app find the server on any network. Strongly recommended before
   the viva, because the room's WiFi will not be the development network.
3. **Manual server-address override** persisted in `SharedPreferences`, as the
   fallback for networks that block multicast.
4. **Sinhala TTS voice** is not installed on the test device.
5. **Real pipeline** behind `/capture` — step 9, in progress.
6. **Two-stage playback on Android** — the app currently expects a single audio
   response; the title-then-body client is not yet written.
7. **Frame selection is not exhaustive** — one frame at sharpness 327 survived.
8. **`R` is fitted at one operating point.** The relationship is mildly affine;
   a two-point calibration would be more accurate across the range.

---

## 16. For the report

### Framing

This is deployment engineering, not a research contribution — but it is a direct
application of one, and it produces a measurable result. Describe it as:

> *Capture requirements specified as a target glyph height, enforced at capture
> time by an on-device estimator selected for invariance to motion blur, rather
> than as a sensor-resolution specification.*

### The claims that are defensible

1. **The corpus was under-resolved and this is measurable.** Seven of ~190 pages
   meet the project's own pass mark. Every full-page framing fails it.
2. **A blur-invariant proxy can enforce a resolution requirement live.** Line
   pitch drifts under 1% across degradation that moves connected-component
   height by 52%.
3. **The obvious approach is quantitatively wrong.** Computing the research
   metric itself on-device is accurate on clean images and inflates by up to 52%
   under hand-held conditions — a negative result that only appears if the
   degradation is simulated rather than assumed.
4. **The calibration constant is device-specific**, and the relationship between
   proxy and metric is affine rather than proportional.
5. **The captures meet the threshold**: eight consecutive auto-fired frames at
   p75 29–34.
6. **The required distance coincides with the detector's best framing**, from
   two independent constraints.

### Citations to attach

- GLIMPSE (2026) — blur rejection and smart frame selection in wearables
- optimal-stopping work on video text recognition — the burst-stopping rule
- Velayuthan & Ambegoda, arXiv:2507.18264 (2025) — Surya as the strongest
  Sinhala OCR, for the future-work argument

### Do not

Do not build custom hardware before the deadline. A phone in a head mount
running this app demonstrates everything the wearable would.

---

## 17. Rebuild checklist

1. New project, **Empty Views Activity**, Kotlin, minSdk 26.
2. `build.gradle.kts`: CameraX 1.4.2 ×4, Retrofit, OkHttp logging, coroutines,
   `viewBinding = true`, Java 17.
3. Manifest: `CAMERA` + `INTERNET`, camera features, TTS `<queries>`,
   `networkSecurityConfig`, portrait lock, `configChanges`.
4. `res/xml/network_security_config.xml`, `res/drawable/crop_frame.xml`,
   `res/layout/activity_main.xml`, strings.
5. Seven Kotlin files (§3).
6. Backend folder, venv, `pip install fastapi uvicorn python-multipart pillow
   opencv-python numpy`, `python -m uvicorn server:app --host 0.0.0.0 --port 8000`.
7. Set `BACKEND_BASE_URL`, or `adb reverse tcp:8000 tcp:8000` and use
   `127.0.0.1`.
8. Calibrate: `minSharpness` from the on-screen `sharp` value steady vs moving;
   `PITCH_PER_GLYPH` from `check_captures.py` over 5–6 bursts.
9. Verify: p75 ≥ 25 on auto-fired captures.

### Diagnostic tooling

| Tool | Purpose |
|---|---|
| on-screen readout | glyph, state, sharp, shutter reason, lines, cv, proc, fps |
| `guidance ->` log | one line per state change with the value that caused it |
| `pitch reasons:` log | rejection census every ~2 s |
| `frame scores` log | per-frame sharpness and which survived selection |
| filename `_g27_s3615_` | app estimate and sharpness at fire time, for pairing |
| `check_captures.py` | p50/p75/sharpness/R_implied per capture, and the fitted constant |
| backend dashboard | received bursts with thumbnails, auto-refreshing |

The filename convention is the single most useful piece of instrumentation in
the project: it is what turned "the captures look fine" into a fitted constant
and a validation table.
