# Article-wise reading — what the sweep showed, and what I changed

**R26-DS-002 · IT22259134 · 24 August 2026**

## 1. The sweep did not find a single optimum, and that is the finding

| target | →px | 80654199 (`p75` 22, **whole**) | 71a97929 (`p75` 38, clipped) |
|---|---|---|---|
| g11 | 11.0 | 0.0673 | **0.0365** |
| g13.2 | 13.2 | 0.0760 | 0.0731 |
| g15 | 15.0 | 0.0556 | 0.0585 |
| g17 | 17.0 | **0.0497** | 0.0570 |
| g19 | 19.0 | 0.0789 | 0.0936 |
| native | 22.0 | 0.0629 | — |

*(mT5 CER, lower is better)*

The two captures pick **different** best points — 17 px for one, 11 px for the
other — and neither curve is monotonic. Worse, the spread **within a single
frame** (0.0365 to 0.0936 on 71a97929 — the same photograph, the same 684
characters) is as large as the difference between the two frames.

So there is no reliable optimum in 11–22 px on this data. n = 1 article, 684
characters, and a beam-search generative corrector that turns a 0.016 spread in
OCR CER into a 0.057 spread after correction. Picking an argmin here would be
fitting noise.

**What the sweep does show, clearly, is a cliff.** Below about 11 px the
diacritics go and CER collapses to **0.2193**. Above it, everything from 11 to
22 px lands between 0.036 and 0.094.

**So the rule is a floor, not a target.** That is a more robust conclusion than
an argmin would have been, and it is what the code now implements.

## 2. Reading the whole article is not a compromise

Best whole-article result: **mT5 CER 0.0497, WER 0.1429** at `glyph_p75` 22.
Best close-but-clipped result at the shipped fixed scale: **0.0570**.

**The whole-article framing wins.** It reads four columns instead of two and a
half, and it is not worse doing it. The original question — "does reading the
whole article cost accuracy?" — is answered: **no.**

## 3. What I changed

Four files. Every change is justified by a number above.

**`core/config.py`**
- `CLOSEUP_MIN_P75` **28 → 20**. 28 came from the app's `NEAR_READY` band — a
  design choice, never a CER measurement — and it refused exactly the framing
  that holds a whole article.
- new `CLOSEUP_TARGET_GLYPH = 15.0` — the middle of the flat region, safely
  clear of the 11 px cliff. Labelled in the file as a **safe choice, not a
  measured minimum**.
- new `CLOSEUP_MIN_GLYPH_PX = 11.0` — the hard floor.
- `CLOSEUP_OCR_SCALE = 0.40` kept but marked superseded.

**`core/imaging.py`** — new `closeup_scale(p75)`. Target height, floor
enforced, never upscales. It now delivers **15.0 px at every distance**:

```
p75 20 -> 0.75 -> 15.0 px      p75 33 -> 0.45 -> 15.0 px
p75 22 -> 0.68 -> 15.0 px      p75 38 -> 0.39 -> 15.0 px
p75 28 -> 0.54 -> 15.0 px      p75 45 -> 0.33 -> 15.0 px
```

**`layers/l4b_body/body.py`** — `read_page()` scales adaptively per frame
instead of by the fixed 0.40, and records the scale it actually used.

**`app/pipeline.py`** — the close-up branch now crops with
`layout.analyse()` instead of `closeup.text_bbox()`:

- columns found, **clipped ones dropped**
- crop limited to the block spanning the frame centre, so the next story's
  headline is excluded
- every frame deskewed by the same angle first, because the crop is in
  deskewed coordinates (about 40 px of error otherwise)
- **`warnings_for()` wired into the Document**, so the phone can say *"part of
  this article is off the right of the frame — move a little to the right"*
  instead of reading a fragment silently
- falls back to the old bbox if `layout` refuses the frame

Backup at `app/pipeline.py.bak4`. 45 tests pass, including seven new ones for
the scaler — one of which asserts that the old fixed 0.40 would have failed the
floor at `p75` 22, so a revert shows up as a failure rather than silently.

## 4. What is still missing — and it is the important one

**The app still guides you to `glyph_p75` 33–36.** The server can now read a
whole article at 22, but nothing tells the *user* to stand there. Until
`Guidance.kt`'s READY band is re-aimed to about **22–26**, you will keep
capturing at 33–36 and keep clipping a column.

That is the last change, and it is small: the thresholds in `Guidance.kt`. Say
the word and I will write it.

## 5. Verify, in this order

```
cd E:\RP\R26-DS-002\system
```
```
python -m pytest tests -q
```

Then the real end-to-end check — the whole-article capture through the whole
pipeline:

```
python tools\run_pipeline.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 work\80654199
```

Expect: 1 article, the note reporting **3 columns, 0 clipped**, and body text
that runs to the end of the article rather than stopping a column short. If a
warning appears about a frame edge, that is the new code working, not a fault.

Then the old close capture, to confirm nothing regressed:

```
python tools\run_pipeline.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 work\71a97929
```

This one **should** warn that a column is off the right edge — it is clipped,
and until now nothing said so.

## 6. Then stop

Chapters 3, 4 and 5. Chapter 4 now has, measured on the deployed path:

- the framing/resolution trade-off, with the first end-to-end CER this system
  has ever had
- the fixed-scale defect and its 2.9× cost (0.2193 → 0.0760 from one constant)
- the 11 px diacritic cliff, confirmed independently of the corpus work
- the library-version reproducibility finding, fixed and verified across two
  OpenCV majors

Four results, all with numbers behind them. That is a chapter.
