# Article-wise reading — diagnosis and fix

**27 August 2026.** You reported that the system reads text that does not
belong to the article. It did, and it was not one bug but three. All three are
measured on **the 70 real phone captures in `system/work`**, not on a sample.

---

## 0. Something I broke and cleaned up

My verification runs earlier today POSTed 60x40 px test images to `/capture`,
which created **560 junk job folders in `system/work`**. They are moved to
`work/_my_test_jobs/` — delete that folder. Every number below is measured on
the 70 genuine phone captures only (>100 KB, 2448x3264).

---

## 1. What was actually happening

| | share of real captures | what it read |
|---|---|---|
| layout path → column/block crop | 48 (69%) | the article… mostly |
| **NOT close-up → YOLO** | **20 (29%)** | whatever the detector returned |
| layout refused → whole-text bbox | 2 (3%) | **every text line in the frame** |

And of the 48 that got the "good" path, **13 had a headline sitting INSIDE
the crop** — a crop spanning an article boundary.

So roughly half of all captures were reading something other than one article.

### Cause 1 — the block finder never identified an article

`row_profile()` sums ink **across all column bands**. A white gap in column 1
is filled by text in column 2, so the profile can never see a horizontal
article boundary. `blocks()` is honest about the consequence in its own
docstring:

> It does NOT distinguish "next article" from "sub-heading" — the geometry is
> the same. **This narrows the crop; it does not identify a story.**

That is the whole bug in one sentence, written by us, a week before you
noticed the symptom.

### Cause 2 — 29% of captures never reached that path at all

The gate was `glyph_p75 >= CLOSEUP_MIN_P75` (20). Real captures have a median
p75 of **22** with the lower quartile **below 20**, so nearly a third fell
through to the YOLO article detector — the path
`Corrections_Register.md` entry 1 records returning *one confident box over
the NEIGHBOURING article's headline*, with the whole-frame fallback never
firing because a box **was** returned.

### Cause 3 — the fallback reads everything

When layout refuses on a close frame, the crop becomes
`closeup.text_bbox()` — the bounding box of **every text line in the frame**.
That is literally "read whatever text is visible". 3% of captures.

---

## 2. The fixes, and what they measured

### Fix 1 · a headline is a hard article boundary

`layout.split_at_headlines()`. A headline band crossing the columns being read
splits its block: what is above belongs to the previous story, and the
headline plus what is below belongs to the next one. A headline off to one
side is ignored — it heads a story in another column.

This works because the headline threshold is now measured
(`TITLE_MIN_LINE_RATIO = 3.0`; body lines reach 1.70x, headlines start at
5.91x, nothing between).

### Fix 2 · try layout FIRST, YOLO only if layout refuses

The p75 gate was refusing frames the analysis handles: re-measured with the
gate lowered, **layout succeeds on 16 of the 20** frames that were going to
YOLO.

**Is that safe?** The gate exists to keep whole newspaper pages out of a path
that assumes one story in full-height columns — but it is not the gate doing
that work. With the p75 gate turned **off entirely**, **12 of 12 corpus full
pages are still refused** by the gutter gate. The structural gates decide, as
they should. `LAYOUT_MIN_P75 = 12`.

### Fix 3 · the fallback now says what it did

The whole-text bbox path warns *"could not find the article boundaries in this
frame; read the text that was visible"*, which the phone speaks.

### Result

| | before | after |
|---|---|---|
| proper article crop | 48 (69%) | **57 (81%)** |
| falls through to YOLO | 20 (29%) | **11 (16%)** |
| whole-text bbox (now warned) | 2 (3%) | 2 (3%) |
| **crops spanning an article boundary** | **13 of 48** | **0 of 57** |
| headline splits applied | — | 68 |

Reproduce: the scan is `tools/measure_headline.py` for the constants; the path
census is in this document's method — re-run it against `system/work`.

Tests: **242**.

---

## 3. Your YOLO model — SETTLED, 27 Aug 2026

`tools/probe_yolo.py`, run on Windows over all 70 real captures, comparing the
detector's most confident box against the article the layout path chose:

| verdict | frames | |
|---|---|---|
| **DISAGREE — a different story** | **35** | **50% of all, 69% of comparable** |
| partial | 5 | 10% of comparable |
| agree | 11 | 22% of comparable |
| detector returned nothing | 8 | |
| layout refused, nothing to compare | 11 | |

51 frames had an answer from both. **The detector picked a different story on
69% of them.**

`Corrections_Register.md` entry 1 recorded this on ONE frame and was treated
as provisional. It is now measured on seventy and it holds.

### Which side is wrong?

Disagreement alone does not say. But the two sides are not equally evidenced:

- the layout crop: **0 of 57 crops span an article boundary** after the
  headline split, and two were confirmed by eye against the photograph
- the detector: one confident box, and a 69% disagreement rate

The detector was trained on **full and half page** framings. The phone shoots
far closer. This is the trade-off entry 1 predicted, now quantified.

### What changed because of it

**The phone path no longer runs the detector.** `SEGMENT_MODE = 'off'`.

When layout cannot identify an article, the system says
*"ලිපිය හඳුනාගත නොහැකි විය. ටිකක් ළං වී නැවත උත්සාහ කරන්න."* — could not
identify the article, move a little closer — and reads **nothing**.

Two reasons, and the first is the one that matters:

1. **A wrong article read confidently to someone who cannot check it is worse
   than no article.** The same reasoning that makes a wrong headline worse
   than none.
2. "Move closer" is the instruction that actually fixes the frame, so the next
   capture is correct rather than this one being wrong.

**The detector is not deleted.** It is Component 1's contribution, it is
evaluated on the framings it was trained for, and `SINHALA_SEGMENT_MODE=yolo`
restores the old behaviour for comparison. `tests/test_api.py` asserts it does
not come back silently.

### For Chapter 4

This is a result, not a workaround: *an article detector trained on full-page
framings disagrees with column-projection segmentation on 69% of close-range
captures, and the deployed system therefore declines to segment rather than
segment wrongly.* It is the same shape as the negative result — a component
measured, found unsuitable for a specific condition, and reported.

---

## 3b. How the probe was run

You asked whether the article segmentation model can be used. **I could not
test it**: `ultralytics` is blocked by the egress proxy in the environment
this was written in, so the only evidence about YOLO at close range remains
`Corrections_Register.md` entry 1 — **one frame**. That is not enough to
retire a trained model, and not enough to trust it either.

`tools/probe_yolo.py` settles it. Run it on Windows:

```
cd E:\RP\R26-DS-002\system
python tools\probe_yolo.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

For every real capture it reports whether YOLO's box **agrees** with the
article the layout path chose, per frame, with confidences.

**What to do with the answer:**

- **Agrees on most of the 16%** → keep it as the fallback, and consider using
  it to confirm the layout crop on the other 81%.
- **Disagrees** → the fallback should not read a confident wrong story. It
  should say *"move a little closer"*, which is one line in
  `app/pipeline.py` and is also the instruction that actually fixes the frame.

Either way it becomes a measured decision instead of an inherited assumption,
and either way it is a paragraph in Chapter 4.

---

## 4. Still true, and worth stating

- **11 captures (16%) cannot be segmented** and now get "move a little
  closer" instead of a read. That is a real cost — one capture in six asks the
  user to try again — and it is the honest one. Better guidance, not a better
  guess, is what reduces it.
- The layout path assumes **one story in full-height columns**. Two stories
  side by side in one frame, each with its own columns and no headline between
  them in view, would still merge. No capture in the 70 shows that; it is a
  gap, not a measured failure.
- Coloured (red) headline words are still lost to the grayscale threshold.
- These measurements ran under **cv2 5.0.0**, not the pinned 4.9.0. The
  functions used are version-stable — unlike `minAreaRect`, which is why
  `deskew_angle` uses a projection profile — but re-run on the pinned
  environment before any number goes into a chapter.
