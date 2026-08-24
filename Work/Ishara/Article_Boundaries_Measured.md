# Article boundaries — what the frames actually show

**R26-DS-002 · IT22259134 · 22 August 2026 — revision 2**
**Supersedes §3 of `Article_Boundaries_Design.md`. Two claims in that document
are now known to be wrong; they are named in §3 below.**
**Revision 2** replaces the deskew estimator after your test failure exposed a
library-version bug (§2a), and adds a second, independent clipping test that
changes the headline count from 6/8 frames to **8/8** (§4).

You ran the tool. The output was nonsense, and it was my nonsense, not your
data's. I have since staged your nine captures and six corpus pages, found the
bug, rewritten the module against real frames, and looked at the overlays.

The headline result reverses the design I gave you: **your captures are not cut
off vertically. They are cut off sideways.** Every one of the eight usable
frames loses a column off the left or right edge — one scene loses one off
*both* — while the article body sits comfortably inside the frame from top to
bottom.

---

## 1. Why the first run produced garbage

The first version built columns and pitch from `closeup.text_lines()` boxes.
On real frames that failed three ways at once.

| Symptom in your output | Cause |
|---|---|
| columns 3, 3, 3, **1**, 3, **1**, 2, 2, 2 across nine frames of **three static scenes** | one merged line box 1811 px wide bridged a gutter and collapsed three columns into one band |
| pitch 51.5, **27**, **15**, **10** px | with columns merged, sorting boxes by `y` interleaves lines from different columns, so median top-to-top spacing halves or thirds |
| inter-line gap median **−0.75 pitch** | a negative median gap is not a page property. It is proof the boxes were not one column's lines |
| `0/9 frames have text running off an edge` | `current_block` picked a small fragment mid-frame, which then has huge margins and looks closed |

Any constant chosen from that run would have been chosen from noise. The one
line in the output that should have stopped me was the negative median gap, and
I should have caught it before you ran anything.

---

## 2. What replaced it

Projections of a **filtered ink mask**, not line boxes.

- **Deskew first.** 1° drags a column sideways by 57 px over 3264 px — about
  one gutter width. Measured skew on your captures: **−0.70° to +0.25°**, with
  0.0–0.1° residual after correction. See §2a: the first estimator was wrong
  on both our machines, in opposite directions.
- **Keep only body-text-sized components.** Height between `GLYPH_H_MIN` and
  `GLYPH_H_MAX` *and* under 2.5× the median. That one test removes the hand
  (one enormous component), the photographs, the page border, the speckle
  **and the headlines** — without naming any of them. Headlines have to go:
  a headline crosses every gutter it spans and would weld the page into one
  column.
- **Pitch by autocorrelation of the row profile**, not by box spacing. Same
  reasoning as your capture app's estimator.

The result is stable where the old one was not:

```
cv2 4.13.0   deskew: projection-profile search (version-independent)

frame                       p75   skew pitch cols blk  lines   top     bot    Link  Rink  open
burst_105855_g27_s3615_3     34  +0.25    52    3   1   28.2  21.79p  12.73p  0.00  0.41   R
burst_105855_g27_s3615_4     33  +0.20    52    3   1   28.3  21.71p  12.77p  0.00  0.39   R
burst_105855_g27_s3615_5     33  +0.10    52    3   1   28.3  21.58p  12.87p  0.00  0.28   R
burst_105901_g26_s3995_2     30  -0.70    48    3   1   29.7  26.90p  11.44p  0.00  0.32   R
burst_105901_g26_s3995_5     32  -0.15    48    2   1   28.2  25.23p  14.52p  0.00  0.40   R
burst_105910_g29_s856_2      36  +0.10    56    3   1   28.0  21.29p   9.02p  0.15  0.25   LR
burst_105910_g29_s856_4      35  -0.60    55    3   1   28.0  21.85p   9.51p  0.17  0.33   LR
burst_105910_g29_s856_5      34  -0.55    55    3   1   27.9  21.69p   9.71p  0.17  0.35   LR
```

Three columns on seven of eight frames, pitch constant to ±1 px within each
burst, and the clipped side identical across every frame of a scene — which is
what you would expect from three static scenes and did not get before.

**A side result worth keeping.** During development, before the close-up gate
was added, the badly blurred frame of burst 105901 (`glyph_p75` 11, 201
spurious line boxes) returned **pitch 48**, against **49 and 48** on the two
sharp frames of the same burst. That is your own blur-invariance finding
reproduced independently, on the server, on a captured frame rather than a
preview. Chapter 4 material.

---

## 2a. Your failing test was a real bug, and a bad one

`test_skew_is_measured_and_removed` failed on your machine with
`deskew_deg = 0.0` where mine returned −1.5. That is not a flaky test. Look at
the skew column of your tool run against mine — **same nine frames, same code**:

| | frames 1–3 | 105901_2 | 105910_2/4/5 |
|---|---|---|---|
| my OpenCV | 0.00 | −1.38 | −0.42, −0.83, −0.77 |
| your OpenCV | +0.90, +0.86, +0.75 | +1.20 | +0.79, +0.99, +1.00 |

**Opposite signs.** `cv2.minAreaRect` changed the range it reports its angle in
— it used to be [−90, 0), it became (0, 90] — so the `if w < h: a += 90` rule
picks a *different family of rectangles* on different builds. On one of our two
machines, deskew was rotating the wrong way and **doubling the tilt instead of
removing it**, and nothing failed. Your columns still came out right only
because the true skew is under 0.7°.

It is now replaced by a projection-profile search: rotate a downscaled
body-text mask through candidate angles and keep the one whose row profile is
sharpest. No ambiguous API, and the sign is defined by what the number is used
for. Measured: exact to 0.00° on synthetic tilts of ±1.5° and ±3.0°, and
0.0–0.1° residual tilt after correcting your nine captures. It costs about
0.2 s a frame, which is nothing against a 12 s round trip.

**Both of the earlier skew tables were wrong.** The true tilt on your captures
is −0.70° to +0.25°.

### The fix is verified, not just asserted

Same eight frames, run on two machines with different major versions of both
libraries:

| | OpenCV | numpy | skew, frames 1–8 |
|---|---|---|---|
| here | 4.13.0 | 2.4.4 | +0.25, +0.20, +0.10, −0.70, −0.15, +0.10, −0.60, −0.55 |
| Ishara | **4.9.0** | **1.26.4** | +0.25, +0.20, +0.10, −0.70, −0.15, +0.10, −0.60, −0.55 |

Identical to the printed precision, and so are pitch, column counts, block
counts, margins and edge-ink fractions. That is the whole point: the previous
estimator disagreed by more than a degree *and* by sign across the same two
machines. This one does not disagree at all.

Worth writing down as a method note, because it is the cheap half of the
lesson. The expensive half — the transformers CER — cannot be fixed this way,
only pinned and reported. Where an ambiguous API can be replaced with a
measurement of the thing you actually want, the version dependence goes away
entirely.

### This is the third one

EXIF orientation (OpenCV/PIL version). CER 0.0730 → 0.0615 (transformers
version). Now a skew angle whose *sign* flips with the OpenCV build. Three
independent instances, in one project, all found by accident.

That is no longer an anecdote about one library. It is a pattern worth stating
in Chapter 5 alongside the transformers finding — and it is a stronger version
of that contribution, because it shows the hazard is not confined to model
inference. `diagnose_article.py` now prints `cv2` and `numpy` versions in its
header for the same reason.

The test now runs **both tilt directions**. A one-sided test would not have
caught this, and mine was one-sided.

---

## 3. Two things I told you that are wrong

**Wrong claim 1: "the frame is cut off vertically — a big article does not
fit."** Measured: the body block sits **21.4–26.2 pitch below the top** of the
frame and **9.1–14.5 pitch above the bottom**. Nothing is vertically cut. The
block also stops cleanly *above* the next article's headline, which means the
gap test is already excluding the neighbouring story.

**Wrong claim 2: "column lock — read the column at frame centre, drop the
rest."** Look at the overlay. **The three columns are the same article.** The
headline spans all of them. Locking to the centre column would have read a
third of the story and stopped. That would have made the system worse, and it
would have looked like it was working.

The corrected rule, which is what the code now does:

> Keep **every whole column**. Drop only columns **clipped by a frame edge**.
> Crop vertically to the block containing the frame centre.

The blue box in the overlay is that crop. It contains the two whole body
columns and excludes, in one operation: the headline, the thumb, the page
margin, the neighbouring article's headline at the bottom, and the sliver of a
column running off the right edge.

---

## 4. The actual defect, and the cheapest fix

**8 of 8 frames lose a column off the side** — and one scene loses one off
*both* sides. That partial column is still
text, so Tesseract reads the first few characters of every line in it, and mT5
then confidently repairs those fragments into words nobody printed. The code
now excludes clipped columns and says so — but the article is still incomplete.

Two of those eight were missed until revision 2. The gutter test alone cannot
see a column cut at 75% of its width (`105901_g26_s3995_2`: right band 613 px
against a median of 613 — not narrow, but ink at the right edge in 32% of
rows), nor one merged into its neighbour by a 26 px gutter
(`105901_g26_s3995_5`). So there is now a second, independent test:
**is there body-text ink in the outermost glyph-width of the frame?** A page
has margins; if ink reaches the frame edge across a tenth of the rows, the page
did not end there, the frame did. Measured 0.00 on both edges of a synthetic
whole-column frame, 0.15–0.41 on all eight real ones.

Here is the useful part. Measured column widths are 795–920 px with gutters
79–125 px on a 2448 px frame. A three-column article therefore needs about
**2600–2900 px** of frame width at your habitual `glyph_p75` of 33–36. The
frame is 2448.

**Every capture you have is missing part of the article, by about 8–18% of
frame width.** Stepping
back until `glyph_p75 ≈ 30` would fit all three columns — and 30 is still above
the 25 capture gate *and* above the 28 close-up threshold. Nothing about the
resolution requirement has to be relaxed.

So the single highest-value change is not panning, not stitching, not YOLO:

> **Aim the guidance at the bottom of the READY band (`glyph_p75` ≈ 28–30)
> rather than letting the user settle at 33–36.**

Caveats, because this is one page: I do not know whether that article has three
columns or four — its headline is also cut off at the right, so it may be
wider still. And this is one article in one newspaper. §6 says how to check.

---

## 5. Corpus pages are refused, on purpose

The projection method returns **1 or 2 columns** on corpus half-pages that
have six or seven, and on `lankadeepa_p20` it returned **pitch 12 px against a
true 30**. Those are not near misses.

The reason is structural, not a tuning problem: a corpus "half page" is a whole
newspaper page carrying seven or eight stories, with photographs and headlines
crossing columns at every height, so the vertical projection never reaches
zero anywhere.

`analyse()` now **refuses** those frames instead of answering. Full-page layout
is what your YOLO detector is for, at the framing it was trained on.
`glyph_p75` separates the two cases cleanly on this sample — corpus 18–24,
phone captures 30–36 — against the existing `CLOSEUP_MIN_P75 = 28`. No new
constant was needed.

This is the §4 "survey then read" split from the design document, arriving on
its own from the measurement: **the two framings need two different methods,
and each one is bad at the other's job.**

---

## 6. One more thing to check, and it may matter

Measured on your captures: **pitch / `glyph_p75` = 1.57–1.63.**
The capture app assumes **`PITCH_PER_GLYPH = 1.80`** to convert a measured
pitch into a glyph estimate.

If that 1.80 is high, the app reports a glyph larger than the truth, tells you
READY sooner than it should, and you stand closer than you think — which is
exactly the condition producing the lateral clipping in §4.

**Do not treat this as a result yet.** The two are not measured the same way:
the app works on a 640×480 centre crop of the 1440×1080 preview and converts by
2.267; this is measured directly on the 2448×3264 captured frame. A difference
is expected. Whether it is *this* difference is checkable, and if 1.80 should
be nearer 1.60 it is a one-constant fix that improves every capture.

---

## 7. Files, and what to run

Rewritten and in place:

```
system/layers/l3_segment/layout.py      projection method, refuses full pages
system/tests/test_layout.py             29 tests, all passing
system/tools/diagnose_article.py        rewritten output
```

```
cd E:\RP\R26-DS-002\system
python -m pytest tests/test_layout.py -q
python tools/diagnose_article.py --root E:\RP\R26-DS-002 --render tools\out\layout --csv tools\out\layout.csv "F:\App\backend\inbox"
```

**Then open one overlay and look at it.** Red column = clipped by the frame
edge. Blue box = what goes to Tesseract. I have checked
`burst_105855_g27_s3615_3` and it is right; I have not checked a page you
capture next.

### The measurement that decides §4

Capture the **same article** three times, deliberately, at three distances —
one where the app first says READY, one at your habitual framing, one a step
further back — and run the tool on all three. That tells you, on a real
article, whether backing off to `glyph_p75` ≈ 30 fits the whole width. It takes
ten minutes and it decides whether the fix is a guidance constant or a
fortnight of panning and stitching.

Do that before anything else in this document.

---

## 8. Priority, unchanged

Chapters 3, 4 and 5. None of this is Component 2, and the deadline has not
moved. What has changed is that the cheapest fix now looks like **one guidance
constant**, not a new capture mode — so the honest ranking is:

1. The three-distance measurement above. Ten minutes.
2. If it confirms §4: change the guidance target, re-capture, done.
3. Warn the user about clipped columns — `warnings_for()` already returns the
   sentences; wiring them into the server response is an hour.
4. Everything else — panning, joining, the survey frame — only after the
   chapters are drafted.
