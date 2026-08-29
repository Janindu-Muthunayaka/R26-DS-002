# Reading one whole article, and knowing when you haven't

**R26-DS-002 · Component 2 / integration · IT22259134 · 22 August 2026**

You asked four things: how the system decides what an article is, whether
reading one article at a time is the right goal, why a big article gets read
only halfway, and what to do about it.

Short version: reading **one complete article** is the right goal, the system
currently has no idea what an article is, a long article **cannot** fit in one
frame at the glyph height your own capture gate demands — that is geometry, not
a bug — and the thing actually worth fixing is that the system does not *notice*
when it has read a fragment.

---

## 1. Three different problems get called "article detection"

Separating them matters, because two are physical facts you cannot remove and
one is a defect you can fix this week.

**A · Lateral bleed.** `text_bbox()` crops to the bounding box of every text
line in the frame. If a neighbouring column is in shot, it is inside the crop,
and Tesseract reads it. Sentences from two unrelated stories come out as one
piece of speech.

**B · Vertical truncation.** A long article does not fit in the frame at close
range. You capture the middle of it. There is no beginning and no end.

**C · No boundary awareness.** Neither A nor B is detected. The system speaks a
fragment with exactly the same confidence it would speak a whole story.

**C is the actual defect.** A blind listener has no way to tell that what they
just heard was half of one article welded to a third of another. If the system
said *"this continues below"*, the same imperfect capture becomes usable. That
is why the work below starts with detection, not with cropping.

---

## 2. Why a big article cannot fit — the arithmetic

Derived, not measured. The inputs are measured; the multiplication is mine.

| Input | Value | Source |
|---|---|---|
| captured frame | ~2448 × 3264 px | inferred from the YOLO box `(0,2692)-(2172,3264)` on `burst_20260820_105855` |
| your captures | `glyph_p75` = 33 | measured, nine captures |
| capture gate | `glyph_p75` ≥ 25 | measured, 168 corpus rows |
| pitch per glyph | 1.80 | `Guidance.kt`, measured |

pitch at glyph 33 ≈ 59 px → about **55 line-heights** down the frame.
pitch at glyph 25 ≈ 45 px → about **73**.

So backing off from your habitual framing to the edge of the pass mark buys
**1.32× more lines** and costs resolution. It does not make a long article fit.
And you cannot buy it back by upscaling afterwards — that is measured and it is
brutal: 2× → CER 0.336, 3× → 0.659, against 0.076 at the optimum.

**Therefore: no single frame will hold a long Sinhala newspaper article at a
glyph height the OCR can read.** The only two honest responses are to read the
article in parts, or to accept measurably worse text. There is no third option
and no clever crop that creates one.

One caveat I want on the record. Those nine captures detected **53–73 text
lines each, across all columns**, which does not sit comfortably next to a
derived 55 lines *per column*. Either the text does not fill the frame, or
lines are merging, or my inferred frame size is wrong. I am not going to
explain that away — §5 measures it.

---

## 3. The design: point, then pan

Four parts. Two are written and tested; two need the numbers from §5 first.

**1 · Column lock** *(written)*
Find the vertical gutters, take the column the **frame centre** falls in, drop
the rest — and report how many were dropped rather than discarding them
silently. The centre of the frame is the only statement of intent the user
makes; aiming *is* the selection.

Gutters are found from the **detected line boxes**, not from an ink projection
of the frame. That is deliberate: `text_lines()` has already thrown away the
hand, the margin and the speckle, so a thumb lying in the gutter cannot bridge
two columns. An ink projection would merge them exactly when a hand is in shot,
which on this path is most of the time. There is a test for it.

**2 · Cut-off detection** *(written)*
White space above the first line and below the last is **positive evidence**
that the article really begins and ends inside the frame. If the text runs to
the edge with no gap, it was cut. Reported as `top_open` / `bottom_open`, with
the margins in units of line pitch so the threshold is device-independent.

This is the direct answer to *"it only captures half of the article"*. It does
not fix it. It makes it visible, which is the prerequisite.

**3 · Guided panning** *(needs §5)*
Bottom edge open → the phone says *"there is more below — tilt down slowly"*,
guidance re-arms, the auto-shutter fires again. Repeat until a frame comes back
with a closed bottom edge. The user does exactly what a sighted reader does:
moves down the column.

**4 · Overlap joining** *(written)*
Consecutive frames overlap on purpose. Match the **trailing lines of one frame
against the leading lines of the next, as text**, and drop the duplicate.

Text, not pixels. Image stitching would have to survive changes of scale,
angle and exposure between frames — the exact conditions your capture app
creates — while the OCR output of the same physical line is nearly identical
across them. Two matching lines minimum; one is not evidence, because Sinhala
newsprint repeats short lines constantly. **A seam with no overlap returns 0
and must be reported to the user, not joined anyway.** Silently concatenating
two frames that are not consecutive is how you would manufacture a sentence
that was never printed.

### What this design does *not* solve

I would rather you knew these now than at the viva.

**Multi-column articles.** In a newspaper one story commonly spans two or three
columns. From a close-up frame you usually **cannot tell** whether the next
column is the same story — the evidence that would settle it (a headline
spanning both, a rule line between them, a change of leading) is outside the
frame. Column lock will read one column of a three-column story and stop
cleanly, which is honest but incomplete. §4 is the real answer to this.

**A block gap is not an article boundary.** A wide white gap separates blocks
of text. It cannot distinguish "next article" from "sub-heading" — the geometry
is identical. It narrows the crop; it does not identify the story.

**A hand *on* the text still destroys line detection.** Not new, not fixed
here, and now pinned by a test so a future change to `text_lines()` is noticed.

---

## 4. The version that would be a research contribution: survey, then read

This is where your YOLO detector comes back — and it resolves the conflict I
flagged in the last document instead of just admitting it.

The conflict was: the detector needs loose framing (trained on full and half
pages, corpus `glyph_p75` median 19–22), the OCR needs tight framing (`p75` ≥
25, your captures 33), and the two ranges barely overlap. I called that a real
deployment trade-off. It is — but only if both constraints must be met *by the
same frame*. **Separate them in time and the conflict disappears.**

```
1. SURVEY FRAME   held at arm's length, p75 ~20 — the detector's own range
   YOLO segments the page, takes the article containing the frame centre,
   and reports its EXTENT and roughly how many screens it will take.
                       │
2. READ FRAMES    held close, p75 ~30, panned down under voice guidance,
   joined by text overlap                     (§3)
                       │
3. COMPLETION     the survey frame's extent is the STOPPING CONDITION —
   the system can say "that is the end of the article" instead of
   "I have run out of frames"
```

Two things make this cheaper than it looks. It needs **no image registration** —
only the article's extent as a fraction of the page and an approximate line
count, used as a stopping condition, not as coordinates. And it puts your
detector, which is well evaluated (mAP50 0.96, 95.6% correct grouping, zero
over-merge) and currently **unused in the deployed path**, back into the system
doing the job it is good at, at the framing it was trained for.

It also gives Chapter 5 something better than an apology: a stated design
principle — *segmentation and recognition have incompatible resolution
requirements and should be satisfied by different frames* — that falls out of
two of your own measurements and would not be predicted from either alone.

Cost, honestly: two capture modes, guidance for both, and a multi-frame session
on the server. That is a fortnight of engineering, not an afternoon. See §6.

---

## 5. Step 1 — measure, before anything is wired in

Three files are attached. Nothing in them touches the pipeline yet.

```
system/layers/l3_segment/layout.py     columns, blocks, cut-off, joining
system/tests/test_layout.py            19 tests, all passing
system/tools/diagnose_article.py       the measurement
```

Install and check:

```
cd E:\RP\R26-DS-002\system
python -m pytest tests/test_layout.py -q
```

Then measure. Three sets, and the third one matters most:

```
python tools/diagnose_article.py --root E:\RP\R26-DS-002 --render tools\out\layout_phone --csv tools\out\layout_phone.csv "F:\App\backend\inbox"

python tools/diagnose_article.py --root E:\RP\R26-DS-002 --render tools\out\layout_corpus --csv tools\out\layout_corpus.csv <20 corpus pages>
```

**Third set — capture it fresh.** Pick one **long** article, five or six frames
panning down it, plus two frames of a **short** article that genuinely fits.
That is the only data that tells us whether a closed edge and an open edge are
actually distinguishable on real pages.

The tool prints the two distributions the constants must come from:

- **`BLOCK_GAP_PITCH`** — from the inter-line gap histogram. It has to sit above
  the paragraph bulk and below the boundary tail. **If there is no gap between
  the two, say so and stop** — that would mean white space alone cannot
  separate blocks on Sinhala newsprint, which is a finding, not a failure.
- **`EDGE_OPEN_PITCH`** — from the top and bottom margins. A frame holding a
  whole article must have **both** margins well above it. If every frame is
  under 1.0p, then every capture you have ever taken was cut off, and the
  answer is more frames rather than a better threshold.

`--render` writes an overlay per frame — columns in grey, the centre column in
green, the block being read in blue, a red bar on any edge the text runs off.
**Look at those before you believe a single number in the table.** Both
percentile figures are useless if the column lock is picking the wrong column,
and that is something you can see in two seconds and I cannot see at all.

Send me the tables and I will write step 2 from measured values.

**Do not let me pick these two constants by eye.** That is exactly how
`CLOSEUP_OCR_SCALE = 0.40` got into the system, and it is still the only
unmeasured constant in the read path.

---

## 6. Priority — and this is the part you will not like

Chapters 3, 4 and 5 are still unwritten and October is six weeks away. Nothing
in this document changes that, and none of it is Component 2. Your research
contribution is the post-OCR corrector and the negative result about gating;
both are finished. All of the above is Component 1 and integration engineering.

So, in order:

**Do now — half a day.** §5. Run the measurement, look at the overlays, send me
the numbers. It is cheap, it needs no models, and it produces a figure you can
put in Chapter 4 regardless of what gets built afterwards.

**Do next — two days, and it is worth it.** Cut-off detection wired into the
server as a warning, and the phone saying *"this article continues below"*.
This does not require panning, joining, or any new capture mode. It converts
the current silent failure into something the user can act on, and it is the
difference between a demo that reads half a story confidently and one that
tells the truth about what it saw.

**Do only if the chapters are drafted — a fortnight.** Guided panning and
overlap joining (§3, parts 3–4), then the survey frame (§4).

**Do not do.** Anything that lowers the glyph gate to make articles fit. The
resolution cost is measured and it is severe.

To answer your question directly: yes, one article read completely is the right
target, and yes, mixing two articles is worse than reading a fragment of one —
because the listener can tell they missed something, but they cannot tell that
two stories were spliced. Getting the system to *know which of those happened*
is worth two days. Getting it to always capture whole articles is worth a
fortnight, and only after the thesis is written.
