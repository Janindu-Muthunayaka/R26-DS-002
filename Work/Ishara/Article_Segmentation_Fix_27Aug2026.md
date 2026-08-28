# Article-wise reading — diagnosis, fix, and the YOLO verdict

**27 August 2026.** Ishara reported the system reading text not belonging to the article. It was three bugs, all measured on the **70 real phone captures in `system/work`**. Full copy at `E:\RP\R26-DS-002\Work\Ishara\Article_Segmentation_Fix_27Aug2026.md`.

## The three causes

| path | share of real captures | what it read |
|---|---|---|
| layout → column crop | 48 (69%) | the article… mostly |
| **not close-up → YOLO** | **20 (29%)** | whatever the detector returned |
| layout refused → text bbox | 2 (3%) | **every text line in the frame** |

Of the 48 on the "good" path, **13 had a headline inside the crop** — two articles merged.

1. **`row_profile()` sums ink across all columns**, so a gap in column 1 is filled by text in column 2 and the profile can never see a horizontal article boundary. `blocks()`' own docstring already said it: *"This narrows the crop; it does not identify a story."*
2. **The gate was `glyph_p75 >= 20`**, but real captures have median 22 with the lower quartile below 20 — so 29% fell through to YOLO.
3. **The fallback** was `text_bbox()` — literally the bounding box of all visible text.

## The fixes

- **`layout.split_at_headlines()`** — a headline crossing the columns is a hard boundary; above it is the previous story, the headline and below is the next. Works because the headline threshold is now measured (3.0×; body lines reach 1.70×, headlines start at 5.91×).
- **Layout tried first, YOLO only on refusal.** Layout succeeds on 16 of the 20 frames that were going to YOLO. Verified safe: with the p75 gate off entirely, **12 of 12 corpus full pages are still refused** by the gutter gate. `LAYOUT_MIN_P75 = 12`.
- **The bbox fallback now warns** that it could not find article boundaries.

| | before | after |
|---|---|---|
| proper article crop | 48 (69%) | **57 (81%)** |
| falls through to YOLO | 20 (29%) | **11 (16%)** |
| **crops spanning two articles** | **13 of 48** | **0 of 57** |

## The YOLO verdict — settled

`tools/probe_yolo.py` over all 70 captures, comparing the detector's most confident box against the layout crop:

| verdict | frames |
|---|---|
| **DISAGREE — different story** | **35 (69% of comparable)** |
| partial | 5 (10%) |
| agree | 11 (22%) |
| returned nothing | 8 |
| layout refused, no comparison | 11 |

`Corrections_Register.md` entry 1 recorded this on **one** frame. It is now measured on seventy and it holds. The layout crop is the better-evidenced side: 0 of 57 span a boundary, two confirmed by eye; the detector has one confident box and a 69% disagreement rate. It was trained on full and half pages; the phone shoots far closer.

**Consequence: `SEGMENT_MODE = 'off'` — the phone path no longer runs the detector.** When layout cannot identify an article the system says *"ලිපිය හඳුනාගත නොහැකි විය. ටිකක් ළං වී නැවත උත්සාහ කරන්න."* and reads nothing. A wrong article read confidently to someone who cannot check it is worse than no article — the same reasoning that makes a wrong headline worse than none — and "move closer" is the instruction that actually fixes the frame.

The detector is not deleted: `SINHALA_SEGMENT_MODE=yolo` restores the old behaviour, and a test asserts it does not come back silently.

**For Chapter 4:** this is a result, not a workaround — *an article detector trained on full-page framings disagrees with column-projection segmentation on 69% of close-range captures, and the deployed system therefore declines to segment rather than segment wrongly.* Same shape as the negative result.

## Cost, stated

11 captures (16%) now get "move a little closer" instead of a read — one in six asks the user to try again. That is the honest cost. Better guidance, not a better guess, is what reduces it.

Also: my earlier verification runs created 560 junk job folders in `system/work` (60×40 test images). Moved to `work/_my_test_jobs/` — delete it.

Tests: **245**.
