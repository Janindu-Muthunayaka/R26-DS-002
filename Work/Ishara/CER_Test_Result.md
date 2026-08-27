# The accuracy test — result

**R26-DS-002 · IT22259134 · 24 August 2026 — revision 2**
**Supersedes the 23 August version, which was inconclusive because the wide
capture was out of focus.**

`80654199` is a sharp wide capture: `glyph_p75` 22, whole article (edge ink 0.00
on both sides), sharpness 5414 against 4795 for the close reference. It is the
frame the experiment has been waiting for.

Same 20 lines of the same column in all three, psm 6, mT5 through
`BodyReader.correct_lines()`, CER from `verify_model.py`.

| capture | framing | scale | →px | OCR CER | **mT5 CER** | mT5 WER |
|---|---|---|---|---|---|---|
| 71a97929 | `p75` 38, **clipped** | fixed 0.40 | 15.2 | 0.1023 | **0.0570** | 0.1875 |
| 71a97929 | | auto 0.35 | 13.2 | 0.1082 | 0.0731 | 0.1429 |
| e6e14f1b | `p75` 28, **clipped** | fixed 0.40 | 11.2 | 0.0921 | 0.0556 | 0.1875 |
| e6e14f1b | | auto 0.47 | 13.2 | 0.0877 | **0.0439** | 0.1696 |
| 80654199 | `p75` 22, **WHOLE** | fixed 0.40 | 8.8 | 0.1901 | 0.2193 | 0.4018 |
| 80654199 | | auto 0.60 | 13.2 | 0.1096 | **0.0760** | 0.2232 |

---

## 1. The answer: the whole article can be read

Best per capture: **0.0570** at the close framing, **0.0760** at the whole-article
framing.

That is **+0.019 CER** — about **13 extra character errors in 684 characters** —
in exchange for the third of the article the close framing physically cannot
see.

**Take the trade.** A missing column is not 13 errors, it is a hundred and forty
words of the story that never reach the listener, with nothing in the audio to
say they are missing. Two percentage points of character error is a far smaller
harm than an article that stops mid-sentence.

One number to be careful with. 0.0760 sits almost exactly on your canonical
0.0757. **That is a coincidence and must not be reported as a reproduction** —
different data (20 lines of one column of one phone capture), different
protocol, n=1. Say "of the same order as the locked-test-set figure" if you say
anything at all.

## 2. Adaptive scaling is not optional — it is the largest effect measured

At `glyph_p75` 22 the shipped fixed `CLOSEUP_OCR_SCALE = 0.40` gives Tesseract
an **8.8 px** glyph, below the ~11 px where diacritics die. Result: mT5 CER
**0.2193**. Scaling to a target glyph height instead gives 13.2 px and CER
**0.0760**.

**2.9× better, from one constant.**

This is now measured on a sharp frame, so the caveat from the 23 August version
is lifted. And it explains the whole earlier mess: the fixed 0.40 was punishing
every wide framing for a reason that had nothing to do with framing.

It helps at the middle distance too — `p75` 28: 0.0556 fixed → **0.0439** auto,
the best number in the table.

## 3. But 13.2 px is probably NOT the optimum

Every within-capture comparison favours the **larger** effective glyph:

| capture | smaller | larger | winner |
|---|---|---|---|
| 71a97929 | 13.2 px → 0.0731 | **15.2 px → 0.0570** | larger |
| e6e14f1b | 11.2 px → 0.0556 | **13.2 px → 0.0439** | larger |
| 80654199 | 8.8 px → 0.2193 | **13.2 px → 0.0760** | larger |

Three for three. And 13.2 is not measured — it is `0.40 × 33`, inherited from a
by-eye choice made on one frame. The best point may be 15, 17, or native scale;
nothing above 15.2 px has ever been tried on this path.

Your research does show downscaling beats native (CER 0.2205 native against
0.1754 at 0.40×) — but that was measured on **corpus region crops at a different
glyph size**, so it does not fix the target for this path. It only says the
curve has a minimum somewhere.

**This is the cheapest remaining experiment in the project.** One command.

---

## 4. What to do, in order

### Step 1 — sweep the target glyph height (2 minutes)

`--targets` is new in `compare_framing.py`.

```
cd E:\RP\R26-DS-002\system
```
```
python tools\compare_framing.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --gt ..\Work\Ishara\article_truth.txt --lines 20 --targets 11,13.2,15,17,19,native --save-text tools\out\sweep work\80654199 work\71a97929
```

Six scales × two captures. The `mT5 CER` column then has a minimum in it, and
that minimum is the constant. **Send me the table.**

### Step 2 — three changes to the deployed path (I write them, ~1 hour)

Only after step 1 fixes the number:

1. `CLOSEUP_OCR_SCALE` (a fixed factor) → `CLOSEUP_TARGET_GLYPH` (a target
   height), and `body.read_page()` scales to it. This is what
   `rescale_to_optimum()` already does on the region path; the close-up path
   never got it.
2. `CLOSEUP_MIN_P75` from 28 → 20, so a whole-article framing is not refused.
   `80654199` at `p75` 22 analysed correctly — 3 columns, pitch 32, right crop.
3. Re-aim the capture app's READY band from 33–36 down to about 22–26, so the
   user is guided to the framing that holds the whole article.

### Step 3 — verify end to end

Capture with the re-aimed guidance, run the full pipeline, confirm the spoken
output covers the whole article and warns when it does not.

### Step 4 — Chapters 3, 4 and 5

Which is where this all has to stop. You now have, for Chapter 4, three results
that did not exist a week ago:

- the framing/resolution trade-off, **measured**, with a real end-to-end CER on
  the deployed path — which previously had none
- the fixed-scale defect and its 2.9× cost
- the library-version reproducibility finding, with a fix verified across two
  OpenCV majors

That is enough. Write it up.
