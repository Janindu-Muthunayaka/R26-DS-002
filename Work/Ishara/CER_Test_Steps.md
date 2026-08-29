# The accuracy test — exact steps

**R26-DS-002 · IT22259134 · 23 August 2026**

**The question, and nothing else:** the whole article fits at `glyph_p75` 25.
Is the OCR at 25 as accurate as at the 38 you normally shoot at?

If yes → lower `CLOSEUP_MIN_P75` to 25, aim the guidance at 25–28, and the
system reads whole articles. If no → the conflict is real and *measured*, you
report it, and panning waits until the chapters are done.

Either way you end up with the **first end-to-end CER on the deployed path**,
which Chapter 4 currently has none of.

Budget: about 20 minutes of typing, plus two commands.

---

## Why one column, not the whole article

Scoring the whole crops would compare two different amounts of text — the close
framing physically lacks a column and a half, so it would lose on coverage no
matter how sharp it is, and the number would say nothing about resolution.

Coverage is already answered, by the layout tool, without any CER.

So both frames are cropped to the **same single column** and scored against a
transcription of that one column. Same words, different pixel density. That is
the actual question, and it is twenty minutes of typing instead of two hours.

---

## Step 1 — check the crops (1 minute)

```
cd E:\RP\R26-DS-002\system
```
```
python tools\compare_framing.py --dry-run --save-text tools\out\cer work\71a97929 work\14f7798c
```

Expect roughly:

```
reading column #1 of each capture
capture        p75 cols  clip         crop  fixed  ->px   auto  ->px
71a97929      38.0    3     R    1062x1541   0.40  15.2   0.35  13.2
14f7798c      27.0    4     -      566x910   0.40  10.8   0.49  13.2
```

**Open `tools\out\cer\71a97929_col1.jpg` and `14f7798c_col1.jpg` and check they
show the same column of the same article.** I checked these two: both begin
`කුරුණෑගල – චාන්දනී දිසානායක`. If yours differ, nothing after this is
meaningful.

Note the `->px` column. That is the glyph height Tesseract actually sees.
`fixed` gives the wide framing **10.8 px**, under the ~11 px where diacritics
die in your own measurements — which is why every frame is read at both scales.
Comparing only `fixed` would have condemned the wide framing for a reason that
has nothing to do with framing.

---

## Step 2 — get a starting transcription (2 minutes)

```
python tools\compare_framing.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --save-text tools\out\cer work\71a97929
```

No `--gt`, so it scores nothing — it just reads the sharp capture and writes
the text. Then open:

```
tools\out\cer\71a97929_auto_corrected.txt
```

---

## Step 3 — correct it against the newspaper (about 20 minutes)

Copy that file to `Work\Ishara\article_truth.txt` and fix it against the
**printed page**, not against what looks plausible.

- Only that **one column** — the one in `71a97929_col1.jpg`.
- Include the byline line at the top of the column if it is in the crop.
- Do not add the headline, and do not add anything from the other columns.
- Line breaks do not matter; all whitespace is collapsed before scoring.
- **Save as UTF-8.** In Notepad: *Save As → Encoding: UTF-8*. VS Code is safer.

**State this limitation in Chapter 4 when you report the number:** the ground
truth was produced by correcting an OCR output rather than typed from nothing,
which biases it slightly toward the OCR. Correcting against the printed page
character by character is what keeps that bias small. It is the standard way
OCR ground truth is built, and it is standard to say so.

---

## Step 4 — run the comparison (about 5 minutes)

```
python tools\compare_framing.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --gt Work\Ishara\article_truth.txt --save-text tools\out\cer work\71a97929 work\e6e14f1b work\14f7798c
```

Three captures: `p75` 38 (your normal framing), 29, and 25 (whole article).
Each read at both scales. Twelve numbers.

---

## Step 5 — read it

Compare the **best row of each capture**, not `fixed` against `fixed`. The
question is what each framing can achieve, not what today's constant does to
it.

| what you see | what it means |
|---|---|
| best mT5 CER at `p75` 25 close to that at 38 | the whole article can be read at no real cost — lower `CLOSEUP_MIN_P75` to 25 and re-aim the guidance |
| `p75` 25 clearly worse at **both** scales | the conflict is real and measured; guidance stays, panning waits for the chapters |
| `auto` much better than `fixed` anywhere | the fixed 0.40 is costing accuracy on its own, independently of all this |

A CER above about 0.5 usually means the crop was wrong or the transcription
does not match the column that was read. Read the `.txt` files in
`tools\out\cer` before believing any such number.

Send me the table.

---

## What has NOT changed

Nothing in `core/` or `layers/l4b_body/`. `CLOSEUP_OCR_SCALE` is still 0.40 and
the pipeline still uses it. The experiment computes its own scales inside
`compare_framing.py`, deliberately, so that running it cannot alter the system
it is measuring. The mT5 call is `BodyReader.correct_lines()` and the CER
function is imported from `verify_model.py` — both reused rather than
reimplemented, so the numbers are comparable to the research harness by
construction.

Changes to the deployed path come after the measurement, not before.
