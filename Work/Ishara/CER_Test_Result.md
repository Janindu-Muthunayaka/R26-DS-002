# The accuracy test — result, and why it does not answer the question yet

**R26-DS-002 · IT22259134 · 23 August 2026**

Your run produced numbers, and two of them are worth keeping. But the headline
question — *is OCR at `glyph_p75` 25 as good as at 38?* — **is not answered**,
because the one capture that holds the whole article is **out of focus**, and
two bugs in my harness were also distorting the table.

I reproduced all of it here (my Tesseract is a different build from yours, so
treat every number below as indicative and re-run for the numbers of record).

---

## 1. The capture that matters is blurred, not just wider

Sharpness measured **after** scaling every crop to the same effective glyph
height, so it measures focus rather than distance:

| capture | `glyph_p75` | sharpness at 13.2 px |
|---|---|---|
| 71a97929 | 38 | 4795 |
| e6e14f1b | 28 | 6398 |
| **14f7798c** | **26.5** | **635** |

**A factor of ten.** `14f7798c` is a soft frame. Its poor CER is focus, not
framing, and no conclusion about resolution can be drawn from it.

That is also why a 5% change in `glyph_p75` (28 → 26.5) appeared to triple the
error. A 5% resolution change cannot do that. Focus can.

The tool now measures this and refuses to let it pass quietly:

```
*** WARNING: focus differs by 10.1x at a matched glyph size
    (worst: 14f7798c at 635). CER would then be measuring FOCUS,
    not framing. Re-capture the soft one before believing anything below. ***
```

## 2. Two bugs of mine, both now fixed

**psm 3 on a single column.** I cropped to one column and then read it with
`TESS_CONFIG_PAGE` (psm 3, automatic page segmentation) instead of
`TESS_CONFIG` (psm 6, single uniform block) — which is exactly the distinction
your own `core/config.py` already documents. Reproduced here:

| crop | psm 3 | psm 6 |
|---|---|---|
| 14f7798c at scale 0.49 | **0 chars** | 784 chars |
| 71a97929 at scale 0.40 | 561 chars | 693 chars |

So the `1.0000` row in your table was not a resolution failure at all. It was
psm 3 silently returning nothing on a 277×445 image — the same class of silent
zero that produced the original YOLO bug. And psm 3 was under-reading the sharp
capture by 20% at the same time, which quietly penalised every row.

**Unequal coverage.** A closer frame holds fewer lines of the same column, so
`71a97929` was being scored against text it physically cannot see. There is now
a `--lines N` option that reads the first N lines of the column in every
capture, and a warning when the captures differ by more than a line.

## 3. What the corrected run does show

Same 20 lines in all three, psm 6, my Tesseract (OCR only — no mT5 here):

| capture | `p75` | scale | →px | OCR CER | OCR WER |
|---|---|---|---|---|---|
| 71a97929 | 38 | 0.40 fixed | 15.2 | 0.0994 | 0.3839 |
| 71a97929 | 38 | **0.35 auto** | 13.2 | **0.0863** | 0.3393 |
| e6e14f1b | 28 | 0.40 fixed | 11.2 | 0.0819 | 0.3482 |
| e6e14f1b | 28 | **0.47 auto** | 13.2 | **0.0804** | 0.3482 |
| 14f7798c | 26.5 | 0.40 fixed | 10.6 | 0.2953 | 0.7411 |
| 14f7798c | 26.5 | 0.50 auto | 13.2 | 0.2003 | 0.5804 |

**Two findings survive the mess.**

**a) Backing off from `p75` 38 to 28 is free.** 0.0863 → 0.0804. No accuracy
cost at all for a 26% wider field of view. That is real headroom you were not
using, even though 28 still clips this article.

**b) `auto` beats `fixed` on every single capture.** The fixed
`CLOSEUP_OCR_SCALE = 0.40` is costing accuracy on its own, independently of the
whole framing question — most dramatically on the soft frame (0.295 → 0.200),
but on the sharp one too (0.099 → 0.086). It is a fixed factor where the thing
that matters is a target glyph height, and your region path
(`rescale_to_optimum`) already gets this right. The close-up path does not.

That second finding is worth a paragraph in Chapter 4 regardless of how the
framing question resolves.

## 4. What to do — one re-capture, then re-run

**Step 1. Re-capture the wide framing, sharp.**

Same article, same lighting, phone at the distance that produced `14f7798c`
(the one where the whole article width was in frame). Take it **three or four
times**, not once — sharpness degrades across a burst, which your own build
record measured. Tap the screen to refocus before each shutter if the app
allows it.

Everything else in `Distance_Test_Steps.md` still applies: paper flat, phone
parallel to the page, fingers off the text.

**Step 2. Check focus before spending any time on OCR.**

```
cd E:\RP\R26-DS-002\system
```
```
python tools\compare_framing.py --dry-run --lines 20 --save-text tools\out\cer work\71a97929 work\e6e14f1b work\<the new job>
```

Look at the `sharp@px` column. **If the new capture is not within about 3x of
the others, stop and re-capture.** No amount of analysis fixes a soft frame,
and the warning will tell you.

**Step 3. Trim the ground truth to 20 lines.**

`article_truth.txt` is 893 characters, which is the whole column. The 20-line
crop ends at `විගණකාධිපතිතු` — cut everything after that word. It becomes 684
characters. (Keep the full version as `article_truth_full.txt`; it will be
useful if you ever score whole crops.)

**Step 4. Re-run for real.**

```
python tools\compare_framing.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --gt ..\Work\Ishara\article_truth.txt --lines 20 --save-text tools\out\cer work\71a97929 work\e6e14f1b work\<the new job>
```

**Step 5.** Send me the table. With a sharp wide capture, the comparison is
finally the one we meant to run.

## 5. What has still not changed

Nothing in `core/` or `layers/l4b_body/`. `CLOSEUP_OCR_SCALE` is still 0.40 in
the deployed path. Finding (b) above says it should become a target glyph
height — but that change waits until the re-run confirms it on a sharp frame,
because right now the strongest evidence for it comes from a blurred one.

## 6. Priority

Unchanged, and worth repeating because this thread has now consumed several
days: **Chapters 3, 4 and 5.** The re-capture is ten minutes and the re-run is
five. If they do not happen this week, drop this and write.
