# The distance test — result

**R26-DS-002 · IT22259134 · 23 August 2026**
**Supersedes §4 of `Article_Boundaries_Measured.md`, whose step-back estimate
was wrong by a factor of three.**

Eight captures of the same article, backing away in steps. Read from your own
`system/work/`, and I looked at the overlays.

## The answer

| capture | `glyph_p75` | columns | col width | R edge ink | verdict |
|---|---|---|---|---|---|
| 71a97929 | 38 | 3 | 954 px | 0.29 | **clipped right** |
| fef2c41f | 37 | 3 | 947 px | 0.43 | **clipped right** |
| e6e14f1b | 29 | 2 | 1122 px | 0.20 | **clipped right** |
| **14f7798c** | **25** | **4** | **550 px** | **0.00** | **whole** |
| 58e14f7f | 20 | 4 | 495 px | 0.36 | clipped right *(aimed differently)* |
| 722bf6b9 | 18 | 3 | 399 px | 0.01 | whole |
| c228e615 | 16 | 3 | 587 px | 0.00 | whole |
| 13de9dea | — | — | — | — | refused, see §3 |

**The whole article first fits at `glyph_p75` = 25.** At 29 it is still clipped.
I checked the overlay for 14f7798c by eye: the crop encloses all four body
columns and excludes the headline above and the two other stories below. It is
correct.

**25 is exactly `CAPTURE_MIN_GLYPH_P75`.**

That is the number with 168 corpus rows behind it. So the framing that holds
the whole article sits precisely on the measured resolution floor — not below
it, and not comfortably above it either. Zero margin.

What excludes that framing today is `CLOSEUP_MIN_P75 = 28`, and your own
config file already labels that one honestly: adopted from the app's
`NEAR_READY` band, a design choice, **not a CER measurement**.

So the outcome is neither of the two I predicted. It is:

> The whole article fits exactly at the measured resolution floor. The thing
> currently blocking it is an unmeasured constant, not a measured one.

## 1. My earlier estimate was wrong, and badly

I said the article was three columns and that backing off 8–18% would fit it.
It is **four columns**, and the real step is `glyph_p75` 38 → 25 — a factor of
1.5 in linear scale, **2.3× in area**. Four columns of 550 px plus gutters is
about 2400 px, against a 2448 px frame: it fits, and only just. At `p75` 38 a
column is 954 px, so barely two and a half columns can be in shot at once. No
small adjustment was ever going to fix this.

Note also that 58e14f7f is clipped at `p75` 20, further away than a frame that
is whole at 25. **Distance is not the whole story — aim is too.** The question
is what is inside the frame, not how far back you stand.

## 2. What this does NOT settle, and it is the important part

**Is OCR at `glyph_p75` 25 as good as at 33?** Nobody knows. That is a CER
question and it has never been measured on this path.

And there is a trap in the way. `CLOSEUP_OCR_SCALE = 0.40` is **fixed**, and it
was chosen by eye on a frame at `p75` 33 — giving an effective glyph of about
13 px after downscaling. Applied unchanged to a `p75` 25 frame it gives about
10 px, which is below the ~11 px where your own measurements show diacritics
die. **A naive comparison would show the wider framing is terrible, for the
wrong reason.**

So before the comparison is worth running, the close-up path has to scale to a
**target glyph height** rather than by a fixed factor — the same thing
`rescale_to_optimum()` already does on the region path. What that target should
be is precisely what the measurement decides.

## 3. A bug this run exposed

`13de9dea` reported `pitch 12 px` against a median glyph of 19 px, four
columns, and pronounced the frame **whole** with complete confidence.

Baselines cannot be closer together than a glyph is tall, or the lines would
overlap. A ratio of 0.63 is physically impossible: the autocorrelation locked
onto the wrong peak, and every number downstream was meaningless. It passed the
close-up gate because that frame's component heights are bimodal — body text at
19 px, headline much larger — which pushed `glyph_p75` up to 37 on a shot that
is plainly not a close-up.

`analyse()` now refuses when `pitch < 1.20 x median glyph`. Measured ratio on
frames that really are close-ups: **1.50 to 1.74**. On this one: 0.63. Clean
separation, and it is a physical constraint rather than a tuned threshold.

Also added: `--min-p75` on the tool. The deployed gate hid `14f7798c`, which is
the frame that answers the whole experiment. Refusing to measure below a
threshold is how you conclude "impossible" from the threshold instead of from
the page. **Measurement only** — the deployed path keeps the default.

```
python tools\diagnose_article.py --min-p75 14 --render tools\out\distance E:\RP\R26-DS-002\system\work
```

31 tests pass, including one for the impossible-pitch refusal and one for the
override.

## 4. What to do

**Not a fortnight of panning.** Not yet, and possibly not at all.

**The next step is one CER measurement, and it is small.** You already have the
two captures. Hand-transcribe that one article — call it 30–45 minutes of
typing — then run both frames through the real pipeline and compare:

| | `glyph_p75` 38 (71a97929) | `glyph_p75` 25 (14f7798c) |
|---|---|---|
| what is in frame | ~2.5 of 4 columns | all 4 columns |
| CER against your transcription | ? | ? |

Two outcomes:

- **CER at 25 is close to CER at 38** → lower `CLOSEUP_MIN_P75` to 25, aim the
  guidance at 25–28 instead of 33–36, and the whole problem is closed. The
  system reads whole articles.
- **CER at 25 is materially worse** → the conflict is real and measured, the
  guidance stays where it is, and panning becomes the answer — after the
  chapters.

Either way you end up with something Chapter 4 wants anyway: **the first
end-to-end CER on the deployed path**, which currently has none, and the
removal of the last by-eye constant in the read path.

Make the close-up scaling adaptive first, or the comparison measures the wrong
thing.

## 5. Priority

Unchanged. Chapters 3, 4 and 5. The transcription in §4 is the one piece of
system work worth doing before them, because it is short, it produces a thesis
number, and it decides whether a fortnight of engineering is needed at all.
