# Session record — 24 August 2026

**R26-DS-002 · Component 2 / integration · IT22259134**

Five changes, every one justified by a measurement made today. 56 tests pass.

---

## 1. The whole article is now read — 2.2× more text

Both captures through the real pipeline, same article:

| capture | framing | raw OCR | after mT5 | warning |
|---|---|---|---|---|
| `71a97929` | `p75` 38, clipped | 1599 chars | 1577 | *"part of this article is off the right of the frame — move a little to the right"* |
| `80654199` | `p75` 22, whole | **3499 chars** | 3504 | none |

**2.2× more of the article**, and the clipped capture now says so out loud
instead of reading a fragment in silence. That is the close-up path working as
intended for the first time.

## 2. The duplicated passages: it was the voter, not the dedup

This has been an open item since the first handoff, filed as *"a repeated
passage `strong_dedup` cannot span"*. It was never a dedup problem.

`vote_lines()` did medoid voting **by line index**: the candidates for output
line *k* were `frame[k]` from every frame. That is only correct while every
frame produces the same number of lines in the same order.

Measured on `work/80654199` — three frames of one static scene, identical
crop:

```
frame 0 -> 105 lines
frame 1 -> 106 lines
frame 2 -> 101 lines
```

From the first divergence onward, index *k* was a **different physical line**
in each frame, and the medoid chose between unrelated candidates.

| | voted lines | near-duplicate lines | final chars |
|---|---|---|---|
| index-based (old) | 100 | **15 (15%)** | 3494 |
| content-aligned (new) | 99 | **0** | **3765** |

The repeats it produced are exactly the ones in the run above —
`බරක සිරවී ඇති බව මෙහිදී පෙන්වා දුන්නේය`,
`ප්‍රධාන වෙළෙඳ සංකීර්ණය ඉදිකිරීමට`. And it was **losing 271 characters** while
doing it.

**The fix.** Two parts, both necessary:

- **Align by content, not index.** The reference frame's line order and count
  are preserved exactly; every other frame may only *correct* a line, never
  insert or reorder one. Matching is limited to ±4 lines and a 0.60 similarity
  floor.
- **Choose the reference as the medoid frame, not the median-length one.**
  Length alone picks a corrupted frame as readily as a good one, and a
  corrupted reference cannot be out-voted — nothing matches its bad line, so it
  has no competition and survives. There is a test for exactly that.

Worth a paragraph in Chapter 4: multi-frame consensus was reported as a
*benefit* (comparable-quality views measured 7.9% better). It was, but the
implementation was silently discarding a fifteenth of the article and repeating
another fifteenth. The measured benefit stands; the number was obtained despite
this, not because of it.

## 3. The capture gate was giving the wrong advice

L2 warned on the best frames the system has ever taken:

```
f0.jpg  p75 22.0  warn — glyph 22px (want >=25) — closer
```

"Closer" is the **wrong direction**. 22 is where the whole article fits and
where it reads better (mT5 CER 0.0497 against 0.0570 at `p75` 38). Following
that advice puts a column back off the edge of every capture — and a blind
listener cannot see that the instruction is wrong.

`CAPTURE_MIN_GLYPH_P75 = 25.0` **stays exactly as it is** — it is the
reproduction of the corpus diagnostics verdict on 168 pages and Chapter 4 cites
it. It answers *"is this page good enough to OCR at all?"*. The phone path asks
a different question: *"should the user move?"*. That now has its own constant,
`CAPTURE_WARN_BELOW_P75 = 20.0`, matching `CLOSEUP_MIN_P75`.

Same lesson as the `p75`/`p90` conflation: two questions, two constants.

## 4. Also changed today

- **Adaptive OCR scaling.** `CLOSEUP_OCR_SCALE = 0.40` (fixed) →
  `CLOSEUP_TARGET_GLYPH = 15.0` with an 11 px floor. The fixed factor gave
  Tesseract 8.8 px at `p75` 22 — measured mT5 CER **0.2193**, against 0.0760
  for the same frame at 13.2 px.
- **`CLOSEUP_MIN_P75` 28 → 20**, so the whole-article framing is accepted, plus
  a **second gate** so corpus full pages cannot slip through the widened one: a
  text band wider than 90 glyph-heights means no gutter was found. Measured —
  phone close-ups 27–59, corpus pages 131 and 272.
- **Layout-based cropping in the pipeline** — clipped columns dropped, crop
  stops at the block boundary so the next story's headline is excluded, all
  frames deskewed by one angle, warnings carried into the response.

## 5. Still open

- **`Guidance.kt` still aims at `glyph_p75` 33–36.** Everything above works on
  a capture taken at 22, but nothing tells the user to stand there. Half a day,
  and it is the last thing between this and a working demonstration.
- **Latin leakage from mT5 got more visible on the longer text** — `with`,
  `ikon`, `One`, `ush`, `kinni`, `high`. Known over-correction mode, still
  unmeasured. An hour to count it across the 217-sentence test set, and it
  would give the taxonomy a named category it currently lacks.
- Articles wider than about four columns still need more than one capture —
  see `Large_Articles_Design.md`.

## 6. Re-run to confirm

```
cd E:\RP\R26-DS-002\system
```
```
python -m pytest tests -q
```
```
python tools\run_pipeline.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --chars 4000 work\80654199
```

Expect no repeated passages, and a raw length near **3765** rather than 3499.
