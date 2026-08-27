# Where the system actually is, and what to do about large articles

**R26-DS-002 · IT22259134 · 24 August 2026**

## 1. Is it fully working? No — two things, and only one is big

**What works now, tested:**

- the frame's columns are found, and a column the frame edge cut is dropped
- the crop stops at the block boundary, so the next story's headline is excluded
- OCR gets a constant 15 px glyph at any distance, instead of 8.8 px at wide
  framings and 15 px at close ones
- the response carries plain sentences the listener can act on — *"part of this
  article is off the right of the frame — move a little to the right"*
- a full newspaper page can no longer sneak into the close-up path
- 105 tests pass

**What does not:**

1. **The app still guides you to `glyph_p75` 33–36.** The server reads whole
   articles at 22 now; nothing tells the user to stand there. Until
   `Guidance.kt` is re-aimed, none of the above changes what you actually
   capture. **Half a day, and it is the single biggest gain left.**

2. **Articles wider than about four columns cannot be captured in one frame.**
   That is arithmetic, not a bug — see §2.

## 2. Large articles: the system detects them correctly and cannot read them

Be precise about which half is solved.

**Detection is solved.** The layout module knows when a column ran off the
frame — measured on every capture you have, including the two cases the gutter
test alone missed. It knows *which* edge, and it says so in words.

**Capture is not.** The ceiling:

| input | value |
|---|---|
| frame width | 2448 px |
| column + gutter at `glyph_p75` 22 | ~560 px (measured) |
| columns that fit | 2448 / 560 ≈ **4.4** |
| at the floor, `glyph_p75` 20 | ≈ 4.0 after the resolution loss |

**About four columns is the hard ceiling of a single capture** on this phone
and this newspaper. Below `p75` 20 the layout method stops working and the OCR
starts losing diacritics — both measured.

So: articles up to ~4 columns are fully readable today. A page lead spanning
five or six is not, at any distance.

## 3. Your idea — guide, re-capture, combine. Yes, and two thirds of it exists

You are right, and it is the correct design. Three parts:

| part | state |
|---|---|
| know the article is incomplete, and which way to move | **done** — `layout.warnings_for()` |
| join two overlapping captures by matching their text | **done** — `layout.overlap()` / `join()`, tested, refuses to join a seam it cannot verify |
| session state, and reading order across the tiles | **not done** — and reading order is the hard part |

Why reading order is hard: panning **down** a column, joining is line-by-line
and the existing `overlap()` handles it. Panning **across**, frame A holds
columns 1–4 and frame B holds 3–6; you have to recognise that 3 and 4 are the
same columns seen twice, and emit 1,2,3,4,5,6 — not A followed by B. That is
column-level matching, not line-level.

**It is tractable, because the columns are already separated.**
`layout.column_bands()` gives their x-ranges. What is missing is OCR *per
column* instead of one crop through psm 3 — and that change is worth making on
its own merits (§4, step B).

**The honest cost.** Full N×M tiling with automatic assembly: a fortnight, plus
the Android side, plus real testing with a blind user, because a capture flow
nobody can see is not something you can validate at a desk. That is not a
six-week-to-deadline project.

**But the two-capture version is a day**, and a five- or six-column article
needs exactly two captures. The generality is only required for a full-page
lead.

## 4. What I recommend, in order

**A · Re-aim `Guidance.kt` — half a day. Do this first.**
READY band from 33–36 down to about 22–26. Everything already built starts
working on real captures. Nothing else comes close for the time.

**B · OCR each column separately with psm 6 — half a day.**
Two reasons, and the first is independent of any of this: measured on your own
frame, psm 6 on a single column returned **693 characters against psm 3's
561** on the same crop, and psm 3 returned **zero** on one 277×445 image. One
crop through psm 3 is the weakest link in the read path. Doing it per column
also produces exactly the unit the joining in step C needs.

**C · Two-capture continuation — one day, only if A and B land quickly.**
The server keeps the previous capture's columns for a short while. When the
next capture arrives and an edge was open, match columns by text overlap and
speak only what is new. Where no overlap can be verified, say so rather than
splicing. `overlap()` already returns 0 for exactly that case and there is a
test for it.

This delivers your idea for articles up to eight columns, with no tiling
engine, no registration and no new algorithm.

**D · Full mosaic capture — after the thesis.**
Write it up as future work with the design above. It is a genuinely good
section: the problem is real, the constraint is measured (four columns), and
the approach is sketched with two of its three parts already built and tested.

## 5. On "industrial level" — one honest word

The capture path is now solid enough to demonstrate, and the measurements
behind it are real. But an industrial system for blind users needs things this
project has not touched: evaluation on more than one article, offline
operation, battery and latency budgets, error recovery when the network drops,
and — most of all — testing with actual blind users rather than a sighted
developer holding the phone.

Claiming "industrial" in the thesis would be the one thing that could undermine
a set of results that are otherwise carefully measured. **A research prototype
with honest numbers is what a final-year project should be, and it is what you
have.** Say that, and the negative result and the measured limits become
strengths instead of gaps.

## 6. Right now — the command that failed

`run_pipeline.py` wanted image files, not a folder. I have fixed it to accept
folders (the server writes one folder per capture, so a folder is the right
unit — and the multi-frame consensus wants all three frames anyway).

```
cd E:\RP\R26-DS-002\system
```
```
python tools\run_pipeline.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --chars 3000 work\80654199
```
```
python tools\run_pipeline.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --chars 3000 work\71a97929
```

Expect `80654199` to report 3–4 columns, 0 clipped, and body text running to
the end of the article. Expect `71a97929` to **warn** that a column is off the
right edge — it is clipped, and until this week nothing said so.

Send me both outputs and I will start on A.
