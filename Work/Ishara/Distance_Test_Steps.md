# The distance test — exact steps

**R26-DS-002 · IT22259134 · 22 August 2026**

**The question this answers, and nothing else:** how far back do you have to
stand before the whole width of an article fits in one frame — and is that
still close enough for the OCR?

Every one of your eight captures loses a column off the side. If backing off a
little fixes it, the whole problem is one guidance constant. If it does not,
panning and stitching is the only answer, and that is a fortnight of work. This
takes about twenty minutes and decides which.

---

## Before you start

**Terminal 1 — clear the work folder** so only today's captures are in it.

```
New-Item -ItemType Directory -Force E:\RP\R26-DS-002\Work\Ishara\jobs_old | Out-Null
```
```
Move-Item E:\RP\R26-DS-002\system\work\* E:\RP\R26-DS-002\Work\Ishara\jobs_old\ -Force
```

**Terminal 1 — start the backend** and leave it running.

```
cd E:\RP\R26-DS-002\system
```
```
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

Two things about that line, both of which I got wrong first time:

- `python -m app.server`, not `python app\server.py`. The second form puts
  `system\app\` on the import path instead of `system\`, so every
  `from core...` fails. (`server.py` now bootstraps its own path as the tools
  in `tools/` always did, so both forms work — but use `-m`.)
- `--root` is where the **models** live, which is the corpus folder, not the
  repository. `E:\RP\R26-DS-002` has no `layout\runs\...\best.pt` and no
  `models\mt5_plain`, which is what "YOLO weights not found" means.

It has started correctly when it prints `phone should POST to http://...`.

**Phone** — long-press the preview, check the server address, confirm it says
"Server connected".

---

## Setting up the page

1. **Pick one big article.** Three columns or more, with a headline across the
   top. The `කුරුණෑගල නගර සභාවේ` one is ideal — it is the one already measured.
2. **Lay the newspaper flat** on a table. Weigh the corners down if it curls.
   **Do not move it again** until all five captures are done.
3. **Same light throughout.** Do not switch a lamp on halfway.
4. **Keep your fingers off the printed text.** Hold the page at the very edge,
   or use a weight. A hand lying *on* the text destroys line detection — that
   is a known limitation and it is not fixed.
5. **Hold the phone flat** — parallel to the page, not tilted. Deskew corrects
   rotation. It does not correct perspective, and nothing in the system does.
6. **Aim at the middle of the body text**, not at the headline.

---

## The five captures

Take **five**, from closest to furthest. Same article every time.

| # | Where to stand |
|---|---|
| 1 | As close as the app will still fire. Move in until the guidance complains, then back off slightly. |
| 2 | A little further back. |
| 3 | Your normal, comfortable position. |
| 4 | **Clearly further back** — far enough that you can see the article's whole width inside the frame. |
| 5 | Further still, right up to where the app starts saying "closer". |

Between each one:

- **Wait for it to finish speaking.** Tap once to stop the reading early.
- **Wait about fifteen seconds** after that, so the server has finished writing
  the frames.
- Do not skip a capture. Five points is what makes the answer readable; three
  is not enough to see where it changes.

**Optional but worth it for the thesis:** measure the distance from the page to
the phone at each position with a tape, and write the five numbers down. It
turns "back off a bit" into a figure.

---

## Reading the answer

**Terminal 2:**

```
cd E:\RP\R26-DS-002\system
```
```
python tools\diagnose_article.py --render tools\out\distance --csv tools\out\distance.csv E:\RP\R26-DS-002\system\work
```

No `--root` here on purpose: this tool loads no models, so the root does not
apply to it.

Look at the block at the very bottom, `CLIPPING vs DISTANCE`. It sorts every
frame by `glyph_p75`, closest first, and marks each one `CLIPPED` or `whole`.

Three possible answers, and the tool prints which one you got:

**A — "Closest framing that keeps the whole article: glyph_p75 30-something",
and it says ABOVE both.**
The fix is a guidance constant. Aim the app's READY band at that number and the
problem is gone at no cost in resolution. An afternoon's work.

**B — the same line, but it says BELOW the close-up threshold.**
Fitting the article across the frame and meeting the resolution requirement are
genuinely in conflict on this newspaper. **Do not lower the gate** — the CER
cost of small glyphs is measured and severe. Panning and stitching is the
answer, and it waits until Chapters 3, 4 and 5 are drafted.

**C — "EVERY frame is clipped."**
Capture 5 was not far enough back. Take one more, clearly further away, and
re-run. Only if that is still clipped does answer B apply.

---

## Last step, and do not skip it

Open two overlays in `tools\out\distance` and **look at them**.

- Grey boxes = columns found.
- **Red** box = a column clipped by the frame edge.
- **Blue** box = exactly what gets sent to Tesseract.

Check that the blue box contains the article's body text and nothing else — not
the headline, not the story below, not your thumb. Every number in the table
depends on the columns being right, and that is something you can see in two
seconds and I cannot see at all.

Then send me the `CLIPPING vs DISTANCE` block and I will write the change.
