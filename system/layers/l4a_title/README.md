# Layer 4A — Title extraction

**OWNER: other team member. Do not edit this folder.**

## Contract

Input : `Article` with `regions` where `label == 'title'`, plus the frame image
Output: set `article.title_raw` (OCR output) and `article.title` (final text)

Implement `extract(img, article) -> article` in `title.py`. Touch nothing
outside this folder.

## Notes for whoever builds this

- Region boxes are in **full-frame coordinates**, not article-relative.
- `core.imaging.rescale_to_optimum()` is available and is what the body path
  uses. Titles are usually larger than body text, so the optimum may differ —
  measure rather than assume.
- If Medial Axis Transform is used, verify it against plain Tesseract at the
  optimal scale first. Sinhala dependent vowel signs (*pilla*) are only a few
  pixels wide and skeletonisation can destroy them.
- Consider running the title through Component 2 correction as well. Titles
  matter more to a listener than body text — they decide whether to keep
  listening.

## Stub behaviour

`title.py` currently returns the article unchanged so the rest of the system
runs end to end while this layer is under development.
