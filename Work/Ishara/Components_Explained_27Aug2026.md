# The four components — what each is, and what actually runs

**R26-DS-002 · 27 August 2026.** Written after reading every file under `Work/Nadee`, `Work/Bumal`, `Work/Janindu`, `system/`, and after measuring article-boundary behaviour on the nine real captures in `F:/App/backend/inbox`. Full copy at `E:\RP\R26-DS-002\Work\Ishara\Components_Explained_27Aug2026.md`, with three evidence figures beside it.

## Component 1 — layout and recognition · Janindu

Five file-driven stages: PaddleOCR `PP-DocLayout_plus-L` (threshold 0.20) after UVDoc unwarp → CRAFT line/word crops → stitch → binarise → MAT → Tesseract → `frontend_summary.json`, driven by a Flask UI on port 5000.

**There is no article concept anywhere in it.** One layout box = one independent block; no headline↔body association, no column merging. Reading order is `sort((y1//100, x1))` — row-banded at 100 px, which interleaves newspaper columns. Worse, the filter discards body text: if any title box exists, every `text` box is dropped and only headlines survive.

`training_summary.json` records EfficientNetV2-S, 315 classes, 91,612 images, **best_val_acc 99.04%** — but nothing in the pipeline loads that model; recognition is Tesseract-only. `stage3_Sentences.run()` calls an undefined `_stitch(...)` (NameError); `stage4`/`stage5` `run()` are `pass` stubs; `trainer.py` points at a missing dataset; hardcoded `C:\Users\JANINDU\...` and `E:\Sliit\...` paths.

**What is used:** his two Tesseract models. Measured on located headline regions — `sin_raw` psm 11 → `'කුරුණෑගල නගර විගණනයක් ලබා'` (near-correct); psm 7 collapses to `'දදු'`; `sin_custom` psm 11 → garbage. `sin_custom` is the MAT model, trained on skeletonised glyphs, garbage on raw pixels exactly as `l4a_title/README.md` warns.

## Component 2 — post-OCR correction · Ishara

Unchanged. CER 0.1197 → **0.0757** (−36.7%), WER 0.3358 → **0.1640** (−51.2%), n=217 page-disjoint, mT5-small plain. SinBERT-gated corrector underperformed — the negative result. Protected by: Layer 4C writes `body_polished` never `body`, and the RAG payload carries a difflib diff with `token_source: "diff"` and no fabricated confidences.

## Component 3 — retrieval and generation · Nadee

`run_pipeline(vs, ocr, voice)` → `{intent, answer_si, retrieved_sources, speakable_text}`. **`corpus/articles.jsonl` does not exist anywhere in the repo** — the real reason it never ran. `ChatOpenAI` and `HuggingFaceEmbeddings` are constructed **at import time**. `"gpt-5.4-mini"` hardcoded in two files. `tokens` is parsed then **never used** — only `corrected_text` is embedded. `pipeline.py` swallows every generation exception into one Sinhala string, hiding auth failures.

`services/rag/` now runs it: her prompt, purity retry, word-limit tables, chunk metadata and "always retrieve the current page" rule kept verbatim; numpy store + API embeddings underneath; corpus built by indexing what it reads.

## Component 4 — voice interaction · Bumal

`handle_voice_command(sinhala_text, user_id, retrieved_chunk_id=None)` returns exactly what Nadee's `parse_voice_input` requires — the one interface two components agreed on unprompted.

Intent: Approach 1 confirmed — NLLB-600M (loaded at import) + Llama 3.2 1B via **Ollama on localhost:11434**. Approach 3's trained classifier reports **89% accuracy / 0.88 macro-F1 / 0.897 mean 5-fold**, 530 samples, 17 classes — the only stored accuracy in his tree. Personalization: `river` TFIDF→SoftmaxRegression pickled to `style_model.pkl`, TinyDB log in `db.json`; it relabels the previous turn when the current one implies the last guess was wrong. STT: all four files are Colab exports, none importable.

**Key alignment:** `stt_all_approaches.py` states *"Final Selection: Google STT (si-LK)"* — which is exactly what the phone's `SpeechRecognizer` with `si-LK` does, on-device. His selection and the deployed path agree.

## Article-wise reading — what changed

1. **Body isolation was already correct** (evidence figure): crop covers the article's columns, clipped column dropped, neighbours excluded.
2. **The headline threshold was wrong.** Measured on nine captures — tallest BODY line 1.28×–1.70× of median line height; tallest HEADLINE band 5.91×–8.71×. Nothing between. The old constant was **1.6**, inside the body range. Now `TITLE_MIN_LINE_RATIO = 3.0`.
3. **The article's headline is now located, attached and read.** Three tests must pass (gap, x-overlap, single contiguous row group) or it **refuses** — mastheads, page numbers and the section strip "ප්‍රාදේශීය පුවත්" are all headline-sized and reading them aloud would be worse than silence. **Attached on 8 of 9 captures**, all producing recognisable Sinhala.
4. **A fixed-scale defect, same shape as the body one:** headline bands are 250–325 px; native-scale OCR collapsed on one capture and recovered at any downscale into 40–90 px. `TITLE_TARGET_BAND_PX = 90`.

Reproduce: `python tools\measure_headline.py --ocr`

**Limits:** coloured (red) headline words are lost to grayscale Otsu; clipped headlines read partially; nine captures from three scenes is a measured separation, not a validated detector; measurements used cv2 5.0.0 not the pinned 4.9.0 (functions used are version-stable, unlike `minAreaRect`) — re-run on the pinned environment before quoting in a chapter.

Tests: **236**.
