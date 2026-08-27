# The four components — what each one is, and what actually runs

**R26-DS-002 · 27 August 2026.** Written after reading every file under
`Work/Nadee`, `Work/Bumal`, `Work/Janindu` and `system/`, and after measuring
the article-boundary behaviour on the nine real captures in
`F:/App/backend/inbox`.

Every number below is measured or quoted from the teammate's own files.
Where something has not been measured, it says so.

---

## The system in one picture

```
                    ┌── READING PATH ─────────────────────────────────┐
 phone              │  L2 select    best frames of the burst          │
  guidance,         │  L3 layout    columns, gutters, deskew,         │
  auto-shutter ────▶│               drop clipped columns, centre block│
  POST /capture     │  L3 headline  attach THIS article's headline    │
                    │  L4A title    read it   (Component 1 models)    │
                    │  L4B body     OCR + mT5 (Component 2)           │
                    │  L4C polish   optional LLM repair, OFF          │
                    │  L5 assemble  + readability gate                │
        ◀───────────┤  session[job] = the article                     │
  speaks it         └────────────────────────────────────────────────-┘
                    ┌── CONVERSATION PATH ───────────────────────────┐
  volume-down ─────▶│  L0 voice     intent + style (Component 4)      │
  POST /ask         │  L6 generator retrieval + answer (Component 3)  │
        ◀───────────┤                                                 │
  speaks it         └────────────────────────────────────────────────-┘
```

---

## Component 1 — layout and recognition · Janindu

### What his code actually does

Five stages under `Work/Janindu/1_Preprocess/`, driven by files on disk:

1. `stage1_LayoutDetection.run(img_path, out_dir)` → `(viz, corrected, boxes)`.
   PaddleOCR `PP-DocLayout_plus-L` at `threshold=0.20`, after `UVDoc` unwarp.
   Keeps `{"text", "document_title", "paragraph_title", "image_caption",
   "aside_text"}`.
2. `stage2_Crops.run(...)` → `dict[block_id, list[{word_id, img, is_dark_bg}]]`,
   line/word detection by **CRAFT** (not PaddleOCR, despite the docstring).
3–5. stitch → binarise → Medial Axis Transform.
Then `2_Recogniton/recognize_helper.py` runs Tesseract and writes
`frontend_summary.json`. A Flask UI on port 5000 drives it.

### What I found that matters

- **There is no article concept anywhere in it.** One layout box is one
  independent block: no headline↔body association, no column merging. Reading
  order is `sort(key=lambda b: (b.y1 // 100, b.x1))` — row-banded at 100 px,
  which interleaves newspaper columns.
- Worse, the filter **throws body text away**: if any title box exists, every
  `text` box is discarded and only headlines survive; otherwise only the three
  largest body blocks do.
- `training_summary.json` records a real result — EfficientNetV2-S, 315
  classes, 91,612 images, **best_val_acc 99.04%** at epoch 19 — but **nothing
  in the pipeline ever loads that model.** Recognition is Tesseract-only.
- `stage3_Sentences.run()` calls an undefined `_stitch(...)` → `NameError`;
  `stage4`/`stage5` `run()` are `pass` stubs. `trainer.py` points at a dataset
  directory that does not exist. Hardcoded paths to `C:\Users\JANINDU\...` and
  `E:\Sliit\...`.

### What is USED in the deployed system

**His two Tesseract models**, and they work. `layers/l4a_title/tessdata/`
holds `sin_raw` and `sin_custom`. Measured on the located headline regions:

| | result |
|---|---|
| `sin_raw` psm 11 | `'කුරුණෑගල නගර විගණනයක් ලබා'` — near-correct |
| `sin_raw` psm 7 | `'දදු'` — collapses |
| `sin_custom` psm 11 | `'දදකදීද්ථීදල, එද්්ට දඉීීීීර්ද'` — garbage |

`sin_custom` is the **MAT** model — trained on skeletonised glyphs, garbage on
raw pixels, exactly as `l4a_title/README.md` warns. The full MAT pipeline is
NOT used; the deployed path reads the headline crop directly with `sin_raw`.

---

## Component 2 — post-OCR correction · Ishara (the research)

Unchanged and untouched by any of this week's work.

| | before | after | change |
|---|---|---|---|
| CER | 0.1197 | **0.0757** | −36.7% |
| WER | 0.3358 | **0.1640** | −51.2% |

n = 217 sentences, page-disjoint, mT5-small plain full-sequence. The
SinBERT-gated span corrector **underperformed** it — the negative result.

Two things now protect that number:
- Layer 4C (LLM post-edit) writes to `body_polished`, **never** to `body`, and
  `tests/test_polish.py` asserts no evaluation tool can enable it.
- The RAG payload carries a difflib diff with `token_source: "diff"` and **no
  per-token confidence**, because there is no classifier to produce one.

---

## Component 3 — retrieval and generation · Nadee

### What her code does

`run_pipeline(vectorstore, ocr_input, voice_input)` →
`{intent, answer_si, retrieved_sources, speakable_text}`. LangChain + Chroma +
`intfloat/multilingual-e5-small`, generation through `ChatOpenAI`.

### What I found

- **`corpus/articles.jsonl` does not exist anywhere in the repository.** That
  is the real reason Component 3 has never run end to end, not any bug.
- `generate.py` builds `ChatOpenAI(...)` and `vectorstore.py` builds
  `HuggingFaceEmbeddings(...)` **at import time** — merely importing the
  module needs a key and downloads a model.
- The model name `"gpt-5.4-mini"` is hardcoded in two files, no override.
- `tokens` is parsed by `parse_ocr_input` and then **never used** — only
  `corrected_text` is ever embedded. This is why sending a diff instead of
  fabricated confidences costs nothing.
- `pipeline.py` swallows every generation exception into one Sinhala string,
  so auth and API failures were invisible.

### What runs now

`services/rag/` — her prompt, her Sinhala-purity retry, her word-limit tables,
her chunk metadata and her "always retrieve the current page" rule, kept
verbatim. `Work/Nadee/` untouched. Underneath: a numpy store and API
embeddings instead of chroma + langchain + sentence-transformers.

**The corpus problem is solved by indexing what it reads** — every captured
article is stored, so the corpus builds itself from use. `--seed <folder>`
still loads a real corpus if one ever arrives.

---

## Component 4 — voice interaction · Bumal

### What his code does

`handle_voice_command(sinhala_text, user_id, retrieved_chunk_id=None)` →
`{route, intent, english_translation, style_class, prompt_modifier,
personalization_flags, retrieved_chunk_id, correction_applied}` — which is
**exactly** what Nadee's `parse_voice_input` requires. The one place in this
project where two components agreed on an interface unprompted.

- **Intent**: Approach 1 is the confirmed choice — NLLB-200-distilled-600M
  (≈600 MB, loaded at import) + Llama 3.2 1B through **Ollama on
  localhost:11434**. Approach 3 is a trained classifier,
  `intent_classifier_bundle.joblib`, **89% accuracy / 0.88 macro-F1, 0.897
  mean 5-fold**, 530 samples, 17 classes — the only stored accuracy number in
  his tree.
- **Personalization**: a `river` online model (`TFIDF → SoftmaxRegression`)
  pickled to `data/style_model.pkl`, and a TinyDB log at `data/db.json`
  (2 user profiles, 21 interactions). It relabels the PREVIOUS turn when the
  current turn implies the last style guess was wrong.
- **STT**: all four files are Colab exports — `from google.colab import
  drive`, bare `!pip install` — and none can be imported.

### The alignment worth knowing

`stt_all_approaches.py` states: *"Final Selection: Google STT (si-LK) — chosen
for best Sinhala accuracy and zero local GPU dependency."*

**That is exactly what the phone now does.** Android's `SpeechRecognizer` with
`si-LK` IS Google STT, on-device, no server model, no audio on the network. His
selection and the deployed path agree; the Wav2Vec2-BERT work is the
comparison that led to it.

Today `layers/l0_voice` uses a literal-keyword stub with nine local commands.
Wrapping `handle_voice_command` behind `POST /interpret` is the remaining
work, and needs Ollama running.

---

## What changed today in article-wise reading

You asked for the system to read **an article**, not whatever text is in the
frame. Three things were measured and two were wrong.

**1. Body isolation was already correct.** `fig_article_isolation_g27.jpg`
shows it: the crop covers the article's two body columns, the clipped right
column is dropped, and the neighbouring articles above and below are outside
it.

**2. The headline threshold was wrong.** Measured on nine captures:

| | of the median line height |
|---|---|
| tallest BODY line | **1.28× – 1.70×** |
| tallest HEADLINE band | **5.91× – 8.71×** |

Nothing lands between them. The old constant was **1.6** — inside the body
range — so the tallest body line of every capture was reported as a headline.
Now `TITLE_MIN_LINE_RATIO = 3.0`, in the middle of the empty gap.

**3. The article's own headline was never read.** It is now located and
attached, with three tests that must all pass — gap, x-overlap, and a single
contiguous row group — and it **refuses** rather than guessing.
`fig_masthead_problem_g26.jpg` shows why: a masthead, a page number and the
section strip "ප්‍රාදේශීය පුවත්" sit above the real headline, all
headline-sized. Reading those aloud as the headline would be worse than
silence.

**Result: a headline is attached on 8 of 9 captures** (the ninth is not a
close-up at all and is refused earlier). All eight produce recognisable
Sinhala.

**And a fixed-scale defect, the same shape as the one you already found for
the body.** Headline bands are 250–325 px; Tesseract collapsed at native
scale on one capture (`'දිමුදු ිළ ුකී ි දු ී'`) and recovered at any
downscale into 40–90 px (`'ණෑගල නගර සම ඔණනයක් ලබා දෛ'`).
`TITLE_TARGET_BAND_PX = 90`.

Reproduce all of it: `python tools\measure_headline.py --ocr`

### Still true, and worth stating

- Coloured headlines are lost. These pages print part of the headline in red
  and a grayscale Otsu threshold drops it — `අකුමිකතා` is missing from every
  reading of that capture.
- A clipped headline reads partially. That is a capture problem.
- Nine captures from three scenes is not a validated detector. The constants
  are a measured separation on this project's own data; re-run the tool if the
  capture path changes.
- These measurements were made with **cv2 5.0.0**, not the pinned 4.9.0. The
  functions used are version-stable (`threshold`, `morphologyEx`,
  `findContours`, `boundingRect`) — unlike `minAreaRect`, which is why
  `deskew_angle` uses a projection profile — but re-run on the pinned
  environment before quoting a number in a chapter.

---

## Switches

| variable | default | what it does |
|---|---|---|
| `SINHALA_TITLE_MODE` | `stub` | `mat` reads the headline with `sin_raw` |
| `SINHALA_POLISH_MODE` | `off` | `auto` \| `on` — LLM post-edit |
| `SINHALA_VOICE_MODE` | `stub` | `http` → Component 4 service |
| `SINHALA_RAG_MODE` | `off` | `http` → Component 3 service |

Tests: **236**. `python -m pytest tests -q`
