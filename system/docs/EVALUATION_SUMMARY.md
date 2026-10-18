# Dakma — Evaluation Summary

**Interactive Sinhala Text Reader for the Visually Impaired using OCR, NLP, AI**
R26-DS-002 · compiled 18 Oct 2026

The four headings for each component and for the full system:

- **Test:** automated tests that check the component behaves as specified.
- **Benchmark:** a fixed dataset, with a comparison against a baseline where one exists.
- **Accuracy:** the main quality metric for the task (CER/WER, Hit@k, % correct).
- **F1:** precision, recall and F1 for every part that makes a classification decision.

Every number below has a source file. Rows marked **TBD** have not been measured yet. Do not fill them with estimates.

---

## 1. Summary

| Component | Owner | Test | Benchmark | Accuracy | F1 |
|---|---|---|---|---|---|
| C1 Sinhala OCR (titles) | Janindu | 60 tests | TBD | TBD (CER) | TBD (char-F1) |
| C2 Reader + mT5 correction | Ishara | 111 tests | 217 pages, page-disjoint | **CER 0.1197 → 0.0757** | TBD (char-F1, boundary F1) |
| C3 RAG question answering | Nadee | 46 tests | 72 questions, 6 retrievers compared | **Hit@5 0.95, MRR 0.82** | **Refusal F1 0.65** |
| C4 Voice + personalisation | Bumal | 154 tests | 500 commands, 4 approaches compared | **Intent acc. 65.6%** | **Macro-F1 0.60** |
| **Full system** | all | **413 tests: 405 pass, 3 fail, 5 skip** | end-to-end set (§6) | TBD (task success) | TBD |

---

## 2. Component 1 — Sinhala OCR (title OCR, layout) · Janindu

### Test
- `tests/test_layout.py`: 33 tests.
- `tests/test_segfix.py`: 13 tests.
- `tests/test_sentences.py`: 7 tests.
- `tests/test_vote_lines.py`: 7 tests.

### Benchmark
- **TBD.** Needs a title test set: crops from `Work/Janindu/0_Data/Test` with correct transcriptions.
- **Baseline:** stock Tesseract `sin`.
- **Compared against:** the custom pipeline (preprocess stages 1–5 + `tessdata` model).

### Accuracy
- **TBD.** Report CER and WER on the title set.

### F1
- **TBD.** Report character-level precision, recall and F1, taken from the same edit-distance alignment as CER.
- If layout detection is evaluated, report region-detection P/R/F1 at IoU ≥ 0.5.

---

## 3. Component 2 — Reader: capture, segmentation, body OCR, mT5 correction · Ishara

### Test

| Test file | Tests | Covers |
|---|---|---|
| `test_closeup.py`, `test_closeup_scale.py` | 28 | capture scale and close-up |
| `test_article_boundary.py` | 16 | article boundaries |
| `test_imaging.py`, `test_exif.py`, `test_quality.py` | 25 | frame quality and selection |
| `test_polish.py`, `test_text.py` | 23 | text clean-up |
| `test_rag_payload.py` | 14 | payload contract to C3 |
| `test_corpus_verdict.py`, `test_metric_hygiene.py` | 5 | metric hygiene |

### Benchmark
- mT5 correction, tested on **n = 217 page-disjoint pages**. Source: `docs/INTEGRATION_CONTRACT.md`.
- **Baseline:** raw OCR output.
- **Negative result:** a SinBERT-gated span corrector was also tested, and it did not beat plain mT5.
- **Distance test:** 8 captures of one article at increasing distance. The whole article first fits at `glyph_p75` = 25. Source: `Work/Ishara/Distance_Test_Result.md`.

### Accuracy

**OCR correction (main result):**

| | CER | WER |
|---|---|---|
| Raw OCR | 0.1197 | 0.3358 |
| **After mT5 correction** | **0.0757** | **0.1640** |
| Relative improvement | −36.8% | −51.2% |

**OCR at different capture distances.** Indicative only: these use 20 lines, Tesseract only, with no correction. Source: `Work/Ishara/CER_Test_Result.md`.

| Capture | `glyph_p75` | CER | WER |
|---|---|---|---|
| 71a97929 | 38 | 0.0863 | 0.3393 |
| e6e14f1b | 28 | 0.0804 | 0.3482 |
| 14f7798c (out of focus) | 26.5 | 0.2003 | 0.5804 |

### F1
- **TBD:** character-level F1 for the corrected text.
- **TBD:** article-boundary detection P/R/F1 against `Work/Ishara/article_truth.txt`.

---

## 4. Component 3 — RAG question answering · Nadee

### Test
- `tests/test_rag_service.py`: 31 tests, of which **2 currently fail**.
- `tests/test_ask.py`: 15 tests.
- In `Work/Nadee`: `test_history.py`, `test_sessions.py` and `test_prompt_history.py`.

### Benchmark

**Evaluation set:** `Work/Nadee/eval_set.jsonl`, 72 questions:
- 56 answerable
- 8 unanswerable but on a related topic ("near")
- 8 unanswerable and off-topic ("far")

**Retrieval comparison** (56 answerable questions, article level). Source: `Work/Nadee/eval_results/retrieval_summary.json`.

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Dense (English) — baseline | 0.571 | 0.714 | 0.768 | 0.657 |
| Dense (Sinhala) | 0.643 | 0.768 | 0.857 | 0.727 |
| BM25 word | 0.625 | 0.875 | 0.929 | 0.748 |
| BM25 n-gram | 0.732 | 0.911 | 0.929 | 0.822 |
| Hybrid | 0.661 | 0.857 | 0.946 | 0.769 |
| **Hybrid (Sinhala)** | **0.732** | **0.911** | **0.946** | **0.823** |

### Accuracy
End-to-end RAG (`Work/Nadee/eval_out_en/summary.json`):

| Metric | Value |
|---|---|
| Article hit@1 / @3 / @5 | 0.589 / 0.804 / 0.821 |
| Article MRR | 0.686 |
| Answers fully correct (score 2/2) | 34 / 56 = **60.7%** |
| Mean correctness (0–2) | 1.339 |
| Key fact in answer | 53.6% |
| Faithful to context | **90.2%** |
| Correct refusal on unanswerable questions | 93.8% (far 100%, near 87.5%) |
| Wrong refusal on answerable questions | 26.8% |
| Sinhala purity of answer | 0.986 |
| Within word limit | 100% |
| Perplexity, answered questions (mean) | 1.143 |
| Perplexity per style (Detailed / Simple / StepByStep) | 1.93 / 1.79 / 1.79 |

### F1
Refusal decision, taking **refuse** as the positive class (72 questions):

| | Should refuse | Should answer |
|---|---|---|
| **Refused** | TP 15 | FP 15 |
| **Answered** | FN 1 | TN 41 |

- **Precision:** 0.50
- **Recall:** 0.94
- **F1:** **0.65**
- **Accuracy:** 0.78

The weakness is clear from this: the system refuses too often, not too rarely.

---

## 5. Component 4 — Voice, intent detection, personalisation · Bumal

### Test
- `tests/test_turns.py`: 119 tests.
- `tests/test_local_commands.py`: 18 tests.
- `tests/test_session.py`: 11 tests, of which **1 currently fails**.
- `tests/test_tts_clean.py`: 6 tests.

### Benchmark
- **Test set:** 500 generated Sinhala voice commands, 17 intents. Source: `Work/Bumal/approach1_generated_results.csv`.
- **Approaches compared:** see the table below. Source: `intent_detection/evaluate_all.py`. That run used only 7 samples, so treat the ranking as indicative.

| Approach | Accuracy (7 samples) | Mean latency |
|---|---|---|
| A1 NLLB translation + Llama | 85.7% | 7.38 s |
| A2 Qwen direct | 0.0% | 8.21 s |
| A3 trained classifier | 57.1% | 1.42 s |
| Hybrid | 71.4% | 2.79 s |

- **STT:** a WER comparison script exists in `stt/sinhala_stt_wer_comparison.py`. Results are **TBD**.

### Accuracy
Approach 1 on 500 commands:
- **Accuracy:** **65.6%** (328 / 500)
- **Latency:** median 5.89 s, p95 7.45 s
- **STT WER:** TBD

### F1
Approach 1 on 500 commands:
- **Macro-F1:** **0.60**
- **Weighted F1:** 0.65

| Intent | n | Precision | Recall | F1 |
|---|---|---|---|---|
| STOP | 31 | 0.97 | 0.97 | **0.97** |
| SUMMARIZE | 57 | 0.85 | 0.81 | 0.83 |
| GO_TO_START | 12 | 0.69 | 0.92 | 0.79 |
| REPEAT | 34 | 0.86 | 0.74 | 0.79 |
| SIMPLIFY | 51 | 1.00 | 0.65 | 0.79 |
| READ_HEADLINES | 29 | 0.84 | 0.72 | 0.78 |
| GO_TO_END | 10 | 0.75 | 0.60 | 0.67 |
| PREVIOUS_PAGE | 25 | 1.00 | 0.48 | 0.65 |
| EXPLAIN | 51 | 0.49 | 0.92 | 0.64 |
| PREVIOUS_ARTICLE | 29 | 0.83 | 0.52 | 0.64 |
| NEXT_ARTICLE | 25 | 0.43 | 0.80 | 0.56 |
| REPHRASE | 48 | 1.00 | 0.38 | 0.55 |
| NEXT_PAGE | 30 | 0.38 | 0.90 | 0.53 |
| NAVIGATE_SECTION | 27 | 0.88 | 0.26 | 0.40 |
| READ_ARTICLE | 8 | 0.26 | 0.75 | 0.39 |
| IDENTIFY_CONTENT | 21 | 0.50 | 0.19 | 0.28 |
| NAVIGATE_PAGE_NUMBER | 12 | 0.00 | 0.00 | 0.00 |

**Most common confusions:**

| True intent | Predicted as | Count |
|---|---|---|
| REPHRASE | EXPLAIN | 24 |
| NAVIGATE_SECTION | NEXT_PAGE | 15 |
| SIMPLIFY | EXPLAIN | 13 |
| NAVIGATE_PAGE_NUMBER | NEXT_PAGE | 12 |
| IDENTIFY_CONTENT | READ_ARTICLE | 11 |

The integrated system adds a rule-based turn policy (`layers/l6_generator/turns.py`) and local command words on top of the classifier. Its intent accuracy on the same 500 commands is **TBD**.

---

## 6. Full system (end-to-end)

### Test
The whole suite runs with `pytest -q` in `system/`. Run on 18 Oct 2026:

| Collected | Passed | Failed | Skipped |
|---|---|---|---|
| 413 | **405** | 3 | 5 |

**Failing tests (fix before the presentation):**
- `test_rag_service.py::test_a_code_switched_answer_is_retried_in_sinhala`
- `test_rag_service.py::test_no_evidence_is_a_failure_even_though_it_is_speakable`
- `test_session.py::test_a_new_capture_resets_the_cursor`

**Integration and contract tests:**
- `test_contracts.py` (6 tests): the shared schema between layers.
- `test_services_http.py` (20 tests): HTTP calls between services.
- `test_api.py` (16 tests): server endpoints.

### Benchmark — TBD (protocol)
1. **Pages:** 15–20 newspaper photos taken with the phone app. For each page, write the correct article text and boundaries.
2. **Turns:** 5 spoken turns per page (≈ 100 turns): read, navigate, a question, a style change, repeat.
3. **Run:** capture → select → segment → OCR → correction → intent → RAG → TTS, with nothing corrected by hand between stages.
4. **Record per turn:**
   - Success (did the user get the right content?)
   - Which stage failed
   - Latency

### Accuracy — TBD

| Metric | Value |
|---|---|
| Task success rate (turns that gave the correct result) | TBD |
| Pages read end-to-end without failure | TBD |
| End-to-end CER, photo → spoken text | TBD |
| Latency per turn (median / p95) | TBD |
| Capture to first audio | TBD |

### F1 — TBD
- **Turn success:** precision, recall and F1, using a human judgement of each turn as ground truth.
- **Error attribution:** what share of failures each stage causes (OCR / segmentation / intent / RAG).

---

## 7. How to reproduce

| What | Command |
|---|---|
| Unit and integration tests | `cd system && pytest -q` |
| RAG retrieval | `python Work/Nadee/eval_retrieval.py` |
| RAG end to end | `python Work/Nadee/eval_rag.py` |
| Perplexity | `python Work/Nadee/eval_perplexity.py` |
| Intent detection | `python Work/Bumal/intent_detection/evaluate_all.py` |
| OCR CER/WER | `python tools/compare_framing.py --gt ../Work/Ishara/article_truth.txt --lines 20 ...` |
