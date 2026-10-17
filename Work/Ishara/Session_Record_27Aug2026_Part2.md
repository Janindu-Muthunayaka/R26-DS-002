# Session record — 27 August 2026, part 2

**Component 3 (RAG) built, plus a readability gate and a guarded LLM post-editor.** Tests 180 → **225**. Full copy at `E:\RP\R26-DS-002\Work\Ishara\Session_Record_27Aug2026_Part2.md`.

## Do this first

**Rotate the OpenAI key** — it was pasted into a chat window, so it is compromised regardless. No key is written into any file in this repo; `.env` at the repository root is gitignored and `services/.env.example` documents the shape. `python tools\check_llm.py` lists what a key can actually reach, because no model name here is a guess (`Work/Nadee/generate.py` hardcodes `gpt-5.4-mini`; whether a key reaches that is not something source code can know).

## svc-rag — Component 3, built (`services/rag/`)

`store.py` (manifest.json + one numpy array), `ingest.py`, `answer.py`, `app.py` (`POST /answer`, `/ingest`, `/forget`, `/health`, `/stats`, `GET /`).

**The corpus problem is solved by indexing what is read.** `Work/Nadee/ingest.py` reads `corpus/articles.jsonl`, which has never existed in this repository — the actual reason Component 3 had never run end to end. The service now stores every article the reader captures (`source_type: "read"`), so the corpus builds itself from use and waits on nobody. `--seed <folder>` loads a real corpus in the shape her ingest expected; `POST /forget` clears what was remembered.

**Kept verbatim from Nadee:** the Sinhala prompt including "base only on the evidence" and the "not enough information" fallback (the grounding guarantee), the purity retry, the style/detail word-limit tables, chunk metadata, and the "always retrieve the current page" rule. `Work/Nadee/` untouched. Changed underneath: chroma + langchain + sentence-transformers → numpy array + API embeddings (3 packages instead of a torch stack, a fourth incompatible dependency set avoided, an on-disk format you can open and read).

**Two bugs caught by testing, not review:** (1) nothing was ever remembered — the page is indexed twice on purpose (`ocr_current`, replaced; `read`, kept) and a text-only dedupe key made the second look like a duplicate, so the self-building corpus silently never grew; key is now source_type + text. (2) A failure read as an answer — Component 3 answers failures with a *sentence*, and the reader saw non-empty and reported `ok: true`; there is now an additive `ok` field honoured by `l6_generator`, and service `notes` reach the warnings. Found in a live run.

## Readability gate — `core/quality.py`, always on, no model

A bad capture does not fail: Tesseract returns something, mT5 corrects it, and the phone reads it aloud confidently. A sighted developer sees garbage; a blind user cannot. Four surface statistics, calibrated on this project's own outputs (`tools/calibrate_quality.py`).

**A measurement bug it exposed in itself:** the first version scored ground truth identically to the worst OCR, because Sinhala dependent vowel signs are combining marks and `str.isalnum()` is False for every one — `ක්‍රියාත්මක` measured as 5 characters. Now counts Unicode categories L/M/N. Worth a methodology sentence: *a token-length measure that ignores combining marks is not measuring Sinhala.*

Honest calibration result, recorded in the module: on these files the continuous measures barely separate anything and `n_words` does the work — one real capture returned zero characters (psm 3 on a single-column crop). `fatal` separates **short** (a six-word brief, read with a warning) from **shattered** (replaced by a request for another photograph).

## Layer 4C — LLM post-editing. Built, OFF by default

Requested, built, and built to be distrusted. A model given shattered Sinhala returns a fluent sentence with names and numbers that were never on the page, read to a blind user in the same voice as real news; and a general model rewriting mT5's output means Chapter 4's CER no longer measures the model the thesis is about.

`SINHALA_POLISH_MODE` = off | auto | on. Writes to `Article.body_polished` — `body` is never overwritten. Four guards, failing any one discards the rewrite: character similarity ≥ 0.75, length ratio 0.70–1.30, Sinhala ratio must not fall, word count must not grow > 20%. Every touched article, including a rejection, carries a warning to the phone. `tests/test_polish.py` asserts no evaluation tool can enable it. **`auto` deliberately refuses `unreadable` text** — that is where invention is most likely and where "take another photograph" is the only real fix.

## Also

`core/llm.py` (chat + embeddings over urllib, no new dependency, never raises, retries once without a parameter the API rejects); `core/env.py`; `tools/run_all.py`; `tools/check_llm.py`. Vocabulary bug fixed: `කියන්න` ("tell me") was a READ keyword, so "මේ ලිපිය ගැන කියන්න" — *tell me about this article* — routed as *read it again*.

## Still not true

- **Nothing has touched the real API** — api.openai.com is unreachable from where this was written, so every LLM path is tested against a deterministic fake. Guards, retries, failure paths and the contract are tested; **answer quality is not and cannot be until it runs with a key.**
- The phone app still has not been built or run.
- The Sinhala strings are not a native speaker's (English aliases exist for every command).
- Quality thresholds are a separation on one article's captures — a hint, not a validated detector, not a result.
- Component 4 is still a stub; nothing routed through it is a measurement.
