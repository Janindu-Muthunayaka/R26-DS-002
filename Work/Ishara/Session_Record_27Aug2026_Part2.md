# Session record — 27 August 2026, part 2

**Component 3 built, plus a readability gate and a guarded LLM post-editor.**
Tests 180 → **225**.

---

## 0. Do this first

**Rotate the OpenAI key.** It was pasted into a chat window, so it is
compromised regardless of what happens to it now. Revoke it at
platform.openai.com, issue a new one, and put the new one in `.env` at the
repository root — which is gitignored, and which `services/.env.example`
documents. No key is written into any file in this repo.

Git does not forget a committed blob, and `origin` is shared with three people.

```
copy services\.env.example .env
python tools\check_llm.py       <- lists what your key can ACTUALLY reach
```

`check_llm.py` exists because no model name here is a guess.
`Work/Nadee/generate.py` hardcodes `gpt-5.4-mini`; whether a key can reach
that is not something source code can know, so it asks.

---

## 1. svc-rag — Component 3, built (`services/rag/`)

| file | what |
|---|---|
| `store.py` | vector store: `manifest.json` + one numpy array |
| `ingest.py` | chunking; a seed corpus from `.txt`/`.jsonl` |
| `answer.py` | retrieval + generation — Nadee's prompt and guards, kept |
| `app.py` | `POST /answer`, `/ingest`, `/forget`, `/health`, `/stats`, `GET /` |

### The corpus problem is solved by indexing what is read

`Work/Nadee/ingest.py` reads `corpus/articles.jsonl`. **That file has never
existed in this repository** — the actual reason Component 3 had never run end
to end. Nobody was going to produce it before October.

So the service stores every article the reader captures (`source_type:
"read"`). The corpus builds itself out of use, and it waits on nobody. For a
personal reading assistant that is arguably the better corpus. If a real
newspaper corpus ever arrives, `--seed <folder>` loads it in the shape her
ingest expected, and `POST /forget` clears what was remembered.

### What is Nadee's, and is kept verbatim

The Sinhala prompt — including *"base only on the evidence"* and the *"not
enough information in the text that was read"* fallback, which is the
grounding guarantee — the Sinhala-purity retry, the style/detail word-limit
tables, the chunk metadata and the "always retrieve the current page" rule.
**`Work/Nadee/` is untouched.**

What changed is underneath: chroma + langchain + sentence-transformers became
a numpy array and API embeddings. Three packages instead of a torch stack, a
fourth incompatible dependency set avoided, and an on-disk format you can open
and read when a retrieval looks wrong at 11pm. `store.py` argues it.

### Two bugs caught by testing, not by review

- **Nothing was ever remembered.** The page is indexed twice on purpose —
  `ocr_current`, replaced each request, and `read`, kept. The dedupe key was
  the text alone, so the second looked like a duplicate of the first and the
  self-building corpus silently never grew. The key is now source_type + text.
- **A failure read as an answer.** Component 3 answers a failure with a
  *sentence* — a non-empty string. The reader saw non-empty and reported
  `ok: true`. There is now an additive `ok` field, honoured by
  `l6_generator`, and the service's `notes` reach the warnings. Found in a
  live run.

---

## 2. The readability gate — `core/quality.py`, always on, no model

A capture that goes wrong **does not fail**. Tesseract returns something, mT5
corrects that something, and the phone reads it aloud in the same confident
voice it uses for real news. A sighted developer sees garbage on a screen; a
blind user hears fluent nonsense and cannot tell it from the article.

Four surface statistics, no model, no network. Calibrated on this project's
own outputs — `python tools\calibrate_quality.py` reproduces the table.

**A measurement bug it exposed in itself.** The first version scored ground
truth identically to the worst OCR. Cause: Sinhala dependent vowel signs are
combining marks and `str.isalnum()` is False for every one of them, so
`ක්‍රියාත්මක` measured as 5 characters and a third of ordinary Sinhala looked
like fragments. Fixed to count Unicode categories L, M and N — and that is
worth a sentence in the methodology: *a token-length measure that ignores
combining marks is not measuring Sinhala.*

The honest result of the calibration is recorded in the module: on these files
the two continuous measures barely separate anything, and `n_words` is the
measure doing the work — one real capture returned **zero characters** (psm 3
on a single-column crop) and the system would previously have gone on to
correct nothing and read nothing.

`fatal` separates **short** from **shattered**: a six-word news brief is read
with a warning; fragments, Latin, or undecodable bytes are replaced by a
sentence asking for another photograph.

---

## 3. Layer 4C — LLM post-editing. Built, OFF, and it must stay off for Ch. 4

You asked for the API to clean up bad OCR. It is built. It is also the most
dangerous thing in this repository and it is built to be distrusted.

**Why.** A model given a shattered Sinhala sentence does not return a
shattered sentence. It returns a fluent, grammatical, plausible one — with
names, numbers and dates that were never on the page — and the phone reads it
to a blind user in the same voice it uses for the real article. Separately, if
a general-purpose model rewrites mT5's output, **the CER in Chapter 4 is no
longer measuring the model the thesis is about.**

- `SINHALA_POLISH_MODE` = `off` (default) | `auto` | `on`
- Writes to `Article.body_polished`. **`body` is never overwritten.**
- Four guards; failing any one DISCARDS the rewrite: character similarity
  ≥ 0.75, length ratio 0.70–1.30, Sinhala ratio must not fall, word count must
  not grow > 20%.
- Every touched article — **including a rejection** — carries a warning to the
  phone.
- `tests/test_polish.py` asserts no evaluation tool can enable it and that
  `l4b_body/body.py` does not import it.

**`auto` deliberately refuses `unreadable` text.** That is where a repair
would be most welcome and where invention is most likely — with little real
signal left, fluency is all the model has. The honest answer there is *"I
could not read that, try again"*, which is also the only thing that fixes it.

---

## 4. Also

- `core/llm.py` — chat + embeddings over `urllib`. **No new dependency**, same
  reasoning as `core/svc.py`. Never raises. Retries 429/5xx. Retries once
  without a parameter the API rejects, because model families disagree about
  `temperature` and `max_tokens` and this file refuses to guess which is in
  play.
- `core/env.py` — twenty-line `.env` loader. The shell wins over the file.
- `tools/run_all.py` — starts the reader plus whichever components are ready;
  Ctrl-C stops everything. Four processes in four environments is the cost of
  the architecture, and it is not a cost worth paying at a viva with three
  command windows to start in the right order.
- A vocabulary bug: `කියන්න` ("tell me") was a READ keyword, so *"මේ ලිපිය ගැන
  කියන්න"* — **tell me about this article** — was routed as *read it again*.
  Removed. A word that appears in ordinary questions is not a command word.

---

## 5. What is still not true

- **Nothing here has touched the real API.** api.openai.com is unreachable
  from where this was written, so every LLM path is tested against a
  deterministic fake. The guards, the retries, the failure paths and the
  contract are tested; **answer quality is not, and cannot be, until you run
  it with a key.**
- **The phone app has still not been built or run.**
- **The Sinhala strings are not a native speaker's.** Every voice command has
  an English alias so testing is not blocked on them.
- The quality thresholds are a separation on one article's captures. A hint
  that is better than nothing, which is what the system had. **Not a validated
  detector, and not a result.**
- Component 4 is still a stub. Nothing routed through it is a measurement.

---

## 6. Try it

```
cd E:\RP\R26-DS-002\system
python -m pytest tests -q                 # expect 225 passed
python tools\check_llm.py                 # after putting the NEW key in .env
python tools\run_all.py                   # reader + svc-rag
```

Then `http://127.0.0.1:8000/debug`: read a page, press the preset buttons, and
ask *"මේ ලිපිය ගැන කියන්න"* — that one now reaches Component 3.
