# Integration analysis — 27 August 2026

**R26-DS-002 · four components into one running system**
**Written after reading every source file in `system/`, `Work/*`, and `F:\App`.**

Nothing in this file is measured. It is a reading of the code as it stands and
a proposal. Every number quoted comes from `Handoff_26Aug2026.md`.

---

## 1. What actually exists, per component

| | owner | code state | how it is invoked today |
|---|---|---|---|
| **L1 capture app** | Ishara | working, calibrated 26 Aug | CameraX + `POST /capture` |
| **L2 select** | Ishara | working, tested | in-process |
| **L3 segment / layout** | Ishara | working, tested | in-process |
| **L4A title** | Janindu | **code delivered, not wired** | `title.py` is still `return article` |
| **L4B body + mT5** | Ishara | working, tested | in-process |
| **L5 assemble** | shared | working | in-process |
| **L6 RAG** | Nadee | runs standalone, **no corpus present** | `python main.py` |
| **L6/L7 voice + TTS** | Bumal | partly notebook code | never called by anything |

`python -m pytest tests -q` = 115 passed. That is the reading path only.

### The one structural fact that shapes everything else

These are **two systems, not four components of one**.

```
READING PATH        phone -> /capture -> L2 L3 L4A L4B L5 -> text -> phone TTS
                    exists, works, is measured

CONVERSATION PATH   voice -> STT -> intent -> personalization -> RAG -> answer -> TTS
                    exists as four folders of code and zero connections
```

The conversation path has **no entry point at all**:

- `AndroidManifest.xml` requests CAMERA and INTERNET only. **No `RECORD_AUDIO`.**
- `MainActivity.kt` has no microphone, no `SpeechRecognizer`, no second request.
- `app/server.py` exposes `/capture`, `/health`, `/audio/{job}`. There is no
  endpoint that takes a question.
- Nothing anywhere holds "the article that was just read", so a follow-up
  question has nothing to refer to.

Integration is therefore not "call four functions in a row". It is: **add a
second request path, and give the system one piece of memory.**

---

## 2. Interface mismatches found, in priority order

### 2.1 The `tokens[]` contract cannot be honoured — and must not be faked

`layers/l5_assemble/TempFormatPleaseRead.txt` and Nadee's `contracts.py`
both specify the L5 -> RAG payload as:

```json
{"corrected_text": "...",
 "tokens": [{"original": "...", "corrected": "...",
             "label": "ERROR", "confidence": 0.42, "was_changed": true}]}
```

**L4B cannot produce `label` or `confidence`.** mT5 is full-sequence seq2seq;
there is no per-token classifier in the deployed path. The model that would
have produced those fields is the SinBERT-gated corrector — **the negative
result**. Emitting plausible confidences would be inventing numbers, in the
one project whose strongest contribution is a carefully reported negative.

**Proposal.** L5 emits `corrected_text` plus a `tokens` array derived by
aligning `body_raw` against `body` with `difflib`, carrying only
`original`, `corrected`, `was_changed` — and a `token_source: "diff"` field
saying so. Nadee's `parse_ocr_input()` requires only `corrected_text` and
defaults `tokens` to `[]`, so nothing downstream breaks either way.

### 2.2 Three mutually incompatible dependency sets

| | numpy | transformers | other |
|---|---|---|---|
| `system/` (mine) | **1.26.4** | 5.1.0 | cv2 **4.9.0**, torch+cu121 |
| Bumal | **2.4.4** | **5.7.0** | torch 2.11.0, river, tinydb, Ollama |
| Nadee | langchain stack | — | chromadb, langchain-openai, network |
| Janindu | own `venv311` + paddle venv | — | paddleocr, skimage |

The pinned cv2 4.9.0 / numpy 1.26.4 is **not** a preference — it is what the
deskew reproducibility finding was measured under, and Chapter 4 cites the
0.0730 vs 0.0615 library-version result. A single shared venv puts a reported
result at risk.

**Conclusion: components talk over HTTP, not by import.** Each runs in its own
venv as its own process. This also means a teammate's broken component
degrades one feature instead of killing the viva demo.

### 2.3 The RAG component cannot start

- `ingest.py` reads `corpus/articles.jsonl` — **that file does not exist**
  anywhere in the repo.
- No `chroma_db/` has ever been built.
- `generate.py` calls `ChatOpenAI(model="gpt-5.4-mini")` — needs a key and
  live internet **at demo time**.

### 2.4 The STT component is not importable

`Work/Bumal/.../stt/stt.py` is a Colab export: `from google.colab import drive`,
`!pip install`, `files.upload()`. It cannot be imported by a server. The model
itself (`L-Inuri/Wav2Vec-BERT`) is real; the wrapper is not.

`intent_detection/approach1_nllb_llm.py` loads NLLB-600M at import and calls a
local **Ollama** server for Llama 3.2 1B. Two more models, one more daemon.

### 2.5 L4A is delivered as files, not as an integration

`system/layers/l4a_title/` already contains `title_extractor_p1..p4.py` and
`tessdata/sin_custom.traineddata` (7.7 MB) — **nothing imports them**.

They also expect a different shape: p1 stitches from
`<crop_root>/<stem>/<block_id>/line_*.png` on disk; p4 is a CLI that scans a
`Processes/` tree and writes `frontend_summary.json`. The L4A contract is
`extract(img, article) -> article`, in memory, with a title `Region` box in
full-frame coordinates.

The adapter is real but small: **crop the title region -> line-split ->
stitch (p1) -> binarize (p2) -> MAT (p3) -> Tesseract `sin_custom`**. Stage 1
(full-page Paddle layout) is skipped, because L3 already did that job.

Their own README warns: MAT skeletonisation can destroy *pilla*. **Measure MAT
against plain Tesseract at the optimal scale before adopting it.**

### 2.6 Smaller things

- `Work/Janindu/1_Preprocess/MainPreProcess.py` hardcodes
  `E:\Sliit\Research\Main Repository\R26-DS-002\...` — not a path on this machine.
- `README.md` says L5 = Nadee, L6 = Bumal, L4A = Janindu.
  `Handoff_26Aug2026.md` says components 3 and 4 are both Bumal's and L4A is
  "another member". The folders say Nadee = RAG, Bumal = voice. **Three
  documents, three ownership maps.** Fix before writing Chapter 3.
- `layers/l6_generator/` and `layers/l7_speech/` are empty `__init__.py` —
  an intended split that was never carried out.
- On-device Android TTS already works and is the better choice (build record
  §14). `l6_speech.speak() -> None` should stay a null adapter, not become a
  server TTS.

---

## 3. Proposed architecture

One gateway, three sidecar services, one session store.

```
                       ┌──────────────── system/app/server.py (my venv) ────────────┐
phone                  │                                                            │
 │  POST /capture      │  L2 select → L3 segment → L4A title → L4B body → L5 assemble│
 │  (5 frames)     ───▶│                              │                             │
 │                     │                              └──HTTP──▶ svc-title (Janindu)│
 │  ◀── {job, title,   │                                                            │
 │       body, warn}   │  SESSIONS[job] = Document          (dict + TTL, one user)  │
 │                     │                                                            │
 │  POST /ask          │  ──HTTP──▶ svc-voice (Bumal)  → route/intent/style          │
 │  {job, text|audio}──▶│  ──HTTP──▶ svc-rag   (Nadee)  → answer_si                  │
 │  ◀── {speakable}    │                                                            │
 └─ phone speaks it    └────────────────────────────────────────────────────────────┘
    with GuidanceSpeaker.readAloud()
```

**Layer numbering made honest:**

| layer | is | where |
|---|---|---|
| L1 | capture + guidance + **voice input** | phone |
| L2–L5 | unchanged | in-process, my venv |
| L4A | title OCR adapter | HTTP to svc-title |
| **L6** | generator / RAG | HTTP to svc-rag |
| **L7** | speech | on-device TTS; `audio_url` stays null |
| **L0** | voice interaction (STT, intent, personalization) | HTTP to svc-voice |

`core/schemas.py` gains **one** model — a `Question`/`Answer` pair for `/ask`.
Nothing existing changes shape. That is the point of having had a contract.

---

## 4. Order of work

Each step ends with something runnable and verifiable.

**Step 0 — freeze the contracts (half a day, no code).**
Write `INTEGRATION_CONTRACT.md`: `/capture`, `/ask`, the three service APIs,
the L5 payload with `token_source: "diff"`. Circulate. Nobody codes until the
three others have agreed, or this gets rebuilt three times.

**Step 1 — the loop, with stub brains (1 day).**
`/ask` endpoint + session store + `RECORD_AUDIO` + a physical trigger on the
phone (long-press or volume key — a blind user cannot find a button) + two
services that return canned JSON. **This is the step that de-risks the demo.**
After it, the system is conversational end to end and every later step is a
drop-in replacement.

**Step 2 — svc-rag (1 day).** FastAPI wrapper around Nadee's `run_pipeline`
in its own venv. Blocked on: the `articles.jsonl` corpus and an API key.
Add a hardcoded offline fallback answer so a dead network does not kill a viva.

**Step 3 — svc-voice (1–2 days).** Wrapper around `handle_voice_command`.
STT is the real decision — see §5.

**Step 4 — svc-title (1 day).** Janindu's stages 2–5 on the title region.
**Measure MAT vs plain Tesseract first**, then wire it behind a config flag.

**Step 5 — one integration test per boundary**, and the end-to-end run.

**Everything behind a flag, defaulting off.** The reading path must still work
with all three services down. It does today; it must still do so in October.

---

## 5. The honest caution

This is roughly **one week of integration work that produces no new Chapter 4
result.** `Handoff_26Aug2026.md` §10 says the chapters are the only real risk
and that a thesis with an unfinished chapter fails while one without another
experiment does not. That still holds.

So: **Step 0 and Step 1 are worth doing** — they are what turns four folders
into a system, they are cheap, and they are demonstrable. Steps 2–5 should be
fitted around chapter writing as teammates deliver, not before it.

---

## 6. Decisions I cannot make alone

1. **How much time is integration allowed to take** against Chapters 3/4/5.
2. **Ownership of L4A, L5, L6** — three documents currently disagree.
3. **STT: on-device or server?** Android `SpeechRecognizer` with `si-LK` is
   lower latency, needs no server model, and removes a whole failure mode —
   but it means Bumal's Wav2Vec2-BERT is reported as researched rather than
   deployed. That is partly a team-political call, not a technical one.
4. **Where the RAG corpus comes from**, and whether an LLM API call is
   acceptable in the viva demo.
5. **Whether Janindu's MAT title path is adopted at all**, pending the
   measurement his own README asks for.
