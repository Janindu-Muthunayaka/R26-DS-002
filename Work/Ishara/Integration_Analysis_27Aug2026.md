# Integration analysis — 27 August 2026

**R26-DS-002 · four components into one running system**

Full copy also at `E:\RP\R26-DS-002\Work\Ishara\Integration_Analysis_27Aug2026.md`.
Written after reading every source file in `system/`, `Work/*`, and `F:\App`.
Nothing here is measured — it is a reading of the code and a proposal.

## Headline finding

These are **two systems, not four components of one**.

- READING PATH — phone → `/capture` → L2 L3 L4A L4B L5 → text → phone TTS. Exists, works, 115 tests pass.
- CONVERSATION PATH — voice → STT → intent → personalization → RAG → answer → TTS. Four folders of code, zero connections.

The conversation path has no entry point: no `RECORD_AUDIO` permission, no mic in `MainActivity.kt`, no endpoint that takes a question, and nothing anywhere holds "the article just read".

## Component state

| | owner | state |
|---|---|---|
| L1 capture app | Ishara | working, recalibrated 26 Aug |
| L2 select / L3 segment / L4B body / L5 assemble | Ishara | working, tested |
| L4A title | Janindu | code copied into `layers/l4a_title/` + tessdata, **nothing imports it**; `title.py` still `return article` |
| L6 RAG | Nadee | runs standalone; **`corpus/articles.jsonl` does not exist**, no chroma_db, needs OpenAI key + live network |
| Voice (STT/intent/personalization) | Bumal | `stt.py` is a Colab export (`from google.colab import drive`) — not importable; intent approach 1 needs NLLB-600M + a local Ollama daemon |

## Blocking mismatches

1. **`tokens[]` cannot be honoured.** `TempFormatPleaseRead.txt` and Nadee's `contracts.py` specify per-token `label` + `confidence`. mT5 is full-sequence seq2seq — the model that would produce those is the SinBERT-gated corrector, i.e. the negative result. Faking confidences would be inventing numbers. Proposal: emit `corrected_text` + a difflib-derived `tokens` array with only `original`/`corrected`/`was_changed`, plus `token_source: "diff"`. Nadee's `parse_ocr_input()` only requires `corrected_text`.
2. **Three incompatible dependency sets.** system/ pins numpy 1.26.4 + cv2 4.9.0 (the library-version reproducibility finding depends on it); Bumal pins numpy 2.4.4 / transformers 5.7.0 / torch 2.11.0; Nadee is a langchain/chroma stack; Janindu has his own venv311 + paddle venv. → components must talk over **HTTP, not imports**.
3. **Ownership map disagrees across three documents.** `system/README.md` says L5=Nadee, L6=Bumal, L4A=Janindu. `Handoff_26Aug2026.md` says components 3+4 are both Bumal's, L4A "another member". Folders say Nadee=RAG, Bumal=voice. Fix before Chapter 3.
4. `layers/l6_generator/` and `l7_speech/` are empty placeholders for a split never carried out.
5. `Work/Janindu/1_Preprocess/MainPreProcess.py` hardcodes `E:\Sliit\Research\Main Repository\...` — not a path on this machine.

## Proposed architecture

One gateway (`app/server.py`, my venv, L2–L5 in-process) + three sidecar services over HTTP (svc-title, svc-rag, svc-voice) + one in-memory session store keyed by the `job` id `/capture` already mints. New `POST /ask {job, text|audio}` returns speakable text; the phone speaks it with the existing `GuidanceSpeaker.readAloud()`. `core/schemas.py` gains one Question/Answer pair; nothing existing changes shape. L7 speech stays on-device — `audio_url` remains null.

## Order of work

0. Freeze contracts in `INTEGRATION_CONTRACT.md` (half a day, no code).
1. **The loop with stub brains** (1 day): `/ask` + session store + `RECORD_AUDIO` + a physical trigger + canned service responses. This is what de-risks the demo; everything later is drop-in.
2. svc-rag (1 day) — blocked on the corpus file and an API key; needs an offline fallback answer.
3. svc-voice (1–2 days) — STT decision below.
4. svc-title (1 day) — Janindu's stages 2–5 on the title region; **measure MAT vs plain Tesseract first**, as his own README asks.
5. Integration tests per boundary.

Everything behind flags defaulting off — the reading path must still work with all three services down.

## Honest caution

This is ~1 week of work producing **no new Chapter 4 result**. Handoff §10 stands: the chapters are the only real risk. Steps 0–1 are cheap and turn four folders into a system; steps 2–5 should fit around chapter writing.

## Open decisions (need Ishara)

1. Time budget for integration vs Chapters 3/4/5.
2. Ownership of L4A / L5 / L6 — three documents disagree.
3. STT on-device (Android `SpeechRecognizer`, `si-LK`) vs Bumal's server-side Wav2Vec2-BERT. On-device is lower latency and removes a failure mode, but demotes Bumal's model to "researched, not deployed" — partly a team-political call.
4. Where the RAG corpus comes from; whether a live LLM API call is acceptable in the viva.
5. Whether the MAT title path is adopted at all, pending measurement.
