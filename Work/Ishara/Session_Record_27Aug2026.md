# Session record — 27 August 2026

**Integration step 1 complete: the conversation path exists, server and phone.** Nothing changes the reading path or any number Chapter 4 cites.

Full copy at `E:\RP\R26-DS-002\Work\Ishara\Session_Record_27Aug2026.md`. Frozen contract: `system\docs\INTEGRATION_CONTRACT.md`.

## The problem

`/capture` read an article, sent the text, and forgot it — so a follow-up question had nothing to refer to. Components 3 and 4 had no way in: no `RECORD_AUDIO`, no microphone in `MainActivity.kt`, no endpoint taking a question.

## Backend (built, tested)

| file | what |
|---|---|
| `core/session.py` | NEW — job id → last Document; in memory, TTL 1800s, 32 entries, injectable clock so expiry is tested not slept through |
| `core/svc.py` | NEW — one JSON POST with timeout that never raises; `urllib`, no new dependency |
| `core/schemas.py` | ADDITIVE — `Question`, `Answer` |
| `core/config.py` | ADDITIVE — INTEGRATION section; every switch defaults to today's behaviour |
| `layers/l0_voice/voice.py` | NEW — Component 4 client (`stub` \| `http`) |
| `layers/l6_generator/generate.py` | NEW — Component 3 client (`off` \| `http`) + local intents |
| `layers/l5_assemble/payload.py` | NEW — the payload Component 3 consumes |
| `app/server.py` | `POST /ask`, `GET /session/{job}`, capture now remembered. Backup `server.py.bak_27aug` |
| `tools/stub_services.py` | NEW — stand-ins for both components, stdlib only, `--echo` prints the exact payload |
| `docs/INTEGRATION_CONTRACT.md` | NEW — frozen contract for all four members |

**Tests 115 → 157**, all passing.

## Phone (built, type-checks, not yet run on a device)

`AndroidManifest.xml` — `RECORD_AUDIO` + a `<queries>` entry for `android.speech.RecognitionService` (API 30+ package visibility; the test device runs Android 11). `ReaderApi.kt` — `ask(@Body RequestBody)` + `Backend.jsonBody()`, **no new Gradle dependency**. `QuestionListener.kt` NEW — one utterance, si-LK, exactly-once callback. `GuidanceSpeaker.kt` — `sayThen()` so the mic never opens while the phone is still speaking. `MainActivity.kt` — keeps the job; **volume-down** asks a question. All backed up as `.bak_27aug`.

**Volume-down, not a screen button** — a blind user cannot find a target they cannot see. Cost stated: volume-down no longer lowers volume while the app is in front.

**A bug caught before delivery:** `stopReading()` delivers its callback via `Handler.post`, so `endCycle()` had not run when `startQuestion()` continued — setting `busy = true` would have been undone, re-arming the auto-shutter mid-question. The rest is now posted behind it in the same queue.

**Verification honesty:** all eight Kotlin files compile clean against faithful stubs of the Android/AndroidX/CameraX/Retrofit/OkHttp surface plus the real coroutines library. That catches syntax, resolution and type errors. It is not a Gradle build and tests no behaviour.

## Three decisions worth defending

**No `label`/`confidence` in the RAG payload.** `TempFormatPleaseRead.txt` and Nadee's `contracts.py` both specify them; L4B cannot produce them — plain full-sequence mT5 has no per-token classifier, and the model that would (SinBERT-gated) is the negative result. `tokens` is a difflib word diff with `token_source: "diff"`. `tests/test_rag_payload.py` fails if either field is added. Verified against Nadee's own `SAMPLE_OCR_INPUT`: the diff finds exactly her documented corrections (ආක්‍රමණීකයන් → ආක්‍රමණිකයන්, රදල → රදළ).

**HTTP between components, not imports.** numpy 1.26.4/transformers 5.1.0 here vs 2.4.4/5.7.0 for Bumal; the cv2 4.9.0 pin is what the deskew reproducibility finding and the CER 0.0730 vs 0.0615 library-version result were measured under. Also: a down component degrades one feature instead of killing the demo.

**Failure is a sentence, never a stack trace.** voice down → stub fallback + warning, continue. rag down/timeout/empty → `ok:false`, speakable message, `answer_si` stays empty — nothing invented. Job not in session → 404 with something to say. `tests/test_services_http.py` runs a real socket server over every row of that table.

## Works today with nothing else running

Capture → article held; "නැවත කියවන්න" (read again) answered from session with no network — the correct implementation, not a stub; "නවත්වන්න" (stop) returns empty `speakable` so the phone acts; a real question gets an honest "service not available", never a fabricated answer. Verified live over real HTTP with `tools/stub_services.py`.

## Honest limits

- The phone code type-checks but has not been built or run.
- Check whether a Sinhala model is installed for *recognition* — TTS having a Sinhala voice does not imply the recogniser does. Without it the platform falls back to device locale and the question arrives in English (not fatal — Component 4 translates to English anyway).
- Nothing routed through a stub is a result — both stubs stamp `source: "stub"` and `/ask` copies it into `warnings`.
- The Sinhala strings were not written by a native speaker; two are lifted verbatim from Nadee's source. Have all four checked.
- Session memory is not durable — a restart costs one re-capture.
- Component 3 still needs `corpus/articles.jsonl` (absent from the repo) and an API key with live network.
- `TITLE_MODE` stays `stub`: MAT still unmeasured against plain Tesseract.

## Next

1. **Build the app and try it** — the only remaining unknown in the loop itself.
2. svc-rag wrapper (~3 lines around `run_pipeline`), blocked on the corpus.
3. svc-voice wrapper around `handle_voice_command`; STT decision still open.
4. svc-title after MAT is measured.

Chapters 3, 4 and 5 remain the only real risk.
