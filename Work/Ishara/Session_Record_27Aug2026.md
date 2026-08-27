# Session record — 27 August 2026

**Integration step 1: the conversation path exists, server side.**

Everything below is built, runnable and tested. Nothing in it changes the
reading path or any number Chapter 4 cites.

---

## 1. What was wrong

`/capture` read an article, sent the text to the phone, and **forgot it**.
That one fact is why a follow-up question was impossible: by the time the user
says *"summarise that"*, nothing anywhere knew what *that* was.

Components 3 and 4 were four folders of working code with no way in:

- `AndroidManifest.xml` asks for CAMERA and INTERNET only — no `RECORD_AUDIO`
- `MainActivity.kt` has no microphone
- the server had no endpoint that takes a question

## 2. What was built

| file | what |
|---|---|
| `core/session.py` | **NEW** — job id → the last Document. In memory, TTL 30 min, 32 entries. Injectable clock so expiry is tested, not slept through |
| `core/svc.py` | **NEW** — one JSON POST with a timeout that never raises. `urllib`, stdlib, **no new dependency** |
| `core/schemas.py` | ADDITIVE — `Question`, `Answer`. Nothing above them changed |
| `core/config.py` | ADDITIVE — an INTEGRATION section, every switch defaulting to today's behaviour |
| `layers/l0_voice/voice.py` | **NEW** — Component 4 client, `stub` \| `http` |
| `layers/l6_generator/generate.py` | **NEW** — Component 3 client, `off` \| `http`, plus local intents |
| `layers/l5_assemble/payload.py` | **NEW** — the payload Component 3 consumes |
| `app/server.py` | `POST /ask`, `GET /session/{job}`, and a successful capture is now remembered. Backup: `server.py.bak_27aug` |
| `tools/stub_services.py` | **NEW** — stand-ins for both components so the `http` path can be run today |
| `docs/INTEGRATION_CONTRACT.md` | **NEW** — the frozen contract, for all four of us |

**Tests: 115 → 157.** 42 new, all passing, plus the original 115 unchanged.

## 3. Three decisions worth defending

### 3.1 No `label`, no `confidence` in the RAG payload

`TempFormatPleaseRead.txt` and Nadee's `contracts.py` both specify per-token
`label` and `confidence`. **Layer 4B cannot produce them.** The deployed
corrector is plain full-sequence mT5; there is no per-token classifier. The
model that would have had one is the **SinBERT-gated corrector — the negative
result**.

Filling those fields would be generating numbers with nothing behind them, in
the project whose strongest contribution is a carefully reported negative.

`tokens` is a word-level diff of `body_raw` against `body`, and
`token_source: "diff"` says so. `tests/test_rag_payload.py` **fails if either
field is ever added.**

Checked against Nadee's own `SAMPLE_OCR_INPUT`: the diff finds exactly the
corrections her sample documents (ආක්‍රමණීකයන් → ආක්‍රමණිකයන්, රදල → රදළ).
Her `parse_ocr_input()` needs only `corrected_text`, so this is safe either way.

### 3.2 HTTP between components, not imports

Not a style choice. The four dependency sets cannot coexist:

| | numpy | transformers |
|---|---|---|
| this system | **1.26.4** | 5.1.0 |
| Bumal | **2.4.4** | **5.7.0** |

and the cv2 4.9.0 / numpy 1.26.4 pin is **what the deskew reproducibility
finding and the CER 0.0730 vs 0.0615 library-version result were measured
under**. A shared venv would risk a reported result to save a subprocess.

Second reason, which matters on the day: a component that is down degrades one
feature instead of killing the demo.

### 3.3 Failure is a sentence, never a stack trace

Every call out of the server is on the path to a blind user's ear.

- svc-voice down → fall back to stub routing, warn, **continue**
- svc-rag down, timed out, or returning an empty answer → `ok:false`,
  a speakable "could not get an answer", and `answer_si` stays **empty**
- job not in session → 404, still with something to say

`answer_si` is never filled on a failure path. `tests/test_services_http.py`
runs a real socket server and exercises every row of that table.

## 4. The phone side — also built

| file | what | backup |
|---|---|---|
| `AndroidManifest.xml` | `RECORD_AUDIO` + a `<queries>` entry for `android.speech.RecognitionService` | `.bak_27aug` |
| `ReaderApi.kt` | `ask(@Body RequestBody)` + `Backend.jsonBody()` — **no new Gradle dependency** | `.bak_27aug` |
| `QuestionListener.kt` | **NEW** — one utterance, `si-LK`, exactly-once callback | — |
| `GuidanceSpeaker.kt` | `sayThen()`; `readAloud()` gained a defaulted parameter, existing call sites unchanged | `.bak_27aug` |
| `MainActivity.kt` | keeps the `job`; **volume-down** asks a question; speaks `speakable` | `.bak_27aug` |

**Volume-down, not a screen button.** A blind user cannot find a target they
cannot see — the whole capture flow exists to avoid asking them to. The cost,
stated: volume-down no longer lowers the volume while the app is in front.

**One bug I wrote and caught before you saw it.** `stopReading()` delivers the
article's callback through `Handler.post`, so `endCycle()` has not run when
the call returns. Setting `busy = true` straight afterwards would have been
undone by that queued `endCycle()`, re-arming the auto-shutter *in the middle
of a question*. `startQuestion()` now posts the rest behind it in the same
queue.

### How far the Kotlin was verified

All eight source files **type-check clean** — compiled against faithful stubs
of the Android, AndroidX, CameraX, Retrofit and OkHttp surface they use, with
the real coroutines library. That catches syntax, resolution and type errors.

It is **not** a Gradle build against the real SDK and it tests no behaviour.
Two things to check on the phone:

1. Whether a Sinhala model is installed for *recognition* — TTS having a
   Sinhala voice does not imply the recogniser does. Without it the platform
   falls back to the device locale and the question arrives in English. Not
   fatal (Component 4 translates to English anyway) but it should be a known
   state, not a surprise.
2. If `isRecognitionAvailable()` returns false on Android 11, suspect the
   `<queries>` entry first.

## 5. What works today with nothing else running

Both services off, no network:

- capture → the article is held in session
- *"නැවත කියවන්න"* (read again) → answered from session. **This is not a
  stub** — it is the correct implementation, it needs no service, and it means
  the most common follow-up survives a RAG outage at the viva
- *"නවත්වන්න"* (stop) → `speakable: ""`; the phone acts rather than speaks
- a real question → an honest *"the answering service is not available"*,
  never a fabricated answer

With `tools/stub_services.py` running, the same loop runs over real HTTP
end to end. Verified live, not just unit-tested.

## 5b. The local command set

"Read it again" alone is too thin to demonstrate. Nine commands now answer
**from the article in session** — no network, no API key, no teammate:

| say | intent | what comes back |
|---|---|---|
| `නැවත කියවන්න` | REPEAT | the whole article |
| `මුල සිට කියවන්න` | FIRST | part 1 |
| `ඊළඟ` | NEXT | the next part — walks a long article |
| `කලින් එක` | PREVIOUS | the part before |
| `වචන කීයද` | LENGTH | *"this article has 184 words"* |
| `ශීර්ෂය මොකක්ද` | TITLE | the headline, or an honest *"not read yet"* |
| `මොනවද මඟ හැරුණේ` | WARNINGS | what the capture skipped |
| `නවත්වන්න` | STOP | empty `speakable` — the phone acts |

**These are not stubs.** Each is the correct implementation and none of them
can break because a service is down.

`NEXT` is the one that changes the experience: an article is 2–3 thousand
characters, and a listener who wants the middle should not sit through the
start. Parts are ≤400 characters, split on sentence boundaries. The cursor
lives in the session beside the Document so it expires with the text it
indexes into.

**The phone needed no change** — all of them return `speakable`, which
`MainActivity` already speaks.

Every command has an English alias, so testing never waits on the Sinhala
being right. **Have the Sinhala checked** — a command that will not trigger is
a vocabulary problem before it is a code problem.

## 6. Honest limits

- **The phone code has not been built or run.** It type-checks; that is not
  the same thing. Build it in Android Studio before believing any of it.
- **Nothing routed through a stub is a result.** The built-in stub and
  `stub_services.py` both stamp `source: "stub"`, and `/ask` copies it into
  `warnings`, so no transcript can be read as a Component 3 or 4 measurement.
- **The Sinhala strings in `l6_generator` and `server.py` were not written by
  a native speaker.** Two are lifted verbatim from Nadee's own source. Have
  all four checked before the viva.
- **Session memory is not durable.** A restart loses the last article, costing
  one re-capture. Stated as a limitation, not papered over.
- Component 3 still needs `corpus/articles.jsonl`, which does not exist in the
  repo, and an API key with live network at demo time.
- `TITLE_MODE` stays `stub`: MAT has still not been measured against plain
  Tesseract, which `l4a_title/README.md` asks for.

## 7. Verify it yourself

```
cd E:\RP\R26-DS-002\system
python -m pytest tests -q                      # expect 180 passed
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2

# then, in another window — no phone needed:
python tools\try_ask.py --frames work\<jobid>
```

Or open `http://127.0.0.1:8000/debug`: upload frames, see raw OCR against
corrected text, then ask a question from the preset buttons or the text box.

**That page was broken before today** — it read a `document` key that
`/capture` stopped returning on 21 Aug, so every upload showed "no articles".
It now fetches `GET /document/{job}`.

`python tools\stub_services.py --role rag --port 8102 --echo` prints the exact
payload Component 3 will receive — worth sending Nadee before she writes a
line of wrapper.

**Tests are now 180**: `/document`, two that stop a stub service being
reported as the real component, and `test_local_commands.py`.

## 8. Next

1. **Build the app and try it.** That is the only remaining unknown in the
   loop itself.
2. svc-rag wrapper — three lines around `run_pipeline`, blocked on the corpus.
3. svc-voice wrapper — around `handle_voice_command`; STT decision still open.
4. svc-title — after MAT is measured.

**Chapters 3, 4 and 5 remain the only real risk.** Nothing above changes that.
