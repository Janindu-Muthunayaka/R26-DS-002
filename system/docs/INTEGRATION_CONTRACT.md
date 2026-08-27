# Integration contract — R26-DS-002

**Frozen 27 August 2026. Change it only by telling the other three people
first, the same rule `core/schemas.py` has always had.**

Four components, four Python environments, one system. This file is what each
of them may assume about the others.

---

## 0. The shape of the system

```
READING PATH  (exists, measured, unchanged by any of this)

  phone ──POST /capture (3 frames)──▶  L2 select
                                       L3 segment / layout
                                       L4A title      ← Component 1, Janindu
                                       L4B body + mT5 ← Component 2, Ishara
                                       L5 assemble
        ◀── {job, title, body, warnings} ──┘
  phone speaks it with on-device TTS

                                       └─▶ SESSION[job] = Document   ← NEW

CONVERSATION PATH  (new, both services default OFF)

  phone ──POST /ask {job, text}──────▶  L0 voice     ← Component 4, Bumal
                                        L6 generator ← Component 3, Nadee
        ◀── {speakable, ...} ──────────┘
  phone speaks it with the same TTS
```

**The job id is the only thing that links the two.** `/capture` already minted
and returned it; the phone now keeps it and sends it back.

---

## 1. `POST /capture` — UNCHANGED

Request: multipart, every frame under the field name `frames`.

```json
{"ok": true, "job": "a1b2c3d4", "title": "", "body": "…",
 "warnings": ["…"], "n_articles": 1, "audio_url": null, "timings": {}}
```

Failure: HTTP 4xx/5xx, same field names, plus `"error"`.
`tests/test_api.py` pins every field name by name. Do not rename one.

The only change on this route: a successful capture is now also stored in
session. `/capture` returns exactly what it returned before.

---

## 2. `POST /ask` — NEW

```json
  request   {"job": "a1b2c3d4", "text": "<Sinhala question>", "user_id": null}

  reply     {"ok": true, "job": "a1b2c3d4",
             "route": "GENERATE" | "LOCAL" | "TTS_REPLAY",
             "intent": "SUMMARIZE",
             "speakable": "…",        ← the phone speaks THIS
             "answer_si": "…",
             "sources": [ {...} ],
             "warnings": ["…"],
             "timings": {"voice": 0.4, "generate": 2.1, "total": 2.5}}
```

**THE RULE FOR THE PHONE: speak `speakable` whenever it is non-empty,
whatever `ok` says.** `ok` records whether an answer was actually generated;
it drives logging, not speech. A blind user must hear something, and *"the
answering service is not available"* is something. Silence is the one
unacceptable outcome.

The single exception is `intent: "STOP"`, which returns `speakable: ""` on
purpose — there the phone **acts** (stop speaking) rather than speaks.

`404` means the job is not in session: it expired, or the server restarted, or
nothing was ever read. All three get the same reply, because the user's next
action is the same in all three cases: capture again. `speakable` is filled
even on the 404.

`/ask` never returns 500 because a component is down. See §5.

`GET /session/{job}` reports whether an article is held, without re-running
OCR. Diagnostics only.

---

## 3. Service contracts

Each component runs as **its own process, in its own venv**. Nobody imports
anybody. The reason is in `core/config.py`: the four dependency sets cannot
coexist, and this system's numpy/cv2 pin is what a reported Chapter 4 result
was measured under.

### 3.1 svc-voice — Component 4 (Bumal) · default port 8101

```
POST /interpret   {"text": "<Sinhala>", "user_id": "user_001",
                   "retrieved_chunk_id": null}

     -> exactly what handle_voice_command() already returns:
        {"route", "intent", "english_translation", "style_class",
         "prompt_modifier", "personalization_flags",
         "retrieved_chunk_id", "correction_applied"}
```

The six names in `REQUIRED` (`layers/l0_voice/voice.py`) are checked on
arrival. A reply missing one is rejected **at this boundary** and named in the
warnings, rather than raising two layers later inside Nadee's
`parse_voice_input`.

Nothing else changes in Component 4. This is a wrapper around a function that
already exists and already returns this shape.

### 3.2 svc-rag — Component 3 · BUILT, `services/rag/` · default port 8102

```
POST /answer   {"ocr":   <the Layer 5 payload, §4>,
                "voice": <exactly what svc-voice returned>}

     -> {"ok": true|false,          <- ADDITIVE, and it matters (below)
         "intent", "answer_si", "retrieved_sources", "speakable_text",
         "notes": ["..."]}          <- diagnostics, surfaced as warnings
```

Also `GET /` (a status page, so a browser GET never looks like a fault),
`GET /health`, `GET /stats`, `POST /ingest`, `POST /forget`.

**`ok` is not decoration.** Component 3 answers a failure with a SENTENCE —
a perfectly non-empty string the user can hear. Without the flag, every
failure came back to the reader looking exactly like a successful answer and
was reported `ok: true`. That was found in a live run, not in review.

**The corpus problem is solved by indexing what is read.**
`Work/Nadee/ingest.py` reads `corpus/articles.jsonl`, which has never existed
in this repository — the reason Component 3 had never run end to end. The
service now stores every article the reader captures (`source_type: "read"`),
so the corpus builds itself out of use and waits on nobody. A real corpus, if
one ever arrives, drops in with `--seed <folder>` in the shape her ingest
expected. `POST /forget {"source_type":"read"}` clears it.

**What is Nadee's and is kept verbatim:** the Sinhala prompt including its
"answer only from the evidence" instruction and its "not enough information"
fallback, the Sinhala-purity retry, the style/detail word-limit tables, the
chunk metadata, and the "always retrieve the current page" rule.
`Work/Nadee/` is untouched. What changed is underneath: chroma + langchain +
sentence-transformers became a numpy array and API embeddings — three
packages instead of a torch stack, and an on-disk format you can open and
read when a retrieval looks wrong. `services/rag/store.py` argues it.

Still needed: `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL` in
the repository-root `.env`, and network at demo time. §10.

### 3.3 svc-title — Component 1 (Janindu) · not yet wired

`TITLE_MODE` stays `stub` until MAT is measured against plain Tesseract at the
optimal scale, which `layers/l4a_title/README.md` asks for and nobody has
done. Sinhala dependent vowel signs are a few pixels wide and skeletonisation
can remove them. Wiring an unmeasured OCR path into a system whose whole claim
is measured numbers is the wrong trade.

---

## 4. The Layer 5 → Component 3 payload

```json
{"corrected_text": "…",
 "tokens": [{"original": "ආක්‍රමණීකයන්",
             "corrected": "ආක්‍රමණිකයන්",
             "was_changed": true}],
 "token_source": "diff",
 "articles": [{"index", "title", "body", "polished",
               "glyph_p75", "ocr_scale", "verdict"}],
 "warnings": ["…"]}
```

### There is no `label` and no `confidence`, and that is deliberate

`TempFormatPleaseRead.txt` and Nadee's `contracts.py` both specify them.
**Layer 4B cannot produce them.** The deployed corrector is plain
full-sequence mT5-small: it emits a corrected string, and there is no
per-token classifier anywhere in the reading path. The model that *would* have
produced a label and a confidence per token is the SinBERT-gated span
corrector — which underperformed plain mT5 and is the project's reported
negative result.

Filling those two fields would mean generating numbers with nothing behind
them, in the one project whose strongest contribution is a carefully reported
negative.

`tokens` is instead a word-level diff of `body_raw` against `body`, and
`token_source: "diff"` says so. The alignment is approximate — `correct()`
splits into sentences, corrects each, joins, and dedups, so the two sides need
not have matching token counts. Read `was_changed` as *"these strings differ
here"*, not *"the model decided to change this token"*.

`tests/test_rag_payload.py` fails if `label` or `confidence` is ever added.
That is the point of the file.

Component 3 is unaffected either way: `parse_ocr_input` requires only
`corrected_text` and defaults `tokens` to `[]`.

---

## 5. Failure is a sentence, never a stack trace

Every call out of this server is on the path to a blind user's ear.

| what went wrong | what happens |
|---|---|
| svc-voice unreachable / 500 / bad JSON | fall back to the built-in stub routing, warn, **continue** |
| svc-voice reply missing a required field | same, and the missing field is named |
| svc-rag unreachable / timeout / 500 | `ok:false`, `speakable` = "could not get an answer, please try again", `answer_si` stays **empty** |
| svc-rag returns an empty answer | same — an empty answer is a failure, not silence |
| job not in session | 404 with a speakable sentence |
| anything else | 500 with a speakable sentence |

`answer_si` is never filled on a failure path. Nothing is invented.

`tests/test_services_http.py` runs a real socket server and exercises every
row of that table.

---

## 6. Switches — all default to today's behaviour

`core/config.py`. With the defaults, `/capture` is byte-identical in behaviour
and `/ask` still answers "read that again" from session with no network.

| variable | default | other |
|---|---|---|
| `SINHALA_VOICE_MODE` | `stub` | `http` |
| `SINHALA_VOICE_URL` | `http://127.0.0.1:8101` | |
| `SINHALA_RAG_MODE` | `off` | `http` |
| `SINHALA_RAG_URL` | `http://127.0.0.1:8102` | |
| `SINHALA_TITLE_MODE` | `stub` | `mat` (not implemented) |
| `SINHALA_SESSION_TTL` | `1800` | seconds |
| `SINHALA_SESSION_MAX` | `32` | articles |
| `SINHALA_USER_ID` | `user_001` | matches Bumal's tinydb profile |

Neither session number is measured. Both are design choices, stated as such.

---

## 7. Testing it

```
cd E:\RP\R26-DS-002\system
python -m pytest tests -q                       # expect 225 passed
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

### 7.1 Without a phone — `tools/try_ask.py`

Reads a page, then asks about it, printing route, intent, `ok`, `speakable`
and every warning:

```
python tools\try_ask.py --frames work\<jobid>
python tools\try_ask.py --job <jobid> --ask "මේක සාරාංශ කරන්න"
python tools\try_ask.py --job <jobid> --session
```

`--frames` takes a folder or a list of files; a folder is uploaded in sorted
order, which is the order `f0/f1/f2` were written.

### 7.2 Without a phone, in a browser — `/debug`

`http://127.0.0.1:8000/debug` uploads frames, shows raw OCR beside corrected
text and the per-stage timings, then offers four preset questions and a free
text box.

**Note for anyone who used this page before 27 Aug 2026:** it was broken. It
read a `document` key that `/capture` stopped returning on 21 Aug, so every
upload showed "no articles". It now fetches `GET /document/{job}` instead.

### 7.3 The local command vocabulary — works with every service OFF

These are **not stubs.** Each is the correct implementation, answered from the
article held in session, with no network, no API key and no teammate's
component. They are the part of the conversation that cannot break on the day.

| say | English alias | intent | what comes back |
|---|---|---|---|
| `නැවත කියවන්න` | again / repeat | REPEAT | the whole article |
| `මුල සිට කියවන්න` | from the start | FIRST | part 1 |
| `ඊළඟ` | next | NEXT | the next part — press repeatedly to walk the article |
| `කලින් එක` | back / previous | PREVIOUS | the part before |
| `වචන කීයද` | how long | LENGTH | *"this article has 184 words"* |
| `ශීර්ෂය මොකක්ද` | headline | TITLE | the headline, or an honest *"not read yet"* |
| `මොනවද මඟ හැරුණේ` | what did I miss | WARNINGS | what the capture skipped |
| `නවත්වන්න` | stop | STOP | **empty `speakable`** — the phone acts |
| anything else | — | ASK | needs Component 3; otherwise *"service not available"* |

**Why `NEXT` matters.** An article is two or three thousand characters. A
listener who wants the middle of it should not have to sit through the start.
`NEXT` walks the article in parts of at most `PART_MAX_CHARS` (400, a
readability choice, not a measurement), split on sentence boundaries so a part
never begins mid-sentence. The cursor lives in the session beside the
Document, so it expires with the article it indexes into.

**The phone needs no change for any of this.** Every one of them returns
`speakable`, which `MainActivity` already reads and speaks; `STOP` was already
handled. New commands are a server-side edit.

**The Sinhala was not written by a native speaker.** Every command has an
English alias, so testing is never blocked on getting the Sinhala right — but
have it checked, and treat a command that does not trigger as a vocabulary
problem before suspecting the code. The list is
`_COMMANDS` in `layers/l0_voice/voice.py`; it is ORDERED, because
`මුල සිට කියවන්න` contains `කියවන්න` and the wrong order collapses every
navigation command into "read it all again".

The last row is not a failure. It is the loop working: `answer_si` is empty,
nothing was invented, and the user still hears a sentence.

Once Component 4 is wrapped, this list stops being the vocabulary — its intent
model returns SUMMARIZE, EXPLAIN, SIMPLIFY, ELABORATE, REPHRASE and the rest,
and anything it routes as GENERATE goes to Component 3.

### 7.4 With stand-ins for the missing components

```
python tools\stub_services.py --role voice --port 8101
python tools\stub_services.py --role rag   --port 8102 --echo
set SINHALA_VOICE_MODE=http
set SINHALA_RAG_MODE=http
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

`--echo` prints every request body, so Nadee and Bumal can **see** the exact
payload their service will receive before writing a line of wrapper.

To check a stub is alive, open `http://127.0.0.1:8101/interpret` in a browser.
It answers a GET with a short status page. (Before 27 Aug 2026 it returned
`501 Unsupported method ('GET')` — which meant the service was **healthy** and
the endpoint POST-only, but read as a fault.)

Every answer that comes back this way carries `voice routing: stub-service`
and `rag: stub service, not Component 3` in its warnings. That is deliberate:
a stub service returns a perfectly valid Component 4 shape, and without those
warnings a test transcript would read as evidence that the real components
ran.

---

## 8. The phone side — BUILT 27 Aug 2026

| file | change | backup |
|---|---|---|
| `AndroidManifest.xml` | `RECORD_AUDIO`, and a `<queries>` entry for `android.speech.RecognitionService` — the same API 30+ package-visibility rule that the TTS entry already exists for. Without it `isRecognitionAvailable()` returns false on a phone that recognises speech perfectly well, and the test device (SM-A705FN) runs Android 11 | `.bak_27aug` |
| `ReaderApi.kt` | `ask(@Body RequestBody)` plus `Backend.jsonBody()`. Retrofit accepts a raw `RequestBody` with no converter installed, so this adds **no Gradle dependency** — the same reason `capture()` still returns `ResponseBody` | `.bak_27aug` |
| `QuestionListener.kt` | **NEW** — one utterance through `SpeechRecognizer`, `si-LK`, exactly-once callback | — |
| `GuidanceSpeaker.kt` | `sayThen()`, and `readAloud()` gained a defaulted `announceMissingVoice` parameter. Existing call sites unchanged | `.bak_27aug` |
| `MainActivity.kt` | keeps the `job`; **volume-down** starts a question; speaks `speakable`; `intent == "STOP"` calls `stopReading()` instead | `.bak_27aug` |

### Three things worth knowing before the build

**The trigger is the volume-down key, not a screen button.** A blind user
cannot find a target they cannot see — the whole capture flow was built to
avoid asking them to, which is why the shutter fires itself. A physical key is
findable by touch and works with the phone held against a newspaper. **The
cost, stated plainly: volume-down no longer lowers the volume while the
Activity is in front.** Volume-up and the notification shade still do.

**The prompt is spoken with `sayThen()`, not `say()`.** `say()` fires and
forgets, so the microphone would open while the phone was still talking and
the recogniser would transcribe the app's own voice. `sayThen()` is built on
`readAloud()`, which already guarantees its callback runs exactly once.

**There is an ordering trap in stopping a reading to ask a question.**
`stopReading()` delivers the article's callback through `Handler.post`, so
`endCycle()` has NOT run when the call returns. Setting `busy = true`
immediately would be undone by that queued `endCycle()`, re-arming the
auto-shutter in the middle of a question. `startQuestion()` therefore posts
the rest of the work behind it in the same queue.

Speech-to-text runs **on the phone**: lower latency, no audio on the network,
one fewer model in the demo's failure surface. If server-side STT is ever
wanted, add an `audio` field to `Question` — nothing else in this contract
changes.

### What has NOT been verified

The Kotlin **type-checks clean** — all eight source files compile against
faithful stubs of the Android, AndroidX, CameraX, Retrofit and OkHttp surface
they use, with the real coroutines library. That catches syntax, resolution
and type errors. It does **not** substitute for a Gradle build against the
real SDK, and it cannot test behaviour on a device.

Two things to check on the phone specifically:

1. **Is a Sinhala voice model installed for recognition?** Guidance TTS having
   a Sinhala voice does not imply the recogniser does. If `si-LK` is missing
   the platform falls back to the device locale, and the question arrives in
   English — not fatal, since Component 4 translates to English anyway, but it
   should be a known state rather than a surprise.
2. **The `<queries>` entry.** If `isRecognitionAvailable()` returns false on
   Android 11 with the entry present, that is the first thing to suspect.

---

## 9. Honest status, 27 August 2026

**Built and tested:** session memory, `/ask`, both service clients, both
failure paths, the Layer 5 payload, and stub services for the two components
that do not exist yet, and nine local commands answered from session. 180 tests pass. The phone side is written and
type-checks clean, but has not been built or run on a device.

**Not built:** svc-voice; svc-rag; svc-title.

**Not measured, and must not be reported as if it were:** anything routed
through the built-in voice stub or `tools/stub_services.py`. Both stamp
`source: "stub"` / `stub: true`, and `/ask` copies that into `warnings`, so no
transcript can be mistaken for a Component 3 or Component 4 result.


---

## 10. Secrets, the readability gate, and Layer 4C

### 10.1 The key never goes in a file that git can see

`OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` and `OPENAI_EMBED_MODEL` live in `.env`
at the repository root. `.env` is gitignored; `services/.env.example` shows
the shape with no values.

```
copy services\.env.example .env
python tools\check_llm.py            <- lists what the key can ACTUALLY reach
```

**No model name in this project is a guess.** `Work/Nadee/generate.py`
hardcodes `gpt-5.4-mini`; whether a key can reach that is not something source
code can know, so `check_llm.py` asks and you set `.env` from the answer.

**A key that reaches this repo, a chat window or a screenshot is compromised
and must be ROTATED, not deleted.** Git does not forget a committed blob, and
the remote is shared with three people.

### 10.2 The readability gate — `core/quality.py`, always on, no model

A capture that goes wrong does not fail. Tesseract returns *something*, mT5
corrects that something, and the phone reads it aloud in the same confident
voice it uses for real news. A sighted developer sees garbage on a screen; a
blind user hears fluent nonsense and cannot tell it from the article.

Four surface statistics, no model, no network: Sinhala ratio, short-token
ratio, undecodable characters per 1000, word count. Calibrated on this
project's own OCR outputs — `python tools/calibrate_quality.py` reproduces the
table, and `core/quality.py` records it honestly, including the finding that
on these files the continuous measures barely separate anything and `n_words`
is the measure doing the work.

**`fatal` separates SHORT from SHATTERED.** A six-word news brief is a real
thing a newspaper prints, and is read with a warning. Fragments, Latin where
Sinhala belongs, or undecodable bytes are replaced by a sentence asking for
another photograph — the only thing that actually fixes it. `/capture` now
also returns a `quality` block; the phone ignores it, the debug page shows it,
and a bug report needs it.

### 10.3 Layer 4C — LLM post-editing. OFF, and it must stay off for evaluation

`SINHALA_POLISH_MODE` = `off` (default) | `auto` | `on`.

Handing corrupt Sinhala to a general-purpose model and asking it to "fix" the
text is the most dangerous thing in this repository:

1. **It can invent the news.** A model given a shattered sentence does not
   return a shattered sentence. It returns a fluent one, with names, numbers
   and dates that were never on the page, and the phone reads it aloud in the
   same voice it uses for the real article.
2. **It destroys the measurement.** Chapter 4's CER is mT5's. If a general
   model rewrites that output, the thing measured is no longer the model the
   thesis is about.

So it is built to be distrusted:

- OFF by default. `tests/test_polish.py` asserts no evaluation tool can enable
  it and that `l4b_body/body.py` does not import it.
- It runs after mT5 and writes to `Article.body_polished`. **`body` — the
  research artifact — is never overwritten.** One function,
  `l5_assemble.payload.article_text()`, decides what is spoken.
- Four guards; a rewrite failing any of them is DISCARDED and the mT5 text is
  used: character similarity ≥ `POLISH_MIN_SIMILARITY` (0.75), length ratio in
  [0.70, 1.30], Sinhala ratio must not fall, word count must not grow by more
  than 20%.
- Every article it touches — **including a rejection** — carries a warning all
  the way to the phone.

**`auto` deliberately refuses `unreadable` text.** That is where a repair
would be most welcome and where invention is most likely: with little real
signal left, fluency is all the model has to go on. On unreadable text the
honest system says *"I could not read that, try again"*, which is also what
gets the user to take the better photograph.

### 10.4 Starting it all

```
python tools\run_all.py                 # reader + svc-rag
python tools\run_all.py --stubs         # reader + both stand-ins
python tools\run_all.py --polish auto   # and enable Layer 4C for a demo
```

Ctrl-C stops everything. Four processes in four environments is the cost of
§3; it is worth paying at runtime and is not worth paying at a viva with three
command windows to start in the right order while somebody watches.
