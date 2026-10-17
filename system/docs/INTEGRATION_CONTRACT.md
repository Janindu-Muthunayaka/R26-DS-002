# Integration contract — the four components

**Project R26-DS-002 · reader component (Ishara, IT22259134)**
Re-applied 28 Aug 2026 after the working tree was reset.

This is the agreement between the four components. If you change anything in
here, the other three break, so change it here first and tell them.

---

## 1. Why HTTP and not imports

The four components pin stacks that cannot live in one interpreter:

| component | owner | pins |
|---|---|---|
| reader (this one) | Ishara | numpy 1.26.4, cv2 4.9.0, transformers 5.1.0, torch cu121 |
| voice / personalisation | Bumal | numpy 2.4.4, transformers 5.7.0, torch 2.11.0, river, Ollama |
| RAG / generation | Nadee | langchain, chromadb, langchain-openai |
| Sinhala OCR models | Janindu | its own venv311, plus a separate paddle venv |

The `numpy 1.26.4` / `cv2 4.9.0` pin on this side is **not** arbitrary: the
deskew reproducibility finding and the library-version result (same
checkpoint, CER 0.0730 vs 0.0615) were measured under it, and Chapter 4 cites
them. Importing another component's code would change that environment and
invalidate a reported number.

The second reason matters on the day of the viva: **a component that is down
degrades one feature instead of killing the demo.**

---

## 2. The contract

`system/core/schemas.py` is the only definition. Pydantic models, one file,
imported by every layer on this side. If a field name changes there and not
in `MainActivity.kt`, `tests/test_api.py` fails — that is what it is for.

### `POST /capture` → the phone

multipart, one or more `frames`.

```json
{ "ok": true, "job": "b973c97b", "title": "...", "body": "...",
  "warnings": ["..."], "n_articles": 1, "audio_url": null,
  "timings": {"...": 0.0}, "quality": {"...": 0.0} }
```

`ok`, `job`, `title`, `body`, `warnings`, `n_articles`, `audio_url` are read
**by name** by the Android app. `timings` and `quality` are diagnostics; the
phone ignores fields it does not read.

`ok: false` still carries every one of those fields. A blind user can only be
told what went wrong if the error arrives in the fields the app already reads.

### `POST /ask` → the phone

```json
{ "job": "b973c97b", "text": "නැවත කියවන්න", "user_id": null }
```
```json
{ "ok": true, "job": "...", "route": "LOCAL", "intent": "REPEAT",
  "speakable": "...", "answer_si": "...", "sources": [], "warnings": [],
  "timings": {}, "error": null }
```

**The phone speaks `speakable` whenever it is non-empty, whatever `ok` says.**
`ok` records whether an answer was *generated*. Silence is the one
unacceptable outcome, and "the answering service is not available" is
something to hear.

### `POST /interpret` → Component 4 (Bumal)

Required keys in the reply, checked in `layers/l0_voice/voice.py`:
`route`, `intent`, `english_translation`, `style_class`, `prompt_modifier`,
`personalization_flags`. A reply missing any of them is refused and the
keyword stub answers instead — half-using it fails two layers later inside
`parse_voice_input`.

A reply carrying `"stub": true` is stamped `source: 'stub-service'`, never
`'component4'`.

### `POST /answer` → Component 3 (Nadee)

Request: `{"ocr": <rag_payload>, "voice": <the dict above>}`.
Reply: `ok`, `intent`, `answer_si`, `retrieved_sources`, `speakable_text`,
`notes`.

`ok` is **additive** and it matters. Component 3 signals failure with
`ok: false` and still fills `speakable_text` with a sentence the user can
hear. Without the flag, a failure sentence — a perfectly non-empty string —
came back looking exactly like a successful answer. That happened; see
`tests/test_services_http.py`.

### `rag_payload` — what leaves this component

`layers/l5_assemble/payload.py`. It carries `corrected_text`, the raw text,
and the changed tokens with `token_source: 'diff'`.

**It carries no label and no confidence.** The reported contribution is a
corrected-text improvement (CER 0.1197 → 0.0757, WER 0.3358 → 0.1640,
n = 217 page-disjoint). The SinBERT-gated span corrector is the reported
**negative result**. Sending a per-token label or a confidence would put a
number in front of Component 3 that the research does not support. Asserted
in `tests/test_rag_payload.py` and `tests/test_services_http.py`.

---

## 3. Modes — every default leaves the reading path exactly as it is

| variable | default | other | effect |
|---|---|---|---|
| `SINHALA_VOICE_MODE` | `stub` | `http` | keyword router vs Component 4 |
| `SINHALA_RAG_MODE` | `off` | `http` | local intents only vs Component 3 |
| `SINHALA_TITLE_MODE` | `stub` | `mat` | headline OCR with `sin_raw` |
| `SINHALA_POLISH_MODE` | `off` | `auto`, `on` | Layer 4C LLM post-edit |
| `SINHALA_SEGMENT_MODE` | `off` | `yolo` | YOLO11m article detector |

Two defaults are results, not preferences:

- **`POLISH_MODE=off` is the only setting under which a CER may be quoted.**
  If a general-purpose model rewrites mT5's output, the number being measured
  is no longer the model the thesis is about.
- **`SEGMENT_MODE=off` was measured.** `tools/probe_yolo.py` over 70 real
  captures: of 51 frames comparable against the column-projection layout,
  35 (69%) picked a *different* story, 5 partially agreed, 11 agreed. A
  detector that disagrees with the layout on two thirds of frames cannot
  choose what is read aloud to someone who cannot check it. The fallback
  refuses and asks for a closer frame. Turn it back on when the number
  changes, not before.

---

## 4. Failure rules

1. **`core/svc.py` never raises.** One JSON POST with a timeout. A service
   that is down, a laptop not running Ollama, a container that has not
   started — none of those may become a stack trace. They become
   `(None, reason)`.
2. **Nothing that came from a stub is reported as a result.** Every stub path
   stamps its source and adds a warning, so a transcript can never be read as
   evidence that a component ran.
3. **The common follow-ups never leave this process.** `REPEAT`,
   `READ_ALOUD`, `NEXT`, `PREVIOUS`, `FIRST`, `LENGTH`, `TITLE`, `WARNINGS`,
   `STOP` are answered from the article in session. This is not a stub — it
   is the correct implementation, it needs no network and no API key, and it
   is why the most common interaction survives an outage during the viva.
4. **Two failures must not sound alike.** "No single article in this frame"
   says *move a little closer* — an instruction that fixes the frame.
   "Nothing legible" asks for another photograph. Both were once "nothing
   could be read", which sends nobody anywhere.
5. **The readability gate speaks instead of the text when the text is
   shattered.** A bad capture does not error: Tesseract returns something,
   mT5 corrects that something, and the phone reads it in the same confident
   voice it uses for real news. `core/quality.py` separates SHORT (a six-word
   news brief — read, with a warning) from SHATTERED (replaced).

---

## 5. Session

`core/session.py`. In memory, TTL 1800 s, 32 jobs, cursor stored beside the
Document so it expires with the article it points into. Neither number is
measured; both are design choices and are stated as such.

**This is a stated limitation, not an oversight:** restart the server and
every held article is gone. One process, one user at a time. A database is
the fix and is out of scope for the deadline.

---

## 6. Secrets

`.env` at the repository root, ignored by git. `services/.env.example`
documents the shape with empty values and is tracked.

`origin` is shared with three teammates and **git does not forget**. A key
committed once stays in the history of every clone and every fork; deleting
the file in a later commit removes it from the working tree and from nothing
else. Rotating the key is then the only remedy.

Check with `git check-ignore -v .env` — silence means it is **not** ignored.

---

## 7. Running it

```
python tools/run_all.py              # reader + RAG service
python tools/run_all.py --stubs      # reader + stub voice/RAG, no API key
python tools/run_all.py --title --polish
python tools/check_llm.py            # is the key set and reachable
```

Then `http://127.0.0.1:8000/debug` — upload frames, see every stage, and ask
questions of the article the server is holding. (Ports: reader 8000, voice
8101, RAG 8102; all overridable on `run_all.py`.)
