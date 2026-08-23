# Build Spec: Personalization Module (Component 4)
## Sinhala Assistive Reader — Adaptive Conversational Personalization

Paste this entire document to your coding agent as one task. Follow the steps
in order — do not skip ahead. Each step has a "Verify" block; run it and
confirm the expected output before moving to the next step.

---

## Context (read this first)

This is Component 4 of a larger group project. My part sits right after
intent detection and before the RAG generation module (Component 3, not yet
built). The job of this module: log every user voice interaction, detect
repeated failures, predict the user's preferred communication style using an
**online learning model** (no upfront training dataset — it learns from one
interaction at a time), and hand a personalized prompt modifier forward.

### Existing project structure (do not modify these files)

```
sinhala_assistive_reader/
└── voice_interaction/
    ├── venv/                          # already created, river + tinydb installed
    ├── data/
    │   ├── db.json                    # TinyDB file, already has user_profiles + interaction_logs tables
    │   └── test_samples.py            # 7 Sinhala test samples with expected_intent
    ├── intent_detection/
    │   ├── approach1_nllb_llm.py      # detect_intent_approach1(sinhala_text) -> dict
    │   ├── approach2_direct_llm.py    # detect_intent_approach2(sinhala_text) -> dict
    │   └── evaluate_both.py
    ├── stt/
    │   └── ... (STT scripts)
    └── personalization/               # NEW — this is what we're building
        ├── __init__.py                # already created, empty
        ├── logger.py                  # BUILD THIS — Step 1
        ├── diagnostic.py              # BUILD THIS — Step 2
        ├── style_model.py             # BUILD THIS — Step 3
        └── main_flow.py               # BUILD THIS — Step 4
```

### Exact return shape of `detect_intent_approach1()` / `detect_intent_approach2()`
(both approaches return this same dict shape — confirmed from existing code)

```python
{
    "approach": "Approach 1 — NLLB + Llama3.2:1b",
    "sinhala_input": "මෙය සාරාංශ කරන්න",
    "english_translation": "Summarize this",
    "intent": "SUMMARIZE",
    "personalization_flags": {"detail_level": "brief"},   # may be {}
    "translation_time_sec": 0.42,
    "llm_time_sec": 0.31,
    "total_time_sec": 0.73
}
```

Known intent values already used in the system prompt (approach1):
`SUMMARIZE`, `EXPLAIN`, `SIMPLIFY`, `ELABORATE`, `REPHRASE`,
`IDENTIFY_CONTENT`, `READ_ALOUD`, `STOP`, `REPEAT`, `NEXT` — plus the model
may invent other short verb-phrase intents freely.

### Existing `db.json` schema (TinyDB)

```json
{
  "user_profiles": {
    "1": {"user_id": "user_001", "persona": "Student", "preferred_speed": "fast"},
    "2": {"user_id": "user_002", "persona": "Elderly", "preferred_speed": "slow"}
  },
  "interaction_logs": {
    "1": {"user_id": "user_001", "intent": "SUMMARIZE", "timestamp": "2026-07-10"}
  }
}
```
We will keep this table but insert richer records going forward (see Step 1).
Do not delete the existing stub record.

### Environment
- Windows, PowerShell, venv located at `voice_interaction/venv`
- Already installed: `river==0.25.0`, `tinydb`
- All commands below assume the terminal is open at
  `...\sinhala_assistive_reader\voice_interaction` with the venv activated
  (prompt shows `(venv)`)

---

## Step 1 — `personalization/logger.py`

**Goal:** a function that takes the dict from `detect_intent_approach1()` and
appends a full record to `interaction_logs` in `db.json`. Also a function to
fetch a user's most recent logged interaction (needed by Step 2).

```python
# personalization/logger.py
# Step 1 — Logs each interaction into db.json (TinyDB)

import os
from datetime import datetime
from tinydb import TinyDB, Query

# data/ is a sibling of personalization/, both under voice_interaction/
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "db.json"
)

db = TinyDB(DB_PATH)
interaction_logs = db.table("interaction_logs")


def log_interaction(user_id, result, retrieved_chunk_id=None, style_class=None):
    """
    Takes the dict returned by detect_intent_approach1() / approach2()
    and appends a full record to interaction_logs. Returns the record.
    """
    record = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "sinhala_input": result.get("sinhala_input"),
        "english_translation": result.get("english_translation"),
        "intent": result.get("intent"),
        "personalization_flags": result.get("personalization_flags", {}),
        "retrieved_chunk_id": retrieved_chunk_id,
        "style_class": style_class
    }
    interaction_logs.insert(record)
    return record


def get_last_interaction(user_id):
    """Returns the most recent logged interaction for this user, or None."""
    User = Query()
    records = interaction_logs.search(User.user_id == user_id)
    return records[-1] if records else None


def update_last_interaction_style(user_id, style_class):
    """Updates the style_class field on the user's most recent log entry."""
    User = Query()
    matches = interaction_logs.search(User.user_id == user_id)
    if not matches:
        return None
    last = matches[-1]
    # Match on user_id + timestamp together (timestamp is unique per insert)
    interaction_logs.update(
        {"style_class": style_class},
        (User.user_id == user_id) & (User.timestamp == last["timestamp"])
    )
    return style_class
```

### Verify Step 1
Create `voice_interaction/test_logger.py` (throwaway test file, project root
level — i.e. inside `voice_interaction/`, not inside `personalization/`):

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intent_detection.approach1_nllb_llm import detect_intent_approach1
from personalization.logger import log_interaction, get_last_interaction

result = detect_intent_approach1("මෙය සාරාංශ කරන්න")
log_interaction("user_001", result)

print(get_last_interaction("user_001"))
```

Run: `python test_logger.py`

**Expected:** prints a dict with `intent: "SUMMARIZE"` (or close),
`english_translation` filled in, `retrieved_chunk_id: None`,
`style_class: None`. Then open `data/db.json` and confirm a NEW entry
appeared under `interaction_logs` (in addition to the original stub `"1"`
entry — do not remove that one).

**Do not proceed to Step 2 until this prints correctly and `db.json` shows
the new record.**

---

## Step 2 — `personalization/diagnostic.py`

**Goal:** pure logic (no ML) — check if the current interaction is a repeat
complaint about the same content, so we can skip the ML model and just
replay audio instead.

```python
# personalization/diagnostic.py
# Step 2 — Diagnostic repeat-failure check

REPEAT_INTENTS = {"REPEAT", "REPEAT_AUDIO", "EXPLAIN_AGAIN"}


def is_repeat_failure(current_intent, current_chunk_id, last_interaction):
    """
    Returns True if this looks like the user re-asking about the same
    content immediately after the last interaction (a TTS/comprehension
    failure), rather than a new request.

    current_intent: str, e.g. "REPEAT"
    current_chunk_id: str or None, the retrieved_chunk_id for this turn
    last_interaction: dict or None, from logger.get_last_interaction()
    """
    if last_interaction is None:
        return False

    same_chunk = (
        current_chunk_id is not None
        and current_chunk_id == last_interaction.get("retrieved_chunk_id")
    )
    is_repeat_intent = current_intent in REPEAT_INTENTS

    return same_chunk and is_repeat_intent
```

### Verify Step 2
Add to a test file (or extend `test_logger.py`):

```python
from personalization.diagnostic import is_repeat_failure

fake_last = {"retrieved_chunk_id": "chunk_1", "intent": "EXPLAIN"}
print(is_repeat_failure("REPEAT", "chunk_1", fake_last))   # expect True
print(is_repeat_failure("SUMMARIZE", "chunk_1", fake_last)) # expect False
print(is_repeat_failure("REPEAT", "chunk_2", fake_last))    # expect False (different chunk)
print(is_repeat_failure("REPEAT", "chunk_1", None))         # expect False (no history)
```

**Expected output:** `True`, `False`, `False`, `False`. Confirm exactly this
before moving on.

---

## Step 3 — `personalization/style_model.py`

**Goal:** the online learning model. No dataset needed — it starts blank and
improves after every `learn_style()` call.

```python
# personalization/style_model.py
# Step 3 — River online learning pipeline for style prediction

from river import compose, feature_extraction, linear_model

STYLE_CLASSES = ["Simple", "Detailed", "StepByStep"]
DEFAULT_STYLE = "Detailed"  # used before the model has seen any data

model = compose.Pipeline(
    feature_extraction.TFIDF(),
    linear_model.SoftmaxRegression()
)

_seen_any_data = False  # tracks whether learn_one has ever been called


def predict_style(english_text):
    """
    Predicts the user's preferred style class for this text.
    Falls back to DEFAULT_STYLE if the model hasn't learned anything yet.
    """
    if not _seen_any_data:
        return DEFAULT_STYLE
    prediction = model.predict_one(english_text)
    return prediction if prediction is not None else DEFAULT_STYLE


def learn_style(english_text, confirmed_class):
    """
    Updates the model with the true label for this interaction.
    confirmed_class must be one of STYLE_CLASSES.
    """
    global _seen_any_data
    if confirmed_class not in STYLE_CLASSES:
        raise ValueError(f"confirmed_class must be one of {STYLE_CLASSES}, got {confirmed_class}")
    model.learn_one(english_text, confirmed_class)
    _seen_any_data = True
```

### Verify Step 3
```python
from personalization.style_model import predict_style, learn_style

print(predict_style("make this shorter"))  # expect "Detailed" (default, no data yet)

learn_style("make this shorter", "Simple")
learn_style("explain in simple words", "Simple")
learn_style("give me full details please", "Detailed")
learn_style("walk me through this step by step", "StepByStep")

print(predict_style("make it brief"))          # expect leaning toward "Simple"
print(predict_style("break it into steps"))     # expect leaning toward "StepByStep"
```

**Note for the agent:** with only 4 training examples, predictions may not
always match intuition perfectly — that's expected for online learning with
very little data. The important thing to verify is that: (a) it runs without
error, (b) `predict_style` returns one of the 3 valid classes every time,
(c) repeated calls to `learn_style` do not crash or reset the model.

---

## Step 4 — `personalization/main_flow.py`

**Goal:** wire Steps 1–3 together into the full loop, using the actual
`detect_intent_approach1` function.

```python
# personalization/main_flow.py
# Step 4 — Full personalization flow, from Sinhala input to style-tagged output

from intent_detection.approach1_nllb_llm import detect_intent_approach1
from personalization.logger import (
    log_interaction, get_last_interaction, update_last_interaction_style
)
from personalization.diagnostic import is_repeat_failure
from personalization.style_model import predict_style, learn_style

STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "",  # no modifier needed, standard generation
    "StepByStep": "Break the explanation into numbered steps.",
}


def handle_voice_command(sinhala_text, user_id, retrieved_chunk_id=None):
    """
    Full pipeline for one user voice command.
    retrieved_chunk_id: pass None until Component 3 is integrated.
    Returns a dict describing what happened and what to send forward.
    """
    # 1. Run existing intent detection
    result = detect_intent_approach1(sinhala_text)

    # 2. Check diagnostic BEFORE logging this turn (need previous turn's data)
    last = get_last_interaction(user_id)
    repeat_failure = is_repeat_failure(result["intent"], retrieved_chunk_id, last)

    # 3. Log this interaction now
    log_interaction(user_id, result, retrieved_chunk_id=retrieved_chunk_id)

    if repeat_failure:
        return {
            "route": "TTS_REPLAY",
            "action": "Replay last response at slower speed",
            "intent": result["intent"],
            "english_translation": result["english_translation"],
        }

    # 4. Predict style, build prompt modifier
    style = predict_style(result["english_translation"])
    prompt_modifier = STYLE_PROMPT_MODIFIERS[style]

    # 5. Update the log with the predicted style
    update_last_interaction_style(user_id, style)

    # 6. Learn from this interaction (using predicted style as the confirmed
    #    label for now — later this can be replaced with a corrected label
    #    if the user pushes back on the next turn)
    learn_style(result["english_translation"], style)

    return {
        "route": "GENERATE",
        "intent": result["intent"],
        "english_translation": result["english_translation"],
        "style_class": style,
        "prompt_modifier": prompt_modifier,
        "personalization_flags": result["personalization_flags"],
        "retrieved_chunk_id": retrieved_chunk_id,
    }
```

### Verify Step 4 — full end-to-end test using existing test samples

Create `voice_interaction/test_main_flow.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.test_samples import test_samples
from personalization.main_flow import handle_voice_command

USER = "user_001"
FAKE_CHUNK = "chunk_article_1"

for sample in test_samples:
    print(f"\n--- {sample['id']} ---")
    output = handle_voice_command(sample["stt_output"], USER, retrieved_chunk_id=FAKE_CHUNK)
    print(output)
```

Run: `python test_main_flow.py`

**Expected:**
- All 7 samples process without crashing.
- Each result dict shows `"route": "GENERATE"` (none should be
  `"TTS_REPLAY"` on this first pass, since intents in `test_samples.py` are
  SUMMARIZE/EXPLAIN/IDENTIFY_CONTENT/etc., not REPEAT — this is expected).
  Note: since all 7 use the same `FAKE_CHUNK`, if any sample's detected
  intent happens to be `"REPEAT"`, that one will correctly route to
  `TTS_REPLAY` — that's correct behavior, not a bug.
- `style_class` appears in every `GENERATE` result, and is one of `Simple`,
  `Detailed`, `StepByStep`.
- After running, open `data/db.json` and confirm 7 new entries appended to
  `interaction_logs`, each with `style_class` filled in (not null).

---

## Final acceptance checklist (confirm all before reporting done)

- [ ] Step 1: `logger.py` — new record appears correctly in `db.json`
- [ ] Step 2: `diagnostic.py` — all 4 test assertions pass exactly as specified
- [ ] Step 3: `style_model.py` — no crashes across repeated predict/learn calls
- [ ] Step 4: `main_flow.py` — all 7 test samples run end-to-end, `db.json` shows 7 new logged interactions with `style_class` filled in
- [ ] `requirements.txt` updated via `pip freeze > requirements.txt` (run from the activated venv)

## Known limitations to leave as-is for now (do not try to fix these)
- `retrieved_chunk_id` is manually passed as a placeholder string — real
  values will come from Component 3 once it's integrated.
- The River model resets every time the Python process restarts (it's
  in-memory only). Persisting it to disk (e.g. via `pickle`) is a later
  task, not part of this build.
- `learn_style()` currently learns from the *predicted* class rather than a
  user-confirmed correction. Adding a correction signal (e.g. detecting when
  the user immediately asks for something different) is a later
  enhancement, not part of this build.
