# Build Spec: Personalization Module (Component 4)
## Sinhala Assistive Reader — Adaptive Conversational Personalization
### v2 — adds model persistence + correction-signal learning

Paste this entire document to your coding agent as one task. Follow the steps
in order — do not skip ahead. Each step has a "Verify" block; run it and
confirm the expected output before moving to the next step.

**What changed from v1:** two fixes have been built into this version based
on a design review:
1. The style model now saves itself to disk after every learning update and
   reloads on startup (fixes: model forgetting everything on restart).
2. The system now detects when a user corrects a previous style guess
   (e.g. predicted "Detailed", user immediately says "simplify this") and
   learns from the correction instead of blindly trusting its own guess
   (fixes: self-training bias / model just reinforcing itself).

---

## Context (read this first)

This is Component 4 of a larger group project — the "Adaptive Conversational
Personalization and Voice Interaction Module" (the voice module). It covers
STT, intent detection, personalization, and TTS. This spec covers only the
**personalization stage**, which is where the research contribution lives —
STT, intent detection, and TTS use more established approaches; this stage
is where we're doing something novel (online learning with no upfront
training dataset).

My part sits right after intent detection and before the RAG generation
module (Component 3, not yet built). The job of this module: log every user
voice interaction, detect repeated failures, predict the user's preferred
communication style using an **online learning model**, correct itself when
the user pushes back, and hand a personalized prompt modifier forward.

### Existing project structure (do not modify these files)

```
sinhala_assistive_reader/
└── voice_interaction/
    ├── venv/                          # already created, river + tinydb installed
    ├── data/
    │   ├── db.json                    # TinyDB file, already has user_profiles + interaction_logs tables
    │   ├── test_samples.py            # 7 Sinhala test samples with expected_intent
    │   └── style_model.pkl            # NEW — will be created automatically by style_model.py, don't create manually
    ├── intent_detection/
    │   ├── approach1_nllb_llm.py      # detect_intent_approach1(sinhala_text) -> dict  <-- SELECTED APPROACH
    │   ├── approach2_direct_llm.py    # detect_intent_approach2(sinhala_text) -> dict  (not used going forward)
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

**Confirmed:** Approach 1 (NLLB translation + Llama 3.2:1b via Ollama) is
the selected intent detection method. All code below imports from
`approach1_nllb_llm.py`.

### Exact return shape of `detect_intent_approach1()`

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

Two of these intents matter specially for this build:
- `REPEAT` → routes to the diagnostic fault-check (Step 2), bypassing ML entirely
- `SIMPLIFY` / `ELABORATE` → treated as explicit correction signals (Step 4) when they immediately follow a personalized response

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
appends a full record to `interaction_logs` in `db.json`. Also functions to
fetch a user's most recent logged interaction, and to update a *specific*
past record by timestamp (needed for Step 4's correction logic).

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
    Takes the dict returned by detect_intent_approach1() and appends a full
    record to interaction_logs. Returns the record (including its timestamp,
    which acts as the record's identifier for later updates).
    """
    record = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "sinhala_input": result.get("sinhala_input"),
        "english_translation": result.get("english_translation"),
        "intent": result.get("intent"),
        "personalization_flags": result.get("personalization_flags", {}),
        "retrieved_chunk_id": retrieved_chunk_id,
        "style_class": style_class,
        "corrected": False   # flips to True if Step 4 later relabels this record
    }
    interaction_logs.insert(record)
    return record


def get_last_interaction(user_id):
    """Returns the most recent logged interaction for this user, or None."""
    User = Query()
    records = interaction_logs.search(User.user_id == user_id)
    return records[-1] if records else None


def update_last_interaction_style(user_id, style_class):
    """Updates the style_class field on the user's most recent log entry
    (normal path — this record's own predicted style)."""
    User = Query()
    matches = interaction_logs.search(User.user_id == user_id)
    if not matches:
        return None
    last = matches[-1]
    interaction_logs.update(
        {"style_class": style_class},
        (User.user_id == user_id) & (User.timestamp == last["timestamp"])
    )
    return style_class


def update_interaction_style_by_timestamp(user_id, timestamp, corrected_style_class):
    """Relabels a SPECIFIC past record (identified by its exact timestamp)
    as corrected, and marks corrected=True. Used by Step 4 when the user's
    next turn signals the previous prediction was wrong."""
    User = Query()
    interaction_logs.update(
        {"style_class": corrected_style_class, "corrected": True},
        (User.user_id == user_id) & (User.timestamp == timestamp)
    )
    return corrected_style_class
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
`style_class: None`, `corrected: False`. Then open `data/db.json` and
confirm a NEW entry appeared under `interaction_logs` (in addition to the
original stub `"1"` entry — do not remove that one).

**Do not proceed to Step 2 until this prints correctly and `db.json` shows
the new record.**

---

## Step 2 — `personalization/diagnostic.py`

**Goal:** pure logic (no ML) — two checks: (a) is this a repeat complaint
about the same content (skip ML, just replay audio), and (b) does this turn
signal a correction to the *previous* turn's style prediction.

```python
# personalization/diagnostic.py
# Step 2 — Diagnostic repeat-failure check + correction-signal detection

REPEAT_INTENTS = {"REPEAT", "REPEAT_AUDIO", "EXPLAIN_AGAIN"}

# Explicit user requests that directly imply the PREVIOUS style guess was wrong
CORRECTION_INTENT_TO_STYLE = {
    "SIMPLIFY": "Simple",
    "ELABORATE": "Detailed",
}


def is_repeat_failure(current_intent, current_chunk_id, last_interaction):
    """
    Returns True if this looks like the user re-asking about the same
    content immediately after the last interaction (a TTS/comprehension
    failure), rather than a new request.
    """
    if last_interaction is None:
        return False

    same_chunk = (
        current_chunk_id is not None
        and current_chunk_id == last_interaction.get("retrieved_chunk_id")
    )
    is_repeat_intent = current_intent in REPEAT_INTENTS

    return same_chunk and is_repeat_intent


def detect_correction_signal(current_intent, last_interaction):
    """
    Returns the corrected style class (str) if this turn's intent implies
    the PREVIOUS turn's predicted style was wrong, otherwise returns None.

    Only fires if:
      - the current intent is an explicit complexity request (SIMPLIFY/ELABORATE)
      - there IS a previous interaction with a style_class already set
      - the previous style_class differs from what's now being requested
        (no point "correcting" a guess that was already right)
    """
    if last_interaction is None:
        return None

    corrected_style = CORRECTION_INTENT_TO_STYLE.get(current_intent)
    if corrected_style is None:
        return None

    previous_style = last_interaction.get("style_class")
    if previous_style is None or previous_style == corrected_style:
        return None

    return corrected_style
```

### Verify Step 2
Add to a test file (or extend `test_logger.py`):

```python
from personalization.diagnostic import is_repeat_failure, detect_correction_signal

fake_last = {"retrieved_chunk_id": "chunk_1", "intent": "EXPLAIN", "style_class": "Detailed"}

# Repeat-failure checks
print(is_repeat_failure("REPEAT", "chunk_1", fake_last))    # expect True
print(is_repeat_failure("SUMMARIZE", "chunk_1", fake_last)) # expect False
print(is_repeat_failure("REPEAT", "chunk_2", fake_last))    # expect False (different chunk)
print(is_repeat_failure("REPEAT", "chunk_1", None))         # expect False (no history)

# Correction-signal checks
print(detect_correction_signal("SIMPLIFY", fake_last))      # expect "Simple" (was Detailed, now asked simpler)
print(detect_correction_signal("ELABORATE", fake_last))     # expect None (already Detailed, no correction needed)
print(detect_correction_signal("SUMMARIZE", fake_last))     # expect None (not a correction-type intent)
print(detect_correction_signal("SIMPLIFY", None))           # expect None (no previous interaction)
```

**Expected output, in order:** `True`, `False`, `False`, `False`, `"Simple"`,
`None`, `None`, `None`. Confirm exactly this before moving on.

---

## Step 3 — `personalization/style_model.py`

**Goal:** the online learning model, now with disk persistence. It starts
blank on first-ever run, saves itself to `data/style_model.pkl` after every
learning update, and reloads automatically on the next run — so a user's
learned profile survives a restart.

```python
# personalization/style_model.py
# Step 3 — River online learning pipeline for style prediction, with persistence

import os
import pickle
from river import compose, feature_extraction, linear_model

STYLE_CLASSES = ["Simple", "Detailed", "StepByStep"]
DEFAULT_STYLE = "Detailed"  # used before the model has seen any data

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "style_model.pkl"
)


def _fresh_pipeline():
    return compose.Pipeline(
        feature_extraction.TFIDF(),
        linear_model.SoftmaxRegression()
    )


def _load_state():
    """Loads {'model': ..., 'seen_any_data': ...} from disk if it exists,
    otherwise returns a fresh, untrained state."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return {"model": _fresh_pipeline(), "seen_any_data": False}


def _save_state():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_state, f)


# Module-level state, loaded once when this module is first imported
_state = _load_state()


def predict_style(english_text):
    """
    Predicts the user's preferred style class for this text.
    Falls back to DEFAULT_STYLE if the model hasn't learned anything yet.
    """
    if not _state["seen_any_data"]:
        return DEFAULT_STYLE
    prediction = _state["model"].predict_one(english_text)
    return prediction if prediction is not None else DEFAULT_STYLE


def learn_style(english_text, confirmed_class):
    """
    Updates the model with the true (or corrected) label for this
    interaction, then immediately persists the updated model to disk.
    confirmed_class must be one of STYLE_CLASSES.
    """
    if confirmed_class not in STYLE_CLASSES:
        raise ValueError(f"confirmed_class must be one of {STYLE_CLASSES}, got {confirmed_class}")
    _state["model"].learn_one(english_text, confirmed_class)
    _state["seen_any_data"] = True
    _save_state()
```

**Why the save happens inside `learn_style` and not in a separate "save on
exit" step:** voice interactions can end unpredictably (app closed, device
powered off). Saving immediately after every single learning update
guarantees no learned progress is ever lost, at the cost of a tiny bit of
disk I/O per interaction — a good tradeoff for a low-frequency event like a
voice command (not a hot loop).

### Verify Step 3
```python
from personalization.style_model import predict_style, learn_style, MODEL_PATH
import os

print(predict_style("make this shorter"))  # expect "Detailed" (default, no data yet — unless a previous test run already saved a model, see note below)

learn_style("make this shorter", "Simple")
learn_style("explain in simple words", "Simple")
learn_style("give me full details please", "Detailed")
learn_style("walk me through this step by step", "StepByStep")

print(predict_style("make it brief"))          # expect leaning toward "Simple"
print(predict_style("break it into steps"))     # expect leaning toward "StepByStep"
print(os.path.exists(MODEL_PATH))               # expect True — file should now exist
```

Run this test file TWICE in a row (two separate `python` invocations, not
just two calls in one script). **On the second run**, confirm that
`predict_style("make it brief")` immediately (before calling `learn_style`
again) already leans toward "Simple" — this proves the model persisted
across the restart instead of resetting to `DEFAULT_STYLE`.

**Note for the agent:** with only 4 training examples, predictions may not
always match intuition perfectly — expected for online learning with very
little data. Verify: (a) no errors, (b) `predict_style` always returns one
of the 3 valid classes, (c) `data/style_model.pkl` is created and its
contents persist across two separate script runs.

---

## Step 4 — `personalization/main_flow.py`

**Goal:** wire Steps 1–3 together into the full loop, now including the
correction-signal check: before handling the CURRENT turn, check if it
implies the PREVIOUS turn's style guess was wrong, and if so, relabel and
re-learn from that previous turn using the corrected label.

```python
# personalization/main_flow.py
# Step 4 — Full personalization flow, from Sinhala input to style-tagged output

from intent_detection.approach1_nllb_llm import detect_intent_approach1
from personalization.logger import (
    log_interaction, get_last_interaction,
    update_last_interaction_style, update_interaction_style_by_timestamp
)
from personalization.diagnostic import is_repeat_failure, detect_correction_signal
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

    # 2. Look at the previous turn BEFORE logging this one
    last = get_last_interaction(user_id)

    # 3. Correction-signal check: does THIS turn imply the PREVIOUS
    #    turn's style guess was wrong? If so, relabel and re-learn from
    #    the previous turn's text using the corrected label.
    corrected_style = detect_correction_signal(result["intent"], last)
    if corrected_style is not None:
        update_interaction_style_by_timestamp(user_id, last["timestamp"], corrected_style)
        learn_style(last["english_translation"], corrected_style)

    # 4. Repeat-failure check (unchanged from v1)
    repeat_failure = is_repeat_failure(result["intent"], retrieved_chunk_id, last)

    # 5. Log this interaction now
    log_interaction(user_id, result, retrieved_chunk_id=retrieved_chunk_id)

    if repeat_failure:
        return {
            "route": "TTS_REPLAY",
            "action": "Replay last response at slower speed",
            "intent": result["intent"],
            "english_translation": result["english_translation"],
            "correction_applied": corrected_style,
        }

    # 6. Predict style for THIS turn, build prompt modifier
    style = predict_style(result["english_translation"])
    prompt_modifier = STYLE_PROMPT_MODIFIERS[style]

    # 7. Update the log with the predicted style for this turn
    update_last_interaction_style(user_id, style)

    # 8. Learn from this interaction using its own predicted style.
    #    (This still has the "self-training" characteristic for THIS turn —
    #    that's expected and fine, because we can't know if THIS guess is
    #    right until the NEXT turn's intent tells us. The correction check
    #    in step 3 is what fixes any wrong guess, one turn later.)
    learn_style(result["english_translation"], style)

    return {
        "route": "GENERATE",
        "intent": result["intent"],
        "english_translation": result["english_translation"],
        "style_class": style,
        "prompt_modifier": prompt_modifier,
        "personalization_flags": result["personalization_flags"],
        "retrieved_chunk_id": retrieved_chunk_id,
        "correction_applied": corrected_style,
    }
```

**Important design note for the agent:** the self-training characteristic
isn't fully eliminated — it can't be, since we have no way to know a guess
was wrong until the user's NEXT turn tells us. What this version fixes is
that a wrong guess no longer stays uncorrected forever: the moment the user
signals a problem (via SIMPLIFY/ELABORATE), the previous turn's label gets
overwritten with the corrected value and the model re-learns from it. This
is a one-step-delayed correction loop, which is the standard, defensible
pattern for this kind of implicit-feedback online learning.

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
- Each result dict shows `"route": "GENERATE"` unless a sample's intent is
  `REPEAT` (then `"TTS_REPLAY"`) — either is correct depending on the actual
  test data, not a bug.
- Each result includes `"correction_applied"` — `None` for most turns, and a
  style string (e.g. `"Simple"`) on any turn whose intent is `SIMPLIFY` or
  `ELABORATE` and which follows a turn with a different predicted style.
- `style_class` appears in every `GENERATE` result, and is one of `Simple`,
  `Detailed`, `StepByStep`.
- After running, open `data/db.json` and confirm 7 new entries appended to
  `interaction_logs`. Any entry that was corrected should show
  `"corrected": true` and a `style_class` that reflects the correction, not
  the original guess.

### Extra verification for the correction loop specifically
Run this focused test to see the correction mechanism in isolation:

```python
from personalization.main_flow import handle_voice_command

USER = "user_correction_test"
CHUNK = "chunk_test"

# Turn 1: some request, model guesses a style (likely "Detailed" by default)
r1 = handle_voice_command("පැහැදිලි කරන්න", USER, retrieved_chunk_id=CHUNK)  # "Explain this"
print("Turn 1:", r1)

# Turn 2: user explicitly asks to simplify -> should trigger a correction on Turn 1
r2 = handle_voice_command("සරල කරන්න", USER, retrieved_chunk_id=CHUNK)  # "Simplify this"
print("Turn 2:", r2)
```

**Expected:** `r2["correction_applied"]` should be `"Simple"` (assuming
Turn 1's predicted style wasn't already "Simple"). Open `data/db.json` and
confirm Turn 1's record now shows `"style_class": "Simple"` and
`"corrected": true`.

---

## Final acceptance checklist (confirm all before reporting done)

- [ ] Step 1: `logger.py` — new record appears correctly in `db.json`, includes `corrected: False` by default
- [ ] Step 2: `diagnostic.py` — all 8 test assertions pass exactly as specified
- [ ] Step 3: `style_model.py` — model persists correctly across two separate script runs; `data/style_model.pkl` is created
- [ ] Step 4: `main_flow.py` — all 7 test samples run end-to-end; the focused correction-loop test shows `correction_applied: "Simple"` and the DB record updates accordingly
- [ ] `requirements.txt` updated via `pip freeze > requirements.txt` (run from the activated venv)

## Known limitations to leave as-is for now (do not try to fix these)
- `retrieved_chunk_id` is manually passed as a placeholder string — real
  values will come from Component 3 once it's integrated.
- Correction detection only covers two explicit intents (`SIMPLIFY`,
  `ELABORATE`). It won't catch a correction implied by less direct phrasing
  (e.g. "that was too much" without a clean intent match) — improving intent
  coverage for correction signals is future work, not part of this build.
- TF-IDF on short voice commands has limited generalization across synonyms
  (e.g. "shorter" vs "brief") since it only matches literal word overlap.
  A future improvement (character n-grams, or a small synonym map) is noted
  as future work, not part of this build.
