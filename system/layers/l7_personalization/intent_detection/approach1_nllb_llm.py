# Approach 1: NLLB Translation + OpenAI LLM (intent_detection/llm_backend.py)
# Flow: Sinhala STT → NLLB translates to English → OpenAI extracts intent JSON

import re
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
try:
    from intent_detection.llm_backend import (
        chat, parse_json_reply, get_active_backend, normalize_label, labels_match
    )
except ImportError:
    from layers.l7_personalization.intent_detection.llm_backend import (
        chat, parse_json_reply, get_active_backend, normalize_label, labels_match
    )

# ─── Load NLLB Translation Model ───────────────────────
print("Loading NLLB translation model...")
print("First run will download ~600MB — please wait...")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("[OK] NLLB model loaded successfully!")


try:
    from intent_detection.sinhala_rules import rule_intent
except ImportError:
    from layers.l7_personalization.intent_detection.sinhala_rules import rule_intent


# ─── Translate Sinhala to English ──────────────────────
def translate_to_english(sinhala_text):
    inputs = tokenizer(
        sinhala_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200
    )

    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=200
    )

    return tokenizer.batch_decode(
        translated,
        skip_special_tokens=True
    )[0]


# ─── Extract Intent via Llama 3.2 1B ───────────────────
# Known intents come first: when one fits, the model must use that exact
# label, so the same request always gets the same name. When none fits, the
# model may create a new short label, so commands outside this list (change
# speed, bookmark, translate, ...) are still identified instead of dropped.
#
# The original prompt let the model name EVERY intent freely ("think
# freely"), and the 1B model then produced labels like "SAY IT EASILY" or
# echoed the whole sentence back for common requests. That is why known
# intents are listed and new labels are only a fallback.
KNOWN_INTENTS = [
    "SUMMARIZE", "EXPLAIN", "ELABORATE", "SIMPLIFY", "STEP_BY_STEP",
    "REPHRASE", "IDENTIFY_CONTENT", "ASK_QUESTION", "READ_ALOUD",
    "READ_HEADLINES", "REPEAT", "STOP", "NEXT", "PREVIOUS",
    "GO_TO_SECTION", "GO_TO_PAGE",
]

# Common variants the model produces for a known intent.
INTENT_ALIASES = {
    "SUMMARY": "SUMMARIZE", "SUMMARISE": "SUMMARIZE",
    "SIMPLE": "SIMPLIFY", "MAKE_SIMPLE": "SIMPLIFY", "MAKE_SIMPLER": "SIMPLIFY",
    "MORE_DETAIL": "ELABORATE", "MORE_DETAILS": "ELABORATE", "EXPAND": "ELABORATE",
    "SAY_AGAIN": "REPEAT", "REPEAT_LAST": "REPEAT",
    "PAUSE": "STOP", "STOP_READING": "STOP",
    "NEXT_PAGE": "NEXT", "NEXT_ARTICLE": "NEXT", "GO_FORWARD": "NEXT",
    "GO_BACK": "PREVIOUS", "BACK": "PREVIOUS",
    "PREVIOUS_PAGE": "PREVIOUS", "PREVIOUS_ARTICLE": "PREVIOUS",
    "READ": "READ_ALOUD", "READ_ARTICLE": "READ_ALOUD",
}

# A model-created label is accepted only if it looks like a label: 1-4
# upper-case words joined by "_". Anything else (a copied sentence, empty
# reply) becomes UNKNOWN.
_NEW_LABEL = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+){0,3}$")

# personalization_flags are no longer requested from the LLM. Measured flags
# were mostly wrong (e.g. detail_level "brief" on almost every command), and
# personalization/diagnostic.py only accepts a flag when the sentence's own
# keywords already give the same style — so they never changed a decision.
# Asking for one field keeps the 1B model focused on the intent.
#
# SYSTEM_PROMPT_ORIGINAL is the prompt the thesis benchmark (35/45) was
# measured with. Kept for reference and for evaluate_phase2b.py; production
# now uses SYSTEM_PROMPT = SYSTEM_PROMPT_OPTION_A (Phase 2b decision, below).
SYSTEM_PROMPT_ORIGINAL = """You identify the intent of voice commands for a reading assistant used by visually impaired people.
The command was machine-translated from Sinhala, so the English may be imperfect. Focus on what the user wants done.

Reply with ONLY a JSON object: {"intent": "<LABEL>"}

If one of these labels fits, you MUST use it exactly:
SUMMARIZE - wants a short version, a summary, the main point, or only the important things; has no time for details
EXPLAIN - wants to understand something better: its meaning or its background
ELABORATE - wants more detail, the full details, or a longer answer
SIMPLIFY - wants easier or simpler words, simpler language, says it is too complicated or too difficult
STEP_BY_STEP - wants it broken into ordered steps, one by one
REPHRASE - wants the same thing said in a different way or in other / different words
IDENTIFY_CONTENT - asks what is here, what is in or inside this, what this page or item contains
ASK_QUESTION - asks a specific question about the content (who, when, where, how many, why something happened)
READ_ALOUD - wants text read out: the whole article, the rest of it, or to start reading
READ_HEADLINES - wants to hear the headlines or the list of news titles
REPEAT - wants to hear again what was just said, or did not hear it
STOP - wants the reading to stop or pause, or says that is enough
NEXT - wants to move on to the next page or next item
PREVIOUS - wants to go back to the previous page or previous item
GO_TO_SECTION - wants to go to a named section (sports, politics, business, editorial, ...)
GO_TO_PAGE - wants to go to a page number, the first page or the last page

If NONE of these fit, create a new label: 1 to 3 English words in UPPER_SNAKE_CASE, starting with a verb,
for example CHANGE_SPEED, INCREASE_VOLUME, BOOKMARK_PAGE, TRANSLATE_TEXT, SHARE_ARTICLE, ASK_TIME.
Never copy the user's sentence as the label.

Rules:
- simple / easy / complicated / difficult words -> SIMPLIFY, not EXPLAIN
- "in other words", "another way", "any other way", "differently" -> REPHRASE, even if the sentence also says explain
- what is in this / what is inside / what does it contain -> IDENTIFY_CONTENT, not EXPLAIN
- again / did not hear -> REPEAT, not READ_ALOUD
- summary / briefly / important things / the main point / no time -> SUMMARIZE
- explain / what does it mean / I don't understand -> EXPLAIN, even if it says "a little more",
  unless it also asks for simple words (SIMPLIFY) or another way (REPHRASE)
- a page number, the first page or the last page -> GO_TO_PAGE; only "next page" is NEXT
- a request that is not about reading (time, weather, sending, translating, ...) -> create a new label

Examples:
"Give me a summary" -> {"intent": "SUMMARIZE"}
"Explain the meaning of this paragraph" -> {"intent": "EXPLAIN"}
"Give me the full details" -> {"intent": "ELABORATE"}
"Use easy words, this is too hard" -> {"intent": "SIMPLIFY"}
"Explain it step by step" -> {"intent": "STEP_BY_STEP"}
"Say that in a different way" -> {"intent": "REPHRASE"}
"In other words, explain it" -> {"intent": "REPHRASE"}
"Just tell me the important things" -> {"intent": "SUMMARIZE"}
"What does this page contain?" -> {"intent": "IDENTIFY_CONTENT"}
"Who won the election?" -> {"intent": "ASK_QUESTION"}
"Read the full article" -> {"intent": "READ_ALOUD"}
"I didn't hear, say it again" -> {"intent": "REPEAT"}
"Enough, stop reading" -> {"intent": "STOP"}
"Go to the next page" -> {"intent": "NEXT"}
"Go back to the previous news" -> {"intent": "PREVIOUS"}
"Read a little slower" -> {"intent": "CHANGE_SPEED"}"""


# ─── Phase 2b experiment prompts (ADAPTIVE_VOICE_PLAN.md) ──────────────
# The experiment (intent_detection/evaluate_phase2b.py, 13 Oct) picked
# Option A, so SYSTEM_PROMPT = SYSTEM_PROMPT_OPTION_A (set after both).
#
# Option A — the LLM gives the main action only; repeat / speed / style come
# from the keyword checks in intent_detection/modifiers.py. Differences from
# SYSTEM_PROMPT: CHANGE_SPEED is a KNOWN label, plus one rule and one example
# ("Read the headlines slowly", not in any test set) for action + speed.
KNOWN_INTENTS_OPTION_A = KNOWN_INTENTS + ["CHANGE_SPEED"]

SYSTEM_PROMPT_OPTION_A = """You identify the intent of voice commands for a reading assistant used by visually impaired people.
The command was machine-translated from Sinhala, so the English may be imperfect. Focus on what the user wants done.

Reply with ONLY a JSON object: {"intent": "<LABEL>"}

If one of these labels fits, you MUST use it exactly:
SUMMARIZE - wants a short version, a summary, the main point, or only the important things; has no time for details
EXPLAIN - wants to understand something better: its meaning or its background
ELABORATE - wants more detail, the full details, or a longer answer
SIMPLIFY - wants easier or simpler words, simpler language, says it is too complicated or too difficult
STEP_BY_STEP - wants it broken into ordered steps, one by one
REPHRASE - wants the same thing said in a different way or in other / different words
IDENTIFY_CONTENT - asks what is here, what is in or inside this, what this page or item contains
ASK_QUESTION - asks a specific question about the content (who, when, where, how many, why something happened)
READ_ALOUD - wants text read out: the whole article, the rest of it, or to start reading
READ_HEADLINES - wants to hear the headlines or the list of news titles
REPEAT - wants to hear again what was just said, or did not hear it
STOP - wants the reading to stop or pause, or says that is enough
NEXT - wants to move on to the next page or next item
PREVIOUS - wants to go back to the previous page or previous item
GO_TO_SECTION - wants to go to a named section (sports, politics, business, editorial, ...)
GO_TO_PAGE - wants to go to a page number, the first page or the last page
CHANGE_SPEED - wants the reading or speaking slower or faster, or says it is too fast or too slow

If NONE of these fit, create a new label: 1 to 3 English words in UPPER_SNAKE_CASE, starting with a verb,
for example INCREASE_VOLUME, BOOKMARK_PAGE, TRANSLATE_TEXT, SHARE_ARTICLE, ASK_TIME.
Never copy the user's sentence as the label.

Rules:
- simple / easy / complicated / difficult words -> SIMPLIFY, not EXPLAIN
- "in other words", "another way", "any other way", "differently" -> REPHRASE, even if the sentence also says explain
- what is in this / what is inside / what does it contain -> IDENTIFY_CONTENT, not EXPLAIN
- again / did not hear -> REPEAT, not READ_ALOUD
- summary / briefly / important things / the main point / no time -> SUMMARIZE
- explain / what does it mean / I don't understand -> EXPLAIN, even if it says "a little more",
  unless it also asks for simple words (SIMPLIFY) or another way (REPHRASE)
- a page number, the first page or the last page -> GO_TO_PAGE; only "next page" is NEXT
- slower / faster / too fast / too slow -> CHANGE_SPEED; but if the sentence also asks for an action
  (say it again, read the article, go to the next page, explain ...), return the action, not CHANGE_SPEED
- a request that is not about reading (time, weather, sending, translating, ...) -> create a new label

Examples:
"Give me a summary" -> {"intent": "SUMMARIZE"}
"Explain the meaning of this paragraph" -> {"intent": "EXPLAIN"}
"Give me the full details" -> {"intent": "ELABORATE"}
"Use easy words, this is too hard" -> {"intent": "SIMPLIFY"}
"Explain it step by step" -> {"intent": "STEP_BY_STEP"}
"Say that in a different way" -> {"intent": "REPHRASE"}
"In other words, explain it" -> {"intent": "REPHRASE"}
"Just tell me the important things" -> {"intent": "SUMMARIZE"}
"What does this page contain?" -> {"intent": "IDENTIFY_CONTENT"}
"Who won the election?" -> {"intent": "ASK_QUESTION"}
"Read the full article" -> {"intent": "READ_ALOUD"}
"I didn't hear, say it again" -> {"intent": "REPEAT"}
"Enough, stop reading" -> {"intent": "STOP"}
"Go to the next page" -> {"intent": "NEXT"}
"Go back to the previous news" -> {"intent": "PREVIOUS"}
"Read a little slower" -> {"intent": "CHANGE_SPEED"}
"Read the headlines slowly" -> {"intent": "READ_HEADLINES"}"""

# Option B — the LLM gives everything: the main action AND the modifiers.
# Same labels, rules and example sentences as Option A; only the reply
# format and the field descriptions differ.
SYSTEM_PROMPT_OPTION_B = """You identify the intent of voice commands for a reading assistant used by visually impaired people.
The command was machine-translated from Sinhala, so the English may be imperfect. Focus on what the user wants done.

Reply with ONLY a JSON object with these four fields:
{"intent": "<LABEL>", "speed": "slower" or "faster" or "none", "repeat": true or false, "style_words": [...]}

intent - the ONE main action. If one of these labels fits, you MUST use it exactly:
SUMMARIZE - wants a short version, a summary, the main point, or only the important things; has no time for details
EXPLAIN - wants to understand something better: its meaning or its background
ELABORATE - wants more detail, the full details, or a longer answer
SIMPLIFY - wants easier or simpler words, simpler language, says it is too complicated or too difficult
STEP_BY_STEP - wants it broken into ordered steps, one by one
REPHRASE - wants the same thing said in a different way or in other / different words
IDENTIFY_CONTENT - asks what is here, what is in or inside this, what this page or item contains
ASK_QUESTION - asks a specific question about the content (who, when, where, how many, why something happened)
READ_ALOUD - wants text read out: the whole article, the rest of it, or to start reading
READ_HEADLINES - wants to hear the headlines or the list of news titles
REPEAT - wants to hear again what was just said, or did not hear it
STOP - wants the reading to stop or pause, or says that is enough
NEXT - wants to move on to the next page or next item
PREVIOUS - wants to go back to the previous page or previous item
GO_TO_SECTION - wants to go to a named section (sports, politics, business, editorial, ...)
GO_TO_PAGE - wants to go to a page number, the first page or the last page
CHANGE_SPEED - wants the reading or speaking slower or faster, or says it is too fast or too slow
If NONE of these fit, create a new label: 1 to 3 English words in UPPER_SNAKE_CASE, starting with a verb,
for example INCREASE_VOLUME, BOOKMARK_PAGE, TRANSLATE_TEXT, SHARE_ARTICLE, ASK_TIME.
Never copy the user's sentence as the label.

speed - "slower" if the user wants it slower or says it is too fast; "faster" if the user wants it faster
or says it is too slow; otherwise "none".
repeat - true only if the user wants to hear the same thing again or did not hear it. Saying it again in a
different, simpler or more detailed way is NOT a repeat (false).
style_words - the user's own words about HOW to explain (simply, briefly, in detail, step by step), or [] if none.

Rules:
- simple / easy / complicated / difficult words -> SIMPLIFY, not EXPLAIN
- "in other words", "another way", "any other way", "differently" -> REPHRASE, even if the sentence also says explain
- what is in this / what is inside / what does it contain -> IDENTIFY_CONTENT, not EXPLAIN
- again / did not hear -> REPEAT, not READ_ALOUD
- summary / briefly / important things / the main point / no time -> SUMMARIZE
- explain / what does it mean / I don't understand -> EXPLAIN, even if it says "a little more",
  unless it also asks for simple words (SIMPLIFY) or another way (REPHRASE)
- a page number, the first page or the last page -> GO_TO_PAGE; only "next page" is NEXT
- slower / faster / too fast / too slow -> CHANGE_SPEED; but if the sentence also asks for an action
  (say it again, read the article, go to the next page, explain ...), return the action, not CHANGE_SPEED
- a request that is not about reading (time, weather, sending, translating, ...) -> create a new label

Examples:
"Give me a summary" -> {"intent": "SUMMARIZE", "speed": "none", "repeat": false, "style_words": []}
"Explain the meaning of this paragraph" -> {"intent": "EXPLAIN", "speed": "none", "repeat": false, "style_words": []}
"Give me the full details" -> {"intent": "ELABORATE", "speed": "none", "repeat": false, "style_words": ["full details"]}
"Use easy words, this is too hard" -> {"intent": "SIMPLIFY", "speed": "none", "repeat": false, "style_words": ["easy words"]}
"Explain it step by step" -> {"intent": "STEP_BY_STEP", "speed": "none", "repeat": false, "style_words": ["step by step"]}
"Say that in a different way" -> {"intent": "REPHRASE", "speed": "none", "repeat": false, "style_words": []}
"In other words, explain it" -> {"intent": "REPHRASE", "speed": "none", "repeat": false, "style_words": []}
"Just tell me the important things" -> {"intent": "SUMMARIZE", "speed": "none", "repeat": false, "style_words": []}
"What does this page contain?" -> {"intent": "IDENTIFY_CONTENT", "speed": "none", "repeat": false, "style_words": []}
"Who won the election?" -> {"intent": "ASK_QUESTION", "speed": "none", "repeat": false, "style_words": []}
"Read the full article" -> {"intent": "READ_ALOUD", "speed": "none", "repeat": false, "style_words": []}
"I didn't hear, say it again" -> {"intent": "REPEAT", "speed": "none", "repeat": true, "style_words": []}
"Enough, stop reading" -> {"intent": "STOP", "speed": "none", "repeat": false, "style_words": []}
"Go to the next page" -> {"intent": "NEXT", "speed": "none", "repeat": false, "style_words": []}
"Go back to the previous news" -> {"intent": "PREVIOUS", "speed": "none", "repeat": false, "style_words": []}
"Read a little slower" -> {"intent": "CHANGE_SPEED", "speed": "slower", "repeat": false, "style_words": []}
"Read the headlines slowly" -> {"intent": "READ_HEADLINES", "speed": "slower", "repeat": false, "style_words": []}"""

# ─── Production prompt: Option A (Phase 2b decision, 14 Oct) ───────────
SYSTEM_PROMPT = SYSTEM_PROMPT_OPTION_A
PRODUCTION_KNOWN_INTENTS = KNOWN_INTENTS_OPTION_A


def parse_option_b(raw_reply):
    """
    Option B reply -> (modifiers, complete). modifiers has the same shape as
    build_frame()'s; complete is False when the reply is not JSON or lacks a
    field. style_words are mapped to a style with the same English keyword
    list the style model uses (modifiers.style_from_keywords).
    """
    try:
        from intent_detection.modifiers import style_from_keywords
    except ImportError:
        from layers.l7_personalization.intent_detection.modifiers import style_from_keywords
    parsed = parse_json_reply(raw_reply)
    if not isinstance(parsed, dict):
        parsed = {}
    complete = all(k in parsed for k in ("intent", "speed", "repeat", "style_words"))

    speed = str(parsed.get("speed", "")).strip().lower()
    speed = "slower" if speed.startswith("slow") else "faster" if speed.startswith("fast") else None
    repeat = parsed.get("repeat") in (True, "true", "True", "yes")
    words = parsed.get("style_words") or []
    if isinstance(words, str):
        words = [words]
    style = style_from_keywords(" ".join(str(w) for w in words))
    return {"repeat": repeat, "speed": speed, "style": style}, complete


def normalize_intent(raw_reply, known_intents=KNOWN_INTENTS):
    """
    Returns a known intent when the reply names one (directly or via an
    alias), otherwise the model's own new label if it is well-formed,
    otherwise "UNKNOWN".
    """
    parsed = parse_json_reply(raw_reply)
    candidate = parsed.get("intent", "") if isinstance(parsed, dict) else ""
    key = normalize_label(candidate)
    key = INTENT_ALIASES.get(key, key)
    if key in known_intents:
        return key
    if key and len(key) <= 30 and _NEW_LABEL.match(key):
        return key

    # No usable JSON — look for a known label in the raw text, longest first
    # so STEP_BY_STEP wins over STOP etc.
    text = re.sub(r"[\s\-]+", "_", str(raw_reply).upper())
    for label in sorted(known_intents, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){label}(?![A-Z])", text):
            return label
    return "UNKNOWN"


def extract_intent_llama(english_text):
    user_prompt = f"User command: '{english_text}'"
    raw = chat(SYSTEM_PROMPT, user_prompt)
    return {"intent": normalize_intent(raw, PRODUCTION_KNOWN_INTENTS)}


# ─── Full Pipeline ──────────────────────────────────────
def detect_intent_approach1(sinhala_text):
    """
    Full Approach 1 pipeline:
    Sinhala → NLLB English → Llama JSON
    """
    # Step 1 — Translate (always: the English is shown, logged and used by RAG)
    t1 = time.time()
    english_text = translate_to_english(sinhala_text)
    translation_time = time.time() - t1

    # Step 2 — Extract intent. An unambiguous Sinhala command is decided by
    # the Sinhala rules (intent_detection/sinhala_rules.py) and the LLM is not
    # asked: the translation is what breaks short commands ("පස්සට යන්න" ->
    # "Go back to the drawing board"), and the LLM can only label what it is given.
    t2 = time.time()
    rule = rule_intent(sinhala_text)
    if rule:
        result, source = {"intent": rule}, "sinhala_rule"
    else:
        result, source = extract_intent_llama(english_text), "llm"
    llm_time = time.time() - t2

    total_time = translation_time + llm_time

    return {
        "approach":              f"Approach 1 — NLLB + LLM ({get_active_backend()})",
        "sinhala_input":         sinhala_text,
        "english_translation":   english_text,
        "intent":                result["intent"],
        "intent_source":         source,
        "personalization_flags": {},  # no longer extracted — see SYSTEM_PROMPT note
        "translation_time_sec":  round(translation_time, 3),
        "llm_time_sec":          round(llm_time, 3),
        "total_time_sec":        round(total_time, 3)
    }


# ─── Run All 6 Test Samples ─────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.test_samples import test_samples

    print("=" * 60)
    print("APPROACH 1 — NLLB + Llama3.2:1b")
    print("=" * 60)

    correct_count = 0
    total_time_all = 0

    for sample in test_samples:
        print(f"\nID       : {sample['id']}")
        print(f"Input    : {sample['stt_output']}")
        print(f"Expected : {sample['expected_intent']}")

        result = detect_intent_approach1(sample["stt_output"])
        correct = labels_match(result["intent"], sample["expected_intent"])

        if correct:
            correct_count += 1
        total_time_all += result["total_time_sec"]

        print(f"English  : {result['english_translation']}")
        print(f"Intent   : {result['intent']} {'[OK]' if correct else '[X]'}")
        print(f"Flags    : {result['personalization_flags']}")
        print(f"Trans    : {result['translation_time_sec']}s")
        print(f"LLM      : {result['llm_time_sec']}s")
        print(f"Total    : {result['total_time_sec']}s")
        print("-" * 60)

    accuracy = (correct_count / len(test_samples)) * 100
    avg_time = total_time_all / len(test_samples)

    print(f"\n{'=' * 60}")
    print(f"APPROACH 1 FINAL RESULTS")
    print(f"Correct  : {correct_count}/{len(test_samples)}")
    print(f"Accuracy : {accuracy:.1f}%")
    print(f"Avg Time : {avg_time:.3f}s")
    print(f"{'=' * 60}")