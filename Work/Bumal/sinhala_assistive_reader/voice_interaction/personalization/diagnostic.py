# personalization/diagnostic.py
# Step 2 — Diagnostic repeat-failure check + correction-signal detection

REPEAT_INTENTS = {"REPEAT", "REPEAT_AUDIO", "EXPLAIN_AGAIN"}

# Explicit user requests that directly imply the PREVIOUS style guess was wrong
CORRECTION_INTENT_TO_STYLE = {
    "SIMPLIFY": "Simple",
    "ELABORATE": "Detailed",
}


def _normalize(text):
    """Lowercase + collapse whitespace, so 'Summarize this.' and
    'summarize this' compare equal. Used only for repeat detection."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def is_repeat_failure(current_intent, current_chunk_id, current_sinhala_text, last_interaction):
    """
    Returns True if this looks like the user re-asking about the same
    content immediately after the last interaction (a TTS/comprehension
    failure), rather than a new request.

    Two independent signals can trigger this, either is enough:
      1. Explicit repeat intent — the user SAID "repeat"/"say that again"
         (caught by REPEAT_INTENTS, as before).
      2. Literal repetition — the user's raw Sinhala input this turn is the
         same sentence as last turn's, even though they didn't use a
         "repeat"-type phrase. The LLM has no memory of the previous turn,
         so it will just classify this as a fresh SUMMARIZE/EXPLAIN/etc.
         instead of REPEAT — signal (1) alone misses this case, which is
         why literally re-asking the same question wasn't being caught.

    Both signals still require the SAME chunk as last turn, since asking
    the same style of question about a *different* chunk is a new request,
    not a failure.
    """
    if last_interaction is None:
        return False

    same_chunk = (
        current_chunk_id is not None
        and current_chunk_id == last_interaction.get("retrieved_chunk_id")
    )
    if not same_chunk:
        return False

    is_repeat_intent = bool(current_intent) and current_intent.upper() in REPEAT_INTENTS

    is_same_text = (
        bool(current_sinhala_text)
        and _normalize(current_sinhala_text) == _normalize(last_interaction.get("sinhala_input"))
    )

    return is_repeat_intent or is_same_text


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

    if not current_intent:
        return None

    corrected_style = CORRECTION_INTENT_TO_STYLE.get(current_intent.upper())
    if corrected_style is None:
        return None

    previous_style = last_interaction.get("style_class")
    if previous_style is None or previous_style == corrected_style:
        return None

    return corrected_style
