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
    is_repeat_intent = current_intent.upper() in REPEAT_INTENTS if current_intent else False

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

    if not current_intent:
        return None

    corrected_style = CORRECTION_INTENT_TO_STYLE.get(current_intent.upper())
    if corrected_style is None:
        return None

    previous_style = last_interaction.get("style_class")
    if previous_style is None or previous_style == corrected_style:
        return None

    return corrected_style
