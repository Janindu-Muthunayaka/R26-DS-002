# personalization/diagnostic.py
# Evidence detection — pure logic, no ML.
#
# Answers one question: did the user give REAL evidence about their preferred
# communication style this turn? Only evidence produced here may train the
# online model (see style_model.py).
#
# WHY THERE IS NO FLAG STEP ANY MORE
# Llama 3.2:1b emitted spurious personalization_flags on most inputs.
# Measured on real output:
#     "Simplify"                        -> {'speed': 'fast'}
#     "Step by step"                    -> {'speed': 'fast'}
#     "Give me the full details"        -> {'language_style': 'simple'}   (!)
#     "Can you explain this a little more?" -> {'language_style': 'simple'} (!)
# Flags that named a style were right 4/11 times (36%). Since 12 Oct the LLM
# is not asked for flags at all, and in Phase 3 the unused flag step
# (style_from_flags, FLAG_STYLE_HINTS) was removed. Style evidence comes from
# the intent label and the sentence's own words (style_from_keywords).
#
# NOTE: the repeat-failure check was removed from this build. It depended on
# retrieved_chunk_id, a Component 3 concern not yet available.

import re

try:
    from intent_detection.modifiers import STYLE_KEYWORDS, style_from_keywords
except ImportError:
    from layers.l7_personalization.intent_detection.modifiers import STYLE_KEYWORDS, style_from_keywords

# ── Explicit intents that state the desired style ────────────────────────
# The LLM invents its own intent names, so synonyms are listed here.
# Everything is upper-cased and space/hyphen -> underscore before lookup.
DIRECT_STYLE_INTENTS = {
    # -> Simple
    "SIMPLIFY": "Simple",
    "SIMPLE": "Simple",
    "MAKE_SIMPLE": "Simple",
    "MAKE_SIMPLER": "Simple",
    "SUMMARIZE": "Simple",
    "SHORTEN": "Simple",
    "BRIEF": "Simple",
    # -> Detailed
    "ELABORATE": "Detailed",
    "MORE_DETAILS": "Detailed",
    "MORE_DETAIL": "Detailed",
    "GIVE_DETAILS": "Detailed",
    "FULL_DETAILS": "Detailed",
    "DETAIL": "Detailed",
    "DETAILED": "Detailed",
    "EXPAND": "Detailed",
    "EXPLAIN_MORE": "Detailed",
    # -> StepByStep
    "STEP_BY_STEP": "StepByStep",
    "STEPBYSTEP": "StepByStep",
    "STEPS": "StepByStep",
    "ONE_BY_ONE": "StepByStep",
    "WALK_THROUGH": "StepByStep",
}

# ── Lexical grounding: STYLE_KEYWORDS and style_from_keywords() now live in
#    intent_detection/modifiers.py (Phase 2b), unchanged. Imported back here
#    so every caller of this module keeps working exactly as before.


def _normalize_intent(intent):
    if not intent:
        return ""
    return re.sub(r"[\s\-]+", "_", str(intent).strip().upper())


def style_from_intent(intent):
    """The style the user explicitly asked for via intent, or None."""
    return DIRECT_STYLE_INTENTS.get(_normalize_intent(intent))


def detect_correction_signal(current_intent, last_interaction):
    """
    Returns the corrected style if THIS turn's intent implies the PREVIOUS
    turn's style was wrong, else None.
    """
    if last_interaction is None:
        return None

    requested_style = style_from_intent(current_intent)
    if requested_style is None:
        return None

    previous_style = last_interaction.get("style_class")
    if previous_style is None or previous_style == requested_style:
        return None

    return requested_style


# ── Backward compatibility helpers ───────────────────────────────────────
REPEAT_INTENTS = {"REPEAT", "REPEAT_AUDIO", "EXPLAIN_AGAIN"}

def is_repeat_failure(current_intent, current_chunk_id, last_interaction):
    """Retained for backward compatibility with early v2 specifications."""
    if last_interaction is None:
        return False
    same_chunk = (
        current_chunk_id is not None
        and current_chunk_id == last_interaction.get("retrieved_chunk_id")
    )
    is_repeat = _normalize_intent(current_intent) in REPEAT_INTENTS
    return bool(same_chunk and is_repeat)


def style_from_flags(flags, english_text=None):
    """Retained for backward compatibility with earlier v2 flag parsers."""
    return None
