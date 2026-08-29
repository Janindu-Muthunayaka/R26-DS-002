# personalization/diagnostic.py
# Evidence detection — pure logic, no ML.
#
# Answers one question: did the user give REAL evidence about their preferred
# communication style this turn? Only evidence produced here may train the
# online model (see style_model.py).
#
# WHY THE GROUNDING LAYER EXISTS
# Llama 3.2:1b is a very small model and emits spurious personalization_flags
# on most inputs. Measured on real output:
#     "Simplify"                        -> {'speed': 'fast'}
#     "Step by step"                    -> {'speed': 'fast'}
#     "Give me the full details"        -> {'language_style': 'simple'}   (!)
#     "Can you explain this a little more?" -> {'language_style': 'simple'} (!)
# The last two are plainly wrong and were poisoning the training data with
# "Simple" labels on requests for MORE detail. So a flag is now accepted as
# evidence only if the translated sentence lexically supports it.
#
# NOTE: the repeat-failure check was removed from this build. It depended on
# retrieved_chunk_id, a Component 3 concern not yet available.

import re

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

# ── Explicit flags that state the desired style ──────────────────────────
FLAG_STYLE_HINTS = {
    ("language_style", "simple"): "Simple",
    ("language_style", "technical"): "Detailed",
    ("detail_level", "brief"): "Simple",
    ("detail_level", "detailed"): "Detailed",
    ("detail_level", "step_by_step"): "StepByStep",
}

# ── Lexical grounding: words in the English translation that genuinely
#    indicate each style. Used both to corroborate/reject a flag and as a
#    direct evidence source when the flag is missing or untrustworthy.
STYLE_KEYWORDS = {
    "Simple": [
        "simple", "simply", "simpler", "simplify", "easy", "easier",
        "plain", "basic", "brief", "briefly", "short", "shorter",
        "summary", "summarize", "summarise", "concise", "quick",
    ],
    "Detailed": [
        "detail", "details", "detailed", "elaborate", "more", "full",
        "fully", "thorough", "in-depth", "深", "deeper", "expand",
        "comprehensive", "longer", "complete", "everything", "technical",
    ],
    "StepByStep": [
        "step", "steps", "stepwise", "one by one", "sequentially",
        "in order", "numbered", "walk me through", "stages",
    ],
}


def _normalize_intent(intent):
    if not intent:
        return ""
    return re.sub(r"[\s\-]+", "_", str(intent).strip().upper())


def _keyword_scores(english_text):
    """Counts style-indicating keywords in the translated sentence."""
    if not english_text:
        return {}
    text = str(english_text).lower()
    scores = {}
    for style, words in STYLE_KEYWORDS.items():
        hits = 0
        for w in words:
            if " " in w:
                if w in text:
                    hits += 1
            elif re.search(rf"\b{re.escape(w)}\b", text):
                hits += 1
        if hits:
            scores[style] = hits
    return scores


def style_from_intent(intent):
    """The style the user explicitly asked for via intent, or None."""
    return DIRECT_STYLE_INTENTS.get(_normalize_intent(intent))


def style_from_keywords(english_text):
    """The style indicated by the words of the sentence itself, or None.
    Ties (a sentence containing both 'simple' and 'more') return None so we
    don't guess — the model handles it instead."""
    scores = _keyword_scores(english_text)
    if not scores:
        return None
    best = max(scores.values())
    winners = [s for s, n in scores.items() if n == best]
    return winners[0] if len(winners) == 1 else None


def style_from_flags(personalization_flags, english_text=None):
    """
    The style the user signalled via personalization_flags — but only if the
    sentence corroborates it.

    'speed' is deliberately unmapped: it is a TTS playback property, not a
    communication-style signal.

    Grounding rules:
      - no keywords found in the sentence at all -> reject the flag
        (the LLM emitted it without textual support)
      - keywords point at a DIFFERENT style than the flag -> trust the
        keywords, not the flag
      - keywords agree, or are ambiguous -> accept the flag
    """
    if not personalization_flags:
        return None

    flag_style = None
    for key in ("language_style", "detail_level"):
        value = personalization_flags.get(key)
        if value:
            hint = FLAG_STYLE_HINTS.get((key, str(value).strip().lower()))
            if hint:
                flag_style = hint
                break

    if flag_style is None:
        return None

    if english_text is None:
        return flag_style

    scores = _keyword_scores(english_text)
    if not scores:
        return None                      # unsupported flag -> reject

    if flag_style in scores:
        return flag_style                # corroborated

    keyword_style = style_from_keywords(english_text)
    return keyword_style                 # contradicted -> trust the text


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
