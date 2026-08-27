# personalization/main_flow.py
# Step 4 — Full personalization flow, from Sinhala input to a style-tagged
# prompt for Component 3 (RAG generation).
#
# This file is PURE PIPELINE LOGIC — it does not print anything. It always
# returns one structured result broken into stages, so any caller (a test
# script, app.py, or eventually a real backend) can inspect what happened
# at every step, not just the final output.

from intent_detection.approach1_nllb_llm import detect_intent_approach1
from personalization.logger import (
    log_interaction, get_last_interaction,
    update_last_interaction_style, update_interaction_style_by_timestamp
)
from personalization.diagnostic import is_repeat_failure, detect_correction_signal
from personalization.style_model import predict_style, learn_style

STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.",
    "StepByStep": "Break the explanation into numbered steps.",
}

# If the user's intent DIRECTLY states the style they want, trust it
# outright instead of asking the online model to guess. This is the fix:
# previously predict_style() was called unconditionally, so a user saying
# "simplify this" could still be told the style was "Detailed".
DIRECT_STYLE_OVERRIDE = {
    "SIMPLIFY": "Simple",
    "ELABORATE": "Detailed",
}

# personalization_flags (extracted by the same LLM call, alongside intent)
# carry implicit style evidence even on turns whose intent ISN'T a clean
# SIMPLIFY/ELABORATE — e.g. intent=SUMMARIZE with flags={"detail_level":
# "brief"}. Previously these flags were logged and forwarded to Component 3
# but never actually used to decide anything. This is real, free evidence
# from the LLM call we already paid for, so we use it as a second-tier
# signal: weaker than an explicit SIMPLIFY/ELABORATE intent, but stronger
# than letting the online model guess blind.
FLAG_STYLE_HINTS = {
    ("language_style", "simple"): "Simple",
    ("language_style", "technical"): "Detailed",
    ("detail_level", "brief"): "Simple",
    ("detail_level", "detailed"): "Detailed",
}


def _style_hint_from_flags(personalization_flags):
    """Returns a style hint from personalization_flags if one of the known
    keys/values is present, else None. 'speed' is intentionally not mapped
    here — it's not a style signal, it's forwarded as-is for a future
    TTS/Component 5 to use for playback speed."""
    if not personalization_flags:
        return None
    for key in ("language_style", "detail_level"):
        value = personalization_flags.get(key)
        if value:
            hint = FLAG_STYLE_HINTS.get((key, str(value).lower()))
            if hint:
                return hint
    return None


def handle_voice_command(sinhala_text, user_id, retrieved_chunk_id=None):
    """
    Full pipeline for one user voice command.

    retrieved_chunk_id: pass None until Component 3 is integrated.

    Returns a dict with THREE stages plus the final packaged prompt:
      {
        "stt_stage":            {"sinhala_input": ...},
        "intent_stage":         <full dict from detect_intent_approach1()>,
        "personalization_stage": {repeat_failure, correction_applied,
                                   style_class, style_source},
        "final_prompt":         <the exact payload handed to Component 3>
      }
    """
    # ── Stage 1: STT (already done upstream — sinhala_text IS the STT
    #    output, either from test_samples.py or a live mic/typed input) ──
    stt_stage = {"sinhala_input": sinhala_text}

    # ── Stage 2: Translation + Intent detection (Approach 1) ──
    intent_stage = detect_intent_approach1(sinhala_text)
    intent = intent_stage["intent"]
    english_text = intent_stage["english_translation"]

    # ── Stage 3: Personalization ──
    last = get_last_interaction(user_id)

    # 3a. Correction-signal check: does THIS turn imply the PREVIOUS turn's
    #     style guess was wrong? If so, relabel and re-learn from the
    #     previous turn's text using the corrected label.
    corrected_style = detect_correction_signal(intent, last)
    if corrected_style is not None:
        update_interaction_style_by_timestamp(user_id, last["timestamp"], corrected_style)
        learn_style(user_id, last["english_translation"], corrected_style)

    # 3b. Repeat-failure check — same content, user asking again. Checks
    #     BOTH an explicit "repeat" intent AND literal repetition of the
    #     same Sinhala sentence as last turn (the fix — previously only
    #     the explicit-intent case was checked, so retyping the exact same
    #     question without saying "repeat" was never caught).
    repeat_failure = is_repeat_failure(intent, retrieved_chunk_id, sinhala_text, last)

    # 3c. Log this interaction now (before we know this turn's style)
    log_interaction(user_id, intent_stage, retrieved_chunk_id=retrieved_chunk_id)

    if repeat_failure:
        personalization_stage = {
            "repeat_failure": True,
            "correction_applied": corrected_style,
            "style_class": None,
            "style_source": None,
        }
        final_prompt = {
            "route": "TTS_REPLAY",
            "action": "Replay last response at slower speed",
            "intent": intent,
            "english_translation": english_text,
            "correction_applied": corrected_style,
        }
        return {
            "stt_stage": stt_stage,
            "intent_stage": intent_stage,
            "personalization_stage": personalization_stage,
            "final_prompt": final_prompt,
        }

    # 3d. Decide THIS turn's style, in order of confidence:
    #       1. Explicit intent override (SIMPLIFY/ELABORATE) — user said it outright
    #       2. personalization_flags hint — implicit but LLM-extracted evidence
    #       3. This user's own online model — its guess, which is what learns
    #          and improves over time as this user's turns accumulate
    direct_override = DIRECT_STYLE_OVERRIDE.get(intent.upper()) if intent else None
    flag_hint = _style_hint_from_flags(intent_stage["personalization_flags"])

    if direct_override:
        style = direct_override
        style_source = "direct_override"
    elif flag_hint:
        style = flag_hint
        style_source = "personalization_flags"
    else:
        style = predict_style(user_id, english_text)
        style_source = "model_prediction"

    prompt_modifier = STYLE_PROMPT_MODIFIERS[style]

    # 3e. Update the log with the predicted style for this turn
    update_last_interaction_style(user_id, style)

    # 3f. Learn from this interaction using the style we just decided on
    #     (whichever source it came from — learning from the best label we
    #     currently have makes future predictions better), into THIS user's
    #     own model only.
    learn_style(user_id, english_text, style)

    personalization_stage = {
        "repeat_failure": False,
        "correction_applied": corrected_style,
        "style_class": style,
        "style_source": style_source,
    }

    final_prompt = {
        "route": "GENERATE",
        "intent": intent,
        "english_translation": english_text,
        "style_class": style,
        "prompt_modifier": prompt_modifier,
        "personalization_flags": intent_stage["personalization_flags"],
        "retrieved_chunk_id": retrieved_chunk_id,
        "correction_applied": corrected_style,
    }

    return {
        "stt_stage": stt_stage,
        "intent_stage": intent_stage,
        "personalization_stage": personalization_stage,
        "final_prompt": final_prompt,
    }
