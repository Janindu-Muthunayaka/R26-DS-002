# personalization/main_flow.py
# Full personalization flow: Sinhala input -> personalized prompt for
# Component 3 (RAG generation).
#
# Pure pipeline logic — no printing. Returns one structured result broken
# into stages so any caller (app.py, compare_users.py, a future backend)
# can inspect every decision.
#
# DECISION ORDER for this turn's style:
#   1. Explicit intent      (SIMPLIFY / ELABORATE / STEP_BY_STEP)  -> obey outright
#   2. Explicit flags       (language_style / detail_level)        -> obey outright
#   3. The user's own model (prediction from their history)        -> only if 1 & 2 silent
#
# LEARNING RULE: only cases 1 and 2 (plus corrections) are real user
# evidence, and only they train the model. Case 3 is a prediction and is
# NEVER learned from — learning from your own guess is the self-training
# loop that made the previous version a majority-class predictor.

from intent_detection.approach1_nllb_llm import detect_intent_approach1
from personalization.logger import (
    log_interaction, get_last_interaction,
    update_last_interaction_style, update_interaction_style_by_timestamp
)
from personalization.diagnostic import (
    style_from_intent, style_from_flags, style_from_keywords, detect_correction_signal
)
from personalization.style_model import predict_style, learn_style, get_user_summary
from personalization.system_commands import detect_system_command, get_command_action

STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.",
    "StepByStep": "Break the explanation into clear numbered steps, one action or idea per step.",
}


def handle_voice_command(sinhala_text, user_id):
    """
    Runs one full turn.

    Returns:
      {
        "stt_stage":             {...},
        "intent_stage":          <detect_intent_approach1() output>,
        "personalization_stage": {...decisions and why...},
        "final_prompt":          <payload for Component 3>
      }
    """
    # ── Stage 1: STT output ──
    stt_stage = {"sinhala_input": sinhala_text}

    # ── Stage 2: Translation + intent detection ──
    intent_stage = detect_intent_approach1(sinhala_text)
    intent = intent_stage["intent"]
    english_text = intent_stage["english_translation"]
    flags = intent_stage.get("personalization_flags", {})

    # ── Stage 3: System command check (before anything personalization) ──
    # Navigation commands carry no style information, so they bypass the
    # model entirely and are never used as training evidence.
    command = detect_system_command(intent, english_text)
    if command:
        log_interaction(user_id, intent_stage, style_class=None)
        return {
            "stt_stage": stt_stage,
            "intent_stage": intent_stage,
            "personalization_stage": {
                "is_system_command": True,
                "command": command,
                "style_class": None,
                "style_source": None,
                "learned": False,
                "correction_applied": None,
            },
            "final_prompt": {
                "route": "SYSTEM_COMMAND",
                "command": command,
                "action": get_command_action(command),
                "intent": intent,
                "english_translation": english_text,
            },
        }

    # ── Stage 4: Personalization ──
    last = get_last_interaction(user_id)

    # 4a. Correction check — does this turn say the last turn was wrong?
    corrected_style = detect_correction_signal(intent, last)
    if corrected_style is not None:
        update_interaction_style_by_timestamp(user_id, last["timestamp"], corrected_style)
        # A correction is strong, unambiguous evidence — always learn from it.
        learn_style(user_id, last["english_translation"], corrected_style)

    # 4b. Log this turn (style filled in below)
    log_interaction(user_id, intent_stage)

    # 4c. Decide this turn's style, in strict priority order
    intent_style = style_from_intent(intent)
    # Flags are grounded against the translation — an unsupported flag from
    # the 1B model is rejected rather than trusted (see diagnostic.py).
    flag_style = style_from_flags(flags, english_text)
    keyword_style = style_from_keywords(english_text)

    if intent_style:
        style, style_source, is_evidence = intent_style, "explicit_intent", True
    elif flag_style:
        style, style_source, is_evidence = flag_style, "explicit_flag", True
    elif keyword_style:
        style, style_source, is_evidence = keyword_style, "text_keywords", True
    else:
        style, style_source = predict_style(user_id, english_text)
        is_evidence = False

    prompt_modifier = STYLE_PROMPT_MODIFIERS[style]

    # 4d. Record the decided style on this turn's log entry
    update_last_interaction_style(user_id, style)

    # 4e. Learn ONLY from real evidence. If the style came from the model's
    #     own prediction, we deliberately do not train on it.
    if is_evidence:
        learn_style(user_id, english_text, style)

    return {
        "stt_stage": stt_stage,
        "intent_stage": intent_stage,
        "personalization_stage": {
            "is_system_command": False,
            "command": None,
            "style_class": style,
            "style_source": style_source,
            "learned": is_evidence,
            "correction_applied": corrected_style,
            "user_profile": get_user_summary(user_id),
        },
        "final_prompt": {
            "route": "GENERATE",
            "intent": intent,
            "english_translation": english_text,
            "style_class": style,
            "prompt_modifier": prompt_modifier,
            "personalization_flags": flags,
            "correction_applied": corrected_style,
        },
    }
