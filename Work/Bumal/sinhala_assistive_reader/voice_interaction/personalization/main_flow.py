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
