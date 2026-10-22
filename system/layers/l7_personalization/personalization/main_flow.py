# personalization/main_flow.py
# Full personalization flow: Sinhala input -> personalized prompt for
# Component 3 (RAG generation) + Adaptive Speed & Delivery determination.
#
# Pure pipeline logic — no printing. Returns one structured result broken
# into stages so any caller (app.py, compare_users.py, a future backend)
# can inspect every decision.
#
# DECISION ORDER for this turn's style:
#   1. Explicit intent      (SIMPLIFY / ELABORATE / STEP_BY_STEP)  -> obey outright
#   2. Style words in the English translation (text keywords)      -> obey outright
#   3. The user's own model (prediction from their history)        -> only if 1 & 2 silent
#
# SPEED RATE:
#   Determined dynamically per user profile and clamped strictly between 0.8 and 1.3.

from datetime import datetime

try:
    from intent_detection.approach1_nllb_llm import detect_intent_approach1
    from intent_detection.modifiers import build_frame
    from personalization.logger import (
        log_interaction, log_unmatched_modifiers, get_last_interaction,
        update_last_interaction_style, update_interaction_style_by_timestamp
    )
    from personalization.diagnostic import (
        style_from_intent, style_from_keywords, detect_correction_signal
    )
    from personalization.style_model import predict_style, learn_style, get_user_summary, record_turn
    from personalization.system_commands import detect_system_command, get_command_action
    from delivery.voice_profile import (
        get_profile as get_delivery_profile, save_profile as save_delivery_profile
    )
    from delivery.feedback import apply_turn as apply_delivery_turn
except ImportError:
    from layers.l7_personalization.intent_detection.approach1_nllb_llm import detect_intent_approach1
    from layers.l7_personalization.intent_detection.modifiers import build_frame
    from layers.l7_personalization.personalization.logger import (
        log_interaction, log_unmatched_modifiers, get_last_interaction,
        update_last_interaction_style, update_interaction_style_by_timestamp
    )
    from layers.l7_personalization.personalization.diagnostic import (
        style_from_intent, style_from_keywords, detect_correction_signal
    )
    from layers.l7_personalization.personalization.style_model import predict_style, learn_style, get_user_summary, record_turn
    from layers.l7_personalization.personalization.system_commands import detect_system_command, get_command_action
    from layers.l7_personalization.delivery.voice_profile import (
        get_profile as get_delivery_profile, save_profile as save_delivery_profile
    )
    from layers.l7_personalization.delivery.feedback import apply_turn as apply_delivery_turn

STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.",
    "StepByStep": "Break the explanation into clear numbered steps, one action or idea per step.",
}


# Turns that only change HOW the last answer is spoken (TURN_CONDITIONS.xlsx
# section 1). They are routed LOCAL and are never style evidence: "more
# pauses" contains "more", which the keyword list reads as Detailed.
DELIVERY_ONLY_INTENTS = {"CHANGE_SPEED", "UNKNOWN", "READ_ALOUD"}


def _is_delivery_only(intent, mods):
    if mods.get("style") or mods.get("repeat"):
        return False
    if not (mods.get("speed") or mods.get("pause") or mods.get("chunk") or mods.get("volume")):
        return intent == "CHANGE_SPEED"
    return intent in DELIVERY_ONLY_INTENTS


def _with_last_spoken(profile, ctx, now):
    """The main server owns what was last spoken (per session). It tells us
    whether there is something to replay and whether this is the very next
    turn, which is all feedback.apply_turn needs for its REPEAT rules."""
    if not isinstance(ctx, dict):
        return profile
    if not ctx.get("has_last"):
        profile["last_spoken"] = None
        return profile
    age = float(ctx.get("age_s") or 0)
    profile["last_spoken"] = {
        "text": "", "kind": "answer", "style": None, "completed": True, "resume_at": None,
        "finished_at": datetime.fromtimestamp(now.timestamp() - age).isoformat(timespec="seconds"),
        "turns_since": 0 if ctx.get("next_turn") else 1,
    }
    return profile


def _delivery_fields(profile, before):
    return {
        "speed": round(max(0.8, min(1.3, float(profile.get("rate", 1.0)))), 2),
        "pause_scale": round(float(profile.get("pause_scale", 1.0)), 2),
        "chunk_size": int(profile.get("chunk_size", 3)),
        "delivery_before": before,
    }


# Control intents from intent_detection/sinhala_rules.py -> system command.
RULE_COMMANDS = {"NEXT": "GO_FORWARD", "PREVIOUS": "GO_BACK", "STOP": "STOP",
                 "GO_TO_PAGE": "GO_TO_PAGE", "READ_HEADLINES": "READ_HEADLINES"}


def handle_voice_command(sinhala_text, user_id, retrieved_chunk_id=None, dry_run=False,
                         last_spoken_ctx=None):
    """
    Runs one full turn.
    retrieved_chunk_id: optional chunk identifier from Component 3.
    dry_run: decide the style for this user WITHOUT side effects - nothing is
             logged, learned, corrected or saved. Used to re-generate an old
             answer under a different user without touching their profile.
    last_spoken_ctx: {"has_last", "next_turn", "age_s"} from the main server,
             which holds what was last spoken in this session.

    Returns:
      {
        "stt_stage":             {...},
        "intent_stage":          <detect_intent_approach1() output>,
        "personalization_stage": {...decisions and why...},
        "final_prompt":          <payload for Component 3 and Reader App>,
        "frame":                 <intent_detection.modifiers.build_frame() output:
                                  {intent, modifiers, evidence}>
      }
    """
    now = datetime.now()

    # ── Stage 1: STT output ──
    stt_stage = {"sinhala_input": sinhala_text}

    # ── Stage 2: Translation + intent detection ──
    intent_stage = detect_intent_approach1(sinhala_text)
    intent = intent_stage["intent"]
    english_text = intent_stage["english_translation"]
    frame = build_frame(sinhala_text, english_text, intent)
    if not dry_run:
        log_unmatched_modifiers(user_id, sinhala_text, english_text, frame)

    # ── Delivery & Speed determination (0.8 - 1.3) ──
    delivery_profile = get_delivery_profile(user_id)
    before = {k: delivery_profile.get(k) for k in ("rate", "pause_scale", "chunk_size")}
    mods = frame.get("modifiers", {})
    # Every turn, not only speed turns: a turn with no delivery request moves
    # the rate back towards normal (delivery/feedback.py, RELAX_FACTOR).
    if not dry_run:
        delivery_profile = _with_last_spoken(delivery_profile, last_spoken_ctx, now)
        delivery_profile, _ = apply_delivery_turn(delivery_profile, frame, now)
        save_delivery_profile(user_id, delivery_profile)
    delivery = _delivery_fields(delivery_profile, before)

    # Ensure rate is strictly clamped between 0.8 and 1.3
    user_speed = delivery["speed"]

    # ── Delivery-only turn: change how it sounds, never what it says ──
    if _is_delivery_only(intent, mods):
        last = get_last_interaction(user_id)
        style = (last or {}).get("style_class") or "Detailed"
        if style not in STYLE_PROMPT_MODIFIERS:
            style = "Detailed"
        if not dry_run:
            record_turn(user_id)
        return {
            "stt_stage": stt_stage,
            "intent_stage": intent_stage,
            "personalization_stage": {
                "is_system_command": False, "command": None,
                "style_class": style, "style_source": "delivery_only",
                "learned": False, "correction_applied": None,
                "user_profile": get_user_summary(user_id),
                "speed": user_speed, "delivery_profile": delivery_profile,
            },
            "final_prompt": {
                "route": "LOCAL",
                "intent": "CHANGE_SPEED" if intent == "UNKNOWN" else intent,
                "english_translation": english_text,
                "style_class": style,
                "prompt_modifier": STYLE_PROMPT_MODIFIERS[style],
                "personalization_flags": {},
                "retrieved_chunk_id": retrieved_chunk_id,
                "correction_applied": None,
                "speech_rate": user_speed,
                **delivery,
            },
            "frame": frame,
        }

    # ── Stage 3: System command check (before anything personalization) ──
    # Navigation commands carry no style information, so they bypass the
    # model entirely and are never used as training evidence.
    command = detect_system_command(intent, english_text)
    if not command and intent_stage.get("intent_source") == "sinhala_rule":
        # The Sinhala rules recognised a control command. Its English may be a
        # broken translation ("Next time we go together"), so do not let the
        # English decide - and never learn a style from it.
        command = RULE_COMMANDS.get(intent)
    if command:
        if not dry_run:
            log_interaction(user_id, intent_stage, retrieved_chunk_id=retrieved_chunk_id, style_class=None)
            record_turn(user_id)
        return {
            "stt_stage": stt_stage,
            "intent_stage": intent_stage,
            "personalization_stage": {
                "is_system_command": True,
                "command": command,
                "style_class": "Detailed",
                "style_source": None,
                "learned": False,
                "correction_applied": None,
                "user_profile": get_user_summary(user_id),
                "speed": user_speed,
                "delivery_profile": delivery_profile,
            },
            "final_prompt": {
                "route": "SYSTEM_COMMAND",
                "command": command,
                "action": get_command_action(command),
                "intent": intent,
                "english_translation": english_text,
                "style_class": "Detailed",
                "prompt_modifier": STYLE_PROMPT_MODIFIERS["Detailed"],
                "personalization_flags": {},
                "retrieved_chunk_id": retrieved_chunk_id,
                "correction_applied": None,
                "speech_rate": user_speed,
                **delivery,
            },
            "frame": frame,
        }

    # ── Stage 4: Personalization ──
    last = get_last_interaction(user_id)

    # 4a. Correction check — does this turn say the last turn was wrong?
    corrected_style = None if dry_run else detect_correction_signal(intent, last)
    if corrected_style is not None:
        update_interaction_style_by_timestamp(user_id, last["timestamp"], corrected_style)
        # A correction is strong, unambiguous evidence — always learn from it.
        learn_style(user_id, last["english_translation"], corrected_style)

    # 4b. Log this turn (style filled in below)
    if not dry_run:
        log_interaction(user_id, intent_stage, retrieved_chunk_id=retrieved_chunk_id)

    # 4c. Decide this turn's style, in strict priority order
    intent_style = style_from_intent(intent)
    keyword_style = style_from_keywords(english_text)

    if intent_style:
        style, style_source, is_evidence = intent_style, "explicit_intent", True
    elif keyword_style:
        style, style_source, is_evidence = keyword_style, "text_keywords", True
    else:
        style, style_source = predict_style(user_id, english_text)
        is_evidence = False

    prompt_modifier = STYLE_PROMPT_MODIFIERS[style]

    # 4d. Record the decided style on this turn's log entry
    if not dry_run:
        update_last_interaction_style(user_id, style)

    # 4e. Learn ONLY from real evidence. If the style came from the model's
    #     own prediction, we deliberately do not train on it.
    if is_evidence and not dry_run:
        learn_style(user_id, english_text, style)
    elif not dry_run and corrected_style is None:
        record_turn(user_id, style)          # learn_style already added a point

    return {
        "stt_stage": stt_stage,
        "intent_stage": intent_stage,
        "personalization_stage": {
            "is_system_command": False,
            "command": None,
            "style_class": style,
            "style_source": style_source,
            "learned": is_evidence and not dry_run,
            "correction_applied": corrected_style,
            "user_profile": get_user_summary(user_id),
            "speed": user_speed,
            "delivery_profile": delivery_profile,
        },
        "final_prompt": {
            "route": "GENERATE",
            "intent": intent,
            "english_translation": english_text,
            "style_class": style,
            "prompt_modifier": prompt_modifier,
            "personalization_flags": {},   # kept for the Component 3 contract
            "retrieved_chunk_id": retrieved_chunk_id,
            "correction_applied": corrected_style,
            "speech_rate": user_speed,
            **delivery,
        },
        "frame": frame,
    }
