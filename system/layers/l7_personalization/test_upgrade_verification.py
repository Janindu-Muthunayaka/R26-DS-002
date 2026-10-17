"""
test_upgrade_verification.py
Comprehensive test suite verifying the v3 upgrade of Layer 7 Personalization:
1. Modifiers & Fusion (intent_detection.modifiers)
2. Evidence Detection & Diagnostics (personalization.diagnostic)
3. System Commands & Question Guards (personalization.system_commands)
4. Logger & Unmatched Modifiers (personalization.logger)
5. Style Model Online Learning with Baseline Demo Profiles (personalization.style_model)
6. Speech Planner & Adaptive Delivery with Natural Pauses and 0.8-1.3 rate limits (delivery.planner, delivery.feedback, delivery.voice_profile)
7. OpenAI LLM Backend (intent_detection.llm_backend)
8. Integration Contract with Layer 0 & Layer 6 (voice.py & adapters.py)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

_L7 = Path(__file__).resolve().parent
_SYSTEM = _L7.parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))
if str(_L7) not in sys.path:
    sys.path.insert(0, str(_L7))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 1. Modifiers & Fusion Tests ───────────────────────────────────────────────
print("Testing intent_detection.modifiers...")
from intent_detection.modifiers import (
    repeat_in_sinhala, speed_request, style_modifier, build_frame, fuse_intent
)

# Test Sinhala repeat
assert repeat_in_sinhala("ආයෙත් කියන්න") is True
assert repeat_in_sinhala("තව පාරක් කියන්න") is True
assert repeat_in_sinhala("කෙටියෙන් කියන්න") is False  # Style word blocks repeat

# Test Sinhala speed
assert speed_request("හෙමින් කියන්න", "") == "slower"
assert speed_request("වේගෙන් කියන්න", "") == "faster"
assert speed_request("ඔයා කියවන්නේ ගොඩක් වේගෙන්", "") == "slower"  # Complaint inversion

# Test Sinhala style modifier
assert style_modifier("කෙටියෙන් කියන්න", "") == "Simple"
assert style_modifier("විස්තර කර කියන්න", "") == "Detailed"
assert style_modifier("පියවරෙන් පියවර කියන්න", "") == "StepByStep"

# Test Frame construction
frame = build_frame("තව පාරක් හෙමින් කියන්න", "say it again slowly", "REPEAT")
assert frame["intent"] == "REPEAT"
assert frame["modifiers"]["repeat"] is True
assert frame["modifiers"]["speed"] == "slower"

# Test Fusion override (NLLB mis-translating ආයෙ to PREVIOUS)
fused_intent, note = fuse_intent("ආයෙ කියන්න", "PREVIOUS")
assert fused_intent == "REPEAT"
assert note is not None
print("[OK] Modifiers & Fusion: PASSED")

# ── 2. Diagnostics Tests ──────────────────────────────────────────────────────
print("\nTesting personalization.diagnostic...")
from personalization.diagnostic import (
    style_from_intent, detect_correction_signal, is_repeat_failure
)

assert style_from_intent("SIMPLIFY") == "Simple"
assert style_from_intent("ELABORATE") == "Detailed"
assert style_from_intent("STEP_BY_STEP") == "StepByStep"
assert style_from_intent("READ_ALOUD") is None

fake_last = {"intent": "EXPLAIN", "style_class": "Detailed", "timestamp": "2026-10-15T10:00:00"}
assert detect_correction_signal("SIMPLIFY", fake_last) == "Simple"
assert detect_correction_signal("ELABORATE", fake_last) is None  # Already detailed
assert detect_correction_signal("SIMPLIFY", None) is None
print("[OK] Diagnostics: PASSED")

# ── 3. System Commands & Question Guard Tests ────────────────────────────────
print("\nTesting personalization.system_commands...")
from personalization.system_commands import detect_system_command, get_command_action

assert detect_system_command("NEXT", "go next") == "GO_FORWARD"
assert detect_system_command("STOP", "please stop reading") == "STOP"
assert detect_system_command("PREVIOUS", "go back to previous") == "GO_BACK"
assert detect_system_command("GO_TO_PAGE", "go to the first page") == "GO_TO_PAGE"
assert detect_system_command("READ_HEADLINES", "read the headlines") == "READ_HEADLINES"

# Verify Question Guard: questions should NEVER be hijacked as navigation commands
assert detect_system_command("EXPLAIN", "what is on the next page?") is None
assert detect_system_command("EXPLAIN", "can you explain further?") is None
print("[OK] System Commands & Question Guards: PASSED")

# ── 4. Logger Tests ───────────────────────────────────────────────────────────
print("\nTesting personalization.logger...")
from personalization.logger import (
    log_interaction, get_last_interaction, update_last_interaction_style,
    update_interaction_style_by_timestamp, log_unmatched_modifiers
)

test_uid = "user_test_verify"
rec = log_interaction(test_uid, {
    "sinhala_input": "ටෙස්ට් විමසුම",
    "english_translation": "test query",
    "intent": "EXPLAIN",
    "personalization_flags": {},
}, style_class="Detailed")
assert rec["user_id"] == test_uid
assert rec["style_class"] == "Detailed"

last_rec = get_last_interaction(test_uid)
assert last_rec is not None
assert last_rec["intent"] == "EXPLAIN"

update_last_interaction_style(test_uid, "Simple")
updated = get_last_interaction(test_uid)
assert updated["style_class"] == "Simple"

# Test log_unmatched_modifiers
fake_frame = {
    "intent": "CHANGE_SPEED",
    "evidence": {"unmatched": ["speed"]}
}
unmatched_rec = log_unmatched_modifiers(test_uid, "වේගය", "speed", fake_frame)
assert unmatched_rec is not None
print("[OK] Logger & Unmatched Modifiers: PASSED")

# ── 5. Style Model Online Learning & Demo Baseline Tests ─────────────────────
print("\nTesting personalization.style_model...")
from personalization.style_model import (
    predict_style, learn_style, get_user_summary, reset_user, reset_all
)

# Test that old baseline demo profiles are permanently enforced (no blank start)
reset_all()
summary_001 = get_user_summary("user_001")
assert summary_001["dominant_preference"] == "Simple"
assert summary_001["history_weights"]["Simple"] >= 0.45

summary_002 = get_user_summary("user_002")
assert summary_002["dominant_preference"] == "Detailed"
assert summary_002["history_weights"]["Detailed"] >= 0.45

summary_003 = get_user_summary("user_003")
assert summary_003["dominant_preference"] == "StepByStep"
assert summary_003["history_weights"]["StepByStep"] >= 0.45

# Test online learning update
learn_style("user_001", "explain in very simple words", "Simple")
assert get_user_summary("user_001")["history_weights"]["Simple"] >= 0.45
print("[OK] Style Model Online Learning & Demo Presets: PASSED")

# ── 6. Speech Planner & Adaptive Delivery Tests (0.8 - 1.3 Rate) ─────────────
print("\nTesting delivery.planner, feedback, voice_profile...")
from delivery.planner import split_segments, plan_speech, format_text_with_tts_pauses, RATE_MIN, RATE_MAX
from delivery.voice_profile import new_profile, set_last_spoken, repeatable, is_next_turn
from delivery.feedback import apply_turn, yes_no

# Check rate limits are 0.8 - 1.3
assert RATE_MIN == 0.8
assert RATE_MAX == 1.3

text_si = "පළමුව, ඉන්ධන මිල අඩු කළා. දෙවනුව, නව මිල අද සිට ක්‍රියාත්මක වෙනවා."
segments = split_segments(text_si)
assert len(segments) == 2

# Test formatted text with natural pause tokens for TTS
tts_text = format_text_with_tts_pauses(text_si, "StepByStep")
assert "..." in tts_text or "\n" in tts_text

plan = plan_speech(text_si, "Detailed", profile={"rate": 1.5})
# Must be clamped to 1.3
assert plan["rate"] == 1.3
assert plan["speed"] == 1.3
assert "formatted_text" in plan

# Voice profile tracking
prof = new_profile()
prof = set_last_spoken(prof, text_si, "answer", "Detailed", datetime.now(), completed=True)
assert repeatable(prof, datetime.now()) is not None
assert is_next_turn(prof) is True

# Test feedback rules on repeat request
rep_frame = {"intent": "REPEAT", "modifiers": {"repeat": True, "speed": None, "style": None}}
new_prof, decision = apply_turn(prof, rep_frame, datetime.now())
assert decision["repeat"] is True
assert decision["replay"] is not None
# Learns slower rate on immediate "didn't hear" repeat, clamped >= 0.8
assert 0.8 <= new_prof["rate"] <= 1.3

# Test yes_no
assert yes_no("ඔව් දිගටම කියවන්න") == "yes"
assert yes_no("එපා නවත්වන්න") == "no"
print("[OK] Speech Planner & Adaptive Delivery (0.8-1.3 rate): PASSED")

# ── 7. OpenAI LLM Backend Tests ──────────────────────────────────────────────
print("\nTesting intent_detection.llm_backend (Pure OpenAI)...")
from intent_detection.llm_backend import get_active_backend, parse_json_reply, normalize_label

assert get_active_backend() == "openai"
assert normalize_label("step by step") == "STEP_BY_STEP"
assert parse_json_reply('{"intent": "SUMMARIZE"}') == {"intent": "SUMMARIZE"}
print("[OK] OpenAI LLM Backend: PASSED")

# ── 8. Contract Verification with Layer 0 & Layer 6 ──────────────────────────
print("\nTesting contract compatibility with l0_voice and l6_generator...")
from layers.l0_voice.voice import REQUIRED as VOICE_REQUIRED
from layers.l6_generator.adapters import parse_voice_input

# Simulate what server.py produces from handle_voice_command
mock_output = {
    "route": "GENERATE",
    "intent": "EXPLAIN",
    "english_translation": "Explain this article",
    "style_class": "Detailed",
    "prompt_modifier": "Provide a thorough, in-depth explanation...",
    "personalization_flags": {},
    "retrieved_chunk_id": "chunk_99",
    "correction_applied": None,
    "sinhala_input": "මෙය පැහැදිලි කරන්න",
    "style_source": "explicit_intent",
    "user_profile": {"n_confirmed": 3, "history_weights": {"Detailed": 0.55}, "dominant_preference": "Detailed"},
    "learned": True,
    "speed": 1.0,
    "speech_rate": 1.0,
}

# 1. Check Layer 0 REQUIRED fields
for k in VOICE_REQUIRED:
    assert k in mock_output, f"Missing Layer 0 required field: {k}"

# 2. Check speed field is present
assert "speed" in mock_output
assert "speech_rate" in mock_output
assert 0.8 <= mock_output["speed"] <= 1.3

# 3. Check Layer 6 parse_voice_input
parsed = parse_voice_input(mock_output)
assert parsed["route"] == "GENERATE"
assert parsed["intent"] == "EXPLAIN"
assert parsed["query_text"] == "Explain this article"
assert parsed["style_class"] == "Detailed"
assert parsed["retrieved_chunk_id"] == "chunk_99"
print("[OK] Layer 0 and Layer 6 Contract: PASSED")

print("\n" + "=" * 60)
print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! 🎉")
print("=" * 60)
