# Phase 2b test set (ADAPTIVE_VOICE_PLAN.md). Drafted by Claude, Sinhala checked and
# corrected by Bumal on 13 Oct 2026 (one mixed slow/fast sentence removed).
#
# Two parts:
#   1. EXISTING_MODIFIER_LABELS: modifier labels for the existing 45 commands in
#      test_samples.py (only the ones that carry a modifier; all others = NONE).
#   2. multi_samples: 16 new commands the 45 do not cover (slower/faster, and one
#      action + modifiers). None of these appear in test_samples.py or training data.
#
# modifiers: repeat (bool), speed ("slower" | "faster" | None), style ("Simple" |
# "Detailed" | "StepByStep" | None)

NONE = {"repeat": False, "speed": None, "style": None}

# sam_2 and sam_16: style labels corrected by Bumal (Sinhala speaker) on 14 Oct,
# after the Phase 2b first run (were None).
EXISTING_MODIFIER_LABELS = {
    "sam_2":  {"repeat": False, "speed": None,     "style": "Detailed"}, # a little more detail
    "sam_3":  {"repeat": False, "speed": None,     "style": "Simple"},  # explain very simply and briefly
    "sam_5":  {"repeat": False, "speed": "faster", "style": "Simple"},  # briefly, fast — NLLB drops "fast"
    "sam_8":  {"repeat": False, "speed": None,     "style": "Simple"},  # briefly as a summary
    "sam_16": {"repeat": False, "speed": None,     "style": "Simple"},   # say it easily
    "sam_18": {"repeat": False, "speed": None,     "style": "Simple"},  # in simple language
    "sam_24": {"repeat": False, "speed": None,     "style": None},      # "change it and say again" = REPHRASE, NOT repeat
    "sam_31": {"repeat": True,  "speed": None,     "style": None},      # ආයෙ -> NLLB "I'm going back."
    "sam_32": {"repeat": True,  "speed": None,     "style": None},
    "sam_33": {"repeat": True,  "speed": None,     "style": None},      # තව පාරක් -> NLLB garbled
    "sam_34": {"repeat": True,  "speed": None,     "style": None},
    "sam_35": {"repeat": True,  "speed": None,     "style": None},
}

multi_samples = [
    # ── slower / faster only ─────────────────────────────
    {"id": "m_01", "stt_output": "ටිකක් හෙමින් කියවන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None}, "description": "read a bit slower"},
    {"id": "m_02", "stt_output": "හෙමින් කියන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None}, "description": "say it slowly"},
    {"id": "m_03", "stt_output": "ඔයා කියවන්නේ ගොඩක් වේගෙන්", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None}, "description": "too fast -> means slower"},
    {"id": "m_04", "stt_output": "ටිකක් ඉක්මනට කියවන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "faster", "style": None}, "description": "read a bit faster"},
    
    {"id": "m_05", "stt_output": "වේගය අඩු කරන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None}, "description": "reduce the speed"},
    {"id": "m_06", "stt_output": "වේගය වැඩි කරන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "faster", "style": None}, "description": "increase the speed"},
    # ── one action + modifiers ───────────────────────────
    {"id": "m_07", "stt_output": "ආයෙත් හෙමින් කියන්න", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": "slower", "style": None}, "description": "again, slowly"},
    {"id": "m_08", "stt_output": "මට ඇහුනේ නෑ, ටිකක් හෙමින් ආයෙ කියන්න", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": "slower", "style": None}, "description": "didn't hear, again slower"},
    {"id": "m_09", "stt_output": "ඒක ආයෙත් ඉක්මනට කියන්න", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": "faster", "style": None}, "description": "again, faster"},
    {"id": "m_10", "stt_output": "තව පාරක් කියන්න", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": None, "style": None}, "description": "one more time"},
    {"id": "m_11", "stt_output": "හෙමින් ලිපිය කියවන්න", "expected_intent": "READ_ALOUD",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None}, "description": "read the article slowly"},
    {"id": "m_12", "stt_output": "කෙටියෙන් කියන්න කවුද දිනුවේ", "expected_intent": "ASK_QUESTION",
     "expected_modifiers": {"repeat": False, "speed": None, "style": "Simple"}, "description": "briefly, who won"},
    {"id": "m_13", "stt_output": "සරල වචන වලින් ආයෙත් පැහැදිලි කරන්න", "expected_intent": "SIMPLIFY",
     "expected_modifiers": {"repeat": False, "speed": None, "style": "Simple"}, "description": "again in simple words = style, NOT repeat"},
    {"id": "m_14", "stt_output": "ඊළඟ පිටුවට ගිහින් හෙමින් කියවන්න", "expected_intent": "NEXT",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None}, "description": "next page, read slowly"},
    {"id": "m_15", "stt_output": "පියවරෙන් පියවර හෙමින් පැහැදිලි කරන්න", "expected_intent": "STEP_BY_STEP",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": "StepByStep"}, "description": "step by step, slowly"},
    {"id": "m_16", "stt_output": "නවත්තලා ඊළඟ එක කියවන්න", "expected_intent": "NEXT",
     "expected_modifiers": {"repeat": False, "speed": None, "style": None}, "description": "two actions: last one wins"},
]
