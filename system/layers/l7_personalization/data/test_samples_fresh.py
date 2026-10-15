# Phase 2b HELD-OUT test set (ADAPTIVE_VOICE_PLAN.md). Sinhala sentences written by
# Bumal on 13 Oct 2026, AFTER Phase 2 was committed, without looking at the keyword
# lists in delivery/feedback.py. Labels proposed by Claude; the four borderline ones
# (f_10, f_13, f_14, f_17) were confirmed unchanged by Bumal on 14 Oct.
#
# RULE: do NOT tune keyword lists or prompts on these sentences before the first
# Phase 2b run. The first-run score is the held-out result reported in the thesis.
# Fixes made after error analysis are reported separately.
#
# Same format as multi_samples in test_samples_multi.py.
# modifiers: repeat (bool), speed ("slower" | "faster" | None), style ("Simple" |
# "Detailed" | "StepByStep" | None)

fresh_samples = [
    # --- Repeat ---
    {"id": "f_01", "stt_output": "හරියට ඇහුනේ නෑ", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": None, "style": None},
     "description": "didn't hear properly"},
    {"id": "f_02", "stt_output": "නැවත කියනවද", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": None, "style": None},
     "description": "will you say it again?"},
    {"id": "f_03", "stt_output": "ආපසු කියන්න", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": None, "style": None},
     "description": "say it again (ආපසු + කියන්න = repeat, not go back)"},
    {"id": "f_04", "stt_output": "සමාවෙන්න ආපහු කියන්න පුළුවන්ද", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": None, "style": None},
     "description": "sorry, can you say it again? (spelling variant ආපහු)"},

    # --- Speed only ---
    {"id": "f_05", "stt_output": "පොඩ්ඩක් හෙමින් කියනවද", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None},
     "description": "will you say it a little slower?"},
    {"id": "f_06", "stt_output": "වේගෙ වැඩියි හෙමින් කියන්නකෝ", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None},
     "description": "too fast, say it slowly (contains වැඩි = trap for 'faster')"},
    {"id": "f_07", "stt_output": "වේගය අඩුකර කියන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "slower", "style": None},
     "description": "reduce the speed and say it"},
    {"id": "f_08", "stt_output": "වේගෙන් කියන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "faster", "style": None},
     "description": "say it fast"},
    {"id": "f_09", "stt_output": "තව ටිකක් වේගය වැඩිකර කියන්න", "expected_intent": "CHANGE_SPEED",
     "expected_modifiers": {"repeat": False, "speed": "faster", "style": None},
     "description": "increase the speed a bit more (තව here is NOT repeat)"},

    # --- Repeat + speed (two requests in one sentence) ---
    {"id": "f_10", "stt_output": "තේරුනේ නෑ නැවත හෙමින් කියන්න", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": "slower", "style": None},
     "description": "'didn't understand' but explicitly asks repeat + slower"},
    {"id": "f_11", "stt_output": "තව එකපාරක් හෙමින් කියන්න පුළුවන්ද", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": "slower", "style": None},
     "description": "one more time, slowly (තව එකපාරක් variant)"},
    {"id": "f_12", "stt_output": "ආපසු කියන්න පොඩ්ඩක් හෙමින්", "expected_intent": "REPEAT",
     "expected_modifiers": {"repeat": True, "speed": "slower", "style": None},
     "description": "say it again, a little slower"},

    # --- No repeat / no speed (false-trigger checks) ---
    {"id": "f_13", "stt_output": "ඔබට පේන දේ මට මුලින් කියන්න", "expected_intent": "IDENTIFY_CONTENT",
     "expected_modifiers": {"repeat": False, "speed": None, "style": None},
     "description": "tell me first what you see"},
    {"id": "f_14", "stt_output": "දැකපු දේ විස්තර කර කියන්න", "expected_intent": "ELABORATE",
     "expected_modifiers": {"repeat": False, "speed": None, "style": "Detailed"},
     "description": "describe what you saw in detail"},
    {"id": "f_15", "stt_output": "ඩෙංගු රෝගීන් කීයක් පමණ වාර්තා වුණාද", "expected_intent": "ASK_QUESTION",
     "expected_modifiers": {"repeat": False, "speed": None, "style": None},
     "description": "how many dengue patients were reported?"},
    {"id": "f_16", "stt_output": "කුමන අවුරුද්දෙද වුණේ", "expected_intent": "ASK_QUESTION",
     "expected_modifiers": {"repeat": False, "speed": None, "style": None},
     "description": "which year did it happen? (follow-up)"},
    {"id": "f_17", "stt_output": "රෝගීන් සංඛ්‍යාව කීයක්ද කිව්වේ", "expected_intent": "ASK_QUESTION",
     "expected_modifiers": {"repeat": False, "speed": None, "style": None},
     "description": "'how many patients did you say?' = partial repeat of a missed number"},
]
