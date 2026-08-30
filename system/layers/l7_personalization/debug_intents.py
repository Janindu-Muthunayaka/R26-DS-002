"""
debug_intents.py — shows exactly what NLLB and Llama produce for each
teaching phrase, plus how the personalization layer interprets it.

Run this when a teaching line gets the wrong style. It separates the three
possible culprits:
    1. NLLB translated the Sinhala badly
    2. Llama extracted the wrong intent
    3. Llama extracted the wrong personalization_flags

Usage:
    python debug_intents.py
"""

from intent_detection.approach1_nllb_llm import detect_intent_approach1
from personalization.diagnostic import style_from_intent, style_from_flags, style_from_keywords
from compare_users import TEACHING, NEUTRAL_QUESTION


def check(sinhala, expected=None):
    r = detect_intent_approach1(sinhala)
    intent = r["intent"]
    flags = r.get("personalization_flags", {})

    from_intent = style_from_intent(intent)
    from_flags = style_from_flags(flags, r["english_translation"])
    from_keywords = style_from_keywords(r["english_translation"])
    decided = from_intent or from_flags or from_keywords
    source = ("explicit_intent" if from_intent
              else "explicit_flag" if from_flags
              else "text_keywords" if from_keywords
              else "model")

    ok = "" if expected is None else ("  OK" if decided == expected else f"  WRONG (wanted {expected})")

    print(f"\n  Sinhala     : {sinhala}")
    print(f"  NLLB says   : {r['english_translation']}")
    print(f"  Llama intent: {intent}")
    print(f"  Llama flags : {flags}")
    print(f"  -> style    : {decided}  via {source}{ok}")


EXPECTED = {
    "user_001": "Simple",
    "user_002": "Detailed",
    "user_003": "StepByStep",
}

if __name__ == "__main__":
    print("=" * 70)
    print("INTENT / FLAG DIAGNOSTIC")
    print("=" * 70)

    for user_id, lines in TEACHING.items():
        print(f"\n--- {user_id}  (every line should give {EXPECTED[user_id]}) ---")
        for line in lines:
            check(line, EXPECTED[user_id])

    print("\n--- neutral question (should give None / model) ---")
    check(NEUTRAL_QUESTION, None)
    print()
