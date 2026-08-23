# Approach 2: Direct Sinhala → Qwen2.5:1.5b via Ollama
# Flow: Sinhala STT → Qwen extracts intent JSON directly (no translation)

import time
import json
import requests


# ─── Extract Intent via Ollama Qwen2.5 1.5B ────────────
def extract_intent_qwen(sinhala_text):
    system_prompt = """You are an intelligent assistant that understands what a user wants.

A visually impaired user is interacting with an assistive reading device.
They will give you a voice command. Your job is to understand the TRUE MEANING 
of what they want and extract it as structured JSON.

You must return ONLY a valid JSON object with exactly two keys:

1. "intent": A short English verb phrase describing what the user wants to do.
   Think freely — do not limit yourself to a fixed list.
   Examples of good intents:
   - "SUMMARIZE" — user wants a summary
   - "EXPLAIN" — user wants something explained
   - "SIMPLIFY" — user wants simpler language
   - "ELABORATE" — user wants more detail
   - "REPHRASE" — user wants it said differently
   - "IDENTIFY_CONTENT" — user wants to know what something is
   - "READ_ALOUD" — user wants text read out
   - "STOP" — user wants to stop
   - "REPEAT" — user wants it said again
   - "NEXT" — user wants to move forward
   If none of the above fit, create a short clear verb phrase that best 
   describes what the user wants.

2. "personalization_flags": A JSON object extracting HOW the user wants it done.
   Look for mentions of:
   - "speed": "fast" or "slow"
   - "detail_level": "brief" or "detailed"  
   - "language_style": "simple" or "technical"
   If none are mentioned, return empty object {}.

Return ONLY the JSON. No explanation. No markdown. No extra text.
Example output: {"intent": "SUMMARIZE", "personalization_flags": {"detail_level": "brief"}}"""

    user_prompt = f"Analyze this Sinhala command: '{sinhala_text}'"

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:1.5b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 100
            }
        }
    )

    raw = response.json()["message"]["content"].strip()

    # Clean response — remove markdown if model adds it
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    return json.loads(raw)


# ─── Full Pipeline ──────────────────────────────────────
def detect_intent_approach2(sinhala_text):
    """
    Full Approach 2 pipeline:
    Sinhala → Qwen2.5 JSON directly
    """
    t1 = time.time()
    result = extract_intent_qwen(sinhala_text)
    total_time = time.time() - t1

    return {
        "approach":              "Approach 2 — Direct Qwen2.5:1.5b",
        "sinhala_input":         sinhala_text,
        "english_translation":   "N/A — no translation step",
        "intent":                result.get("intent", "UNKNOWN"),
        "personalization_flags": result.get("personalization_flags", {}),
        "translation_time_sec":  0,
        "llm_time_sec":          round(total_time, 3),
        "total_time_sec":        round(total_time, 3)
    }


# ─── Run All 6 Test Samples ─────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.test_samples import test_samples

    print("=" * 60)
    print("APPROACH 2 — Direct Qwen2.5:1.5b ")
    print("=" * 60)

    correct_count = 0
    total_time_all = 0

    for sample in test_samples:
        print(f"\nID       : {sample['id']}")
        print(f"Input    : {sample['stt_output']}")
        print(f"Expected : {sample['expected_intent']}")

        result = detect_intent_approach2(sample["stt_output"])
        correct = result["intent"] == sample["expected_intent"]

        if correct:
            correct_count += 1
        total_time_all += result["total_time_sec"]

        print(f"Intent   : {result['intent']} {'✅' if correct else '❌'}")
        print(f"Flags    : {result['personalization_flags']}")
        print(f"Total    : {result['total_time_sec']}s")
        print("-" * 60)

    accuracy = (correct_count / len(test_samples)) * 100
    avg_time = total_time_all / len(test_samples)

    print(f"\n{'=' * 60}")
    print(f"APPROACH 2 FINAL RESULTS")
    print(f"Correct  : {correct_count}/{len(test_samples)}")
    print(f"Accuracy : {accuracy:.1f}%")
    print(f"Avg Time : {avg_time:.3f}s")
    print(f"{'=' * 60}")