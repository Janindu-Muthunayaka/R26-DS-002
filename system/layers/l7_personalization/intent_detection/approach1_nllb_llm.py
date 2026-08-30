# Approach 1: NLLB Translation + Llama 3.2 1B via Ollama
# Flow: Sinhala STT → NLLB translates to English → Llama extracts intent JSON

import time
import json
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from layers.l7_personalization.intent_detection.fallback_ollama_openai import extract_intent_fallback

# ─── Load NLLB Translation Model ───────────────────────
print("Loading NLLB translation model...")
print("First run will download ~600MB — please wait...")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("✅ NLLB model loaded successfully!")


# ─── Translate Sinhala to English ──────────────────────
def translate_to_english(sinhala_text):
    inputs = tokenizer(
        sinhala_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200
    )

    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=200
    )

    return tokenizer.batch_decode(
        translated,
        skip_special_tokens=True
    )[0]


# ─── Extract Intent via Ollama Llama 3.2 1B ────────────
def extract_intent_llama(english_text):
    system_prompt = system_prompt = """You are an intelligent assistant that understands what a user wants.

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
   - "STEP_BY_STEP" — user wants it broken into ordered steps
     (e.g. "explain step by step", "one by one", "walk me through it")
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
   - "detail_level": "brief", "detailed", or "step_by_step"  
   - "language_style": "simple" or "technical"
   If none are mentioned, return empty object {}.

Return ONLY the JSON. No explanation. No markdown. No extra text.
Example output: {"intent": "SUMMARIZE", "personalization_flags": {"detail_level": "brief"}}"""

    user_prompt = f"User command: '{english_text}'"

    raw = extract_intent_fallback(system_prompt, user_prompt)

    # Clean response — remove markdown if model adds it
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Prevent Llama 1B hallucinated JSON from crashing the entire server
        return {"intent": "UNKNOWN", "personalization_flags": {}}


# ─── Full Pipeline ──────────────────────────────────────
def detect_intent_approach1(sinhala_text):
    """
    Full Approach 1 pipeline:
    Sinhala → NLLB English → Llama JSON
    """
    # Step 1 — Translate
    t1 = time.time()
    english_text = translate_to_english(sinhala_text)
    translation_time = time.time() - t1

    # Step 2 — Extract intent
    t2 = time.time()
    result = extract_intent_llama(english_text)
    llm_time = time.time() - t2

    total_time = translation_time + llm_time

    return {
        "approach":              "Approach 1 — NLLB + Llama3.2:1b",
        "sinhala_input":         sinhala_text,
        "english_translation":   english_text,
        "intent":                result.get("intent", "UNKNOWN"),
        "personalization_flags": result.get("personalization_flags", {}),
        "translation_time_sec":  round(translation_time, 3),
        "llm_time_sec":          round(llm_time, 3),
        "total_time_sec":        round(total_time, 3)
    }


# ─── Run All 6 Test Samples ─────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.test_samples import test_samples

    print("=" * 60)
    print("APPROACH 1 — NLLB + Llama3.2:1b")
    print("=" * 60)

    correct_count = 0
    total_time_all = 0

    for sample in test_samples:
        print(f"\nID       : {sample['id']}")
        print(f"Input    : {sample['stt_output']}")
        print(f"Expected : {sample['expected_intent']}")

        result = detect_intent_approach1(sample["stt_output"])
        correct = result["intent"] == sample["expected_intent"]

        if correct:
            correct_count += 1
        total_time_all += result["total_time_sec"]

        print(f"English  : {result['english_translation']}")
        print(f"Intent   : {result['intent']} {'✅' if correct else '❌'}")
        print(f"Flags    : {result['personalization_flags']}")
        print(f"Trans    : {result['translation_time_sec']}s")
        print(f"LLM      : {result['llm_time_sec']}s")
        print(f"Total    : {result['total_time_sec']}s")
        print("-" * 60)

    accuracy = (correct_count / len(test_samples)) * 100
    avg_time = total_time_all / len(test_samples)

    print(f"\n{'=' * 60}")
    print(f"APPROACH 1 FINAL RESULTS")
    print(f"Correct  : {correct_count}/{len(test_samples)}")
    print(f"Accuracy : {accuracy:.1f}%")
    print(f"Avg Time : {avg_time:.3f}s")
    print(f"{'=' * 60}")