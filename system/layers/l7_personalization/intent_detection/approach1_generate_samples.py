# Approach 1: NLLB Translation + Llama 3.2 1B via Ollama
# Flow: Sinhala STT → NLLB translates to English → Llama extracts intent JSON
# Modified to run over generate_samples.py dataset

import time
import json
import requests
import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    system_prompt = """You are an intent classifier for an assistive reading device used by visually impaired users.

A user gives a voice command. You must classify it into exactly ONE of the following intents and extract personalization flags.

You must return ONLY a valid JSON object with exactly two keys:

1. "intent": MUST be exactly one of these values (choose the best match):
   - "NEXT_PAGE" — user wants to go to the next page
   - "PREVIOUS_PAGE" — user wants to go back to the previous page
   - "NEXT_ARTICLE" — user wants to read the next article/news item
   - "PREVIOUS_ARTICLE" — user wants to go back to the previous article/news item
   - "SUMMARIZE" — user wants a summary or brief version of the content
   - "EXPLAIN" — user wants something explained or clarified in detail
   - "SIMPLIFY" — user wants content in simpler, easier language
   - "REPHRASE" — user wants the same meaning said in different words
   - "REPEAT" — user wants to hear something again that was already said
   - "STOP" — user wants to stop or pause reading
   - "READ_HEADLINES" — user wants to hear news headlines or titles
   - "READ_ARTICLE" — user wants to read/hear the full article
   - "NAVIGATE_SECTION" — user wants to go to a specific section (sports, politics, business, etc.)
   - "NAVIGATE_PAGE_NUMBER" — user wants to go to a specific page number
   - "GO_TO_START" — user wants to go to the very beginning/first page
   - "GO_TO_END" — user wants to go to the very end/last page
   - "IDENTIFY_CONTENT" — user wants to know what content is on the current page

   IMPORTANT: You MUST pick exactly one intent from the list above. Do NOT create new intents.

DISAMBIGUATION RULES (follow these strictly):
- If the user wants SIMPLER/EASIER language, uses words like "simple", "easy", "not complicated" → SIMPLIFY (NOT EXPLAIN)
- If the user wants the SAME thing said in DIFFERENT WORDS, uses "different words", "another way", "rephrase" → REPHRASE (NOT EXPLAIN)
- If the user wants to UNDERSTAND or KNOW something better, wants clarification → EXPLAIN
- If the user wants to hear something AGAIN that was already said, didn't hear, wants it repeated → REPEAT (NOT EXPLAIN, NOT GO_TO_START)
- "Read from beginning again" or "say it again" = REPEAT
- If "next" refers to an article/news/story/headline → NEXT_ARTICLE
- If "next" refers to a page → NEXT_PAGE
- If "previous/back/before" refers to an article/news/story → PREVIOUS_ARTICLE
- If "previous/back" refers to a page → PREVIOUS_PAGE
- If user wants a SHORT VERSION or key points of content → SUMMARIZE
- If user wants to go to a NAMED section (sports, politics, business, entertainment) → NAVIGATE_SECTION

2. "personalization_flags": A JSON object extracting HOW the user wants it done.
   Look for mentions of:
   - "speed": "fast" or "slow"
   - "detail_level": "brief" or "detailed"
   - "language_style": "simple" or "technical"
   If none are mentioned, return empty object {}.

Return ONLY valid JSON. No explanation. No markdown. No extra text.

EXAMPLES:
Command: "Say it in simpler words"
{"intent": "SIMPLIFY", "personalization_flags": {"language_style": "simple"}}

Command: "Explain this to me"
{"intent": "EXPLAIN", "personalization_flags": {}}

Command: "Say that in different words"
{"intent": "REPHRASE", "personalization_flags": {}}

Command: "I didn't hear that, say it again"
{"intent": "REPEAT", "personalization_flags": {}}

Command: "Read the next article"
{"intent": "NEXT_ARTICLE", "personalization_flags": {}}

Command: "Go to the next page"
{"intent": "NEXT_PAGE", "personalization_flags": {}}

Command: "Go back to the previous news"
{"intent": "PREVIOUS_ARTICLE", "personalization_flags": {}}

Command: "Give me a summary"
{"intent": "SUMMARIZE", "personalization_flags": {}}

Command: "It's too complicated, make it simple"
{"intent": "SIMPLIFY", "personalization_flags": {"language_style": "simple"}}

Command: "What did you say? Tell me again"
{"intent": "REPEAT", "personalization_flags": {}}

Command: "Can you put that another way?"
{"intent": "REPHRASE", "personalization_flags": {}}

Command: "Go to the sports section"
{"intent": "NAVIGATE_SECTION", "personalization_flags": {}}"""

    user_prompt = f"User command: '{english_text}'"

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2:1b",
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


# ─── Run All Test Samples ─────────────────────────────
if __name__ == "__main__":
    import ast
    import csv

    print("=" * 60)
    print("APPROACH 1 — NLLB + Llama3.2:1b (Generated Samples)")
    print("=" * 60)

    # Load generate_samples.py which is structured as a python list
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'generate_samples.py')
    with open(data_path, 'r', encoding='utf-8') as f:
        generate_samples = ast.literal_eval(f.read())

    correct_count = 0
    total_time_all = 0

    # Open CSV for writing results as they come in
    csv_file = open("approach1_generated_results.csv", "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ID", "Input", "Expected", "English", "Intent", "Correct", "Flags", "Trans Time", "LLM Time", "Total Time"])

    for idx, sample in enumerate(generate_samples, start=1):
        sample_id = f"gen_{idx}"
        print(f"\nID       : {sample_id}")
        print(f"Input    : {sample['stt_output']}")
        print(f"Expected : {sample['expected_intent']}")

        try:
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
            
            # Save to CSV
            csv_writer.writerow([
                sample_id,
                sample["stt_output"],
                sample["expected_intent"],
                result["english_translation"],
                result["intent"],
                correct,
                json.dumps(result["personalization_flags"], ensure_ascii=False),
                result["translation_time_sec"],
                result["llm_time_sec"],
                result["total_time_sec"]
            ])
            csv_file.flush() # Ensure it writes immediately
        except Exception as e:
            print(f"❌ Error   : {e}")
            csv_writer.writerow([sample_id, sample["stt_output"], sample["expected_intent"], "ERROR", "ERROR", False, "{}", 0, 0, 0])
            csv_file.flush()
            
        print("-" * 60)

    csv_file.close()

    accuracy = (correct_count / len(generate_samples)) * 100 if generate_samples else 0
    avg_time = total_time_all / len(generate_samples) if generate_samples else 0

    print(f"\n{'=' * 60}")
    print(f"APPROACH 1 FINAL RESULTS (Generated Samples)")
    print(f"Correct  : {correct_count}/{len(generate_samples)}")
    print(f"Accuracy : {accuracy:.1f}%")
    print(f"Avg Time : {avg_time:.3f}s")
    print(f"{'=' * 60}")
