# Approach 3: Trained TF-IDF + LinearSVC classifier
# Flow: Sinhala STT → NLLB translates to English (reused from Approach 1)
#       → trained classifier predicts intent directly (no LLM call)
#
# This model was trained offline in Google Colab on 530 samples generated
# by Approach 1 (NLLB + Llama3.2:1b) — see intent_detection/model/README.md
# for details. It is a distillation of Approach 1's behavior into a much
# smaller, faster model.
#
# IMPORTANT LIMITATION: unlike Approach 1 and Approach 2, this classifier
# was only trained to predict the "intent" label. It does NOT extract
# personalization_flags (speed/detail_level/language_style) — that would
# need a separate classifier or a different training setup. Flags are
# always returned as an empty dict here.

import os
import time
import joblib

from intent_detection.approach1_nllb_llm import translate_to_english, extract_intent_llama

# ─── Load Trained Classifier ───────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
BUNDLE_PATH = os.path.join(MODEL_DIR, "intent_classifier_bundle.joblib")

print("Loading trained intent classifier (TF-IDF + LinearSVC)...")

_bundle = joblib.load(BUNDLE_PATH)
_classifier = _bundle["model"]
_vectorizer = _bundle["vectorizer"]

print("✅ Trained intent classifier loaded successfully!")

# Below this confidence, the hybrid pipeline falls back to Approach 1
# (Llama3.2:1b). Tune this after testing on real STT+NLLB output —
# 0.6 is a reasonable starting point based on Colab evaluation.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


# ─── Classify English Text ─────────────────────────────
def classify_intent_trained(english_text):
    """
    Runs the trained classifier on already-translated English text.
    Returns (intent, confidence).
    """
    vec = _vectorizer.transform([english_text])
    probs = _classifier.predict_proba(vec)[0]
    top_idx = probs.argmax()
    intent = _classifier.classes_[top_idx]
    confidence = float(probs[top_idx])
    return intent, confidence


# ─── Full Pipeline (trained model only, no fallback) ───
def detect_intent_approach3(sinhala_text):
    """
    Full Approach 3 pipeline:
    Sinhala → NLLB English → trained classifier
    """
    # Step 1 — Translate (same NLLB model Approach 1 already loads)
    t1 = time.time()
    english_text = translate_to_english(sinhala_text)
    translation_time = time.time() - t1

    # Step 2 — Classify
    t2 = time.time()
    intent, confidence = classify_intent_trained(english_text)
    classify_time = time.time() - t2

    total_time = translation_time + classify_time

    return {
        "approach":              "Approach 3 — Trained TF-IDF+LinearSVC",
        "sinhala_input":         sinhala_text,
        "english_translation":   english_text,
        "intent":                intent,
        "confidence":            round(confidence, 3),
        "personalization_flags": {},  # not extracted by this model — see module docstring
        "translation_time_sec":  round(translation_time, 3),
        "classify_time_sec":     round(classify_time, 3),
        "total_time_sec":        round(total_time, 3),
    }


# ─── Hybrid Pipeline (trained model + Llama fallback) ──
def detect_intent_hybrid(sinhala_text, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Fast path: trained classifier.
    If its confidence is below confidence_threshold, falls back to
    Approach 1's Llama3.2:1b call for a more reliable (but slower)
    answer — reusing the same English translation, so no re-translation.
    """
    fast_result = detect_intent_approach3(sinhala_text)

    if fast_result["confidence"] >= confidence_threshold:
        fast_result["route"] = "TRAINED_MODEL"
        fast_result["approach"] = "Hybrid — Trained model (high confidence)"
        return fast_result

    # Low confidence — fall back to Llama, reusing the translation we already have
    t1 = time.time()
    llm_result = extract_intent_llama(fast_result["english_translation"])
    llm_time = time.time() - t1

    total_time = (
        fast_result["translation_time_sec"]
        + fast_result["classify_time_sec"]
        + llm_time
    )

    return {
        "approach":                "Hybrid — Llama3.2:1b fallback (low confidence)",
        "sinhala_input":           sinhala_text,
        "english_translation":     fast_result["english_translation"],
        "intent":                  llm_result.get("intent", "UNKNOWN"),
        "personalization_flags":   llm_result.get("personalization_flags", {}),
        "trained_model_guess":     fast_result["intent"],
        "trained_model_confidence": fast_result["confidence"],
        "route":                   "LLM_FALLBACK",
        "translation_time_sec":    fast_result["translation_time_sec"],
        "llm_time_sec":            round(llm_time, 3),
        "total_time_sec":          round(total_time, 3),
    }


# ─── Run All Test Samples ───────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.test_samples import test_samples

    print("=" * 60)
    print("APPROACH 3 — Trained TF-IDF+LinearSVC")
    print("=" * 60)

    correct_count = 0
    total_time_all = 0

    for sample in test_samples:
        print(f"\nID       : {sample['id']}")
        print(f"Input    : {sample['stt_output']}")
        print(f"Expected : {sample['expected_intent']}")

        result = detect_intent_approach3(sample["stt_output"])
        correct = result["intent"] == sample["expected_intent"]

        if correct:
            correct_count += 1
        total_time_all += result["total_time_sec"]

        print(f"English  : {result['english_translation']}")
        print(f"Intent   : {result['intent']} {'✅' if correct else '❌'} (confidence: {result['confidence']})")
        print(f"Total    : {result['total_time_sec']}s")
        print("-" * 60)

    accuracy = (correct_count / len(test_samples)) * 100
    avg_time = total_time_all / len(test_samples)

    print(f"\n{'=' * 60}")
    print(f"APPROACH 3 FINAL RESULTS")
    print(f"Correct  : {correct_count}/{len(test_samples)}")
    print(f"Accuracy : {accuracy:.1f}%")
    print(f"Avg Time : {avg_time:.3f}s")
    print(f"{'=' * 60}")
