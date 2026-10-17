# Evaluation Script — Compares Approach 1, Approach 2, Approach 3, and the
# Hybrid (Approach 3 + Approach 1 fallback) on all test samples.
#
# This is separate from evaluate_both.py (which still compares only
# Approach 1 vs Approach 2, unchanged) so you can run either comparison
# independently.

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.test_samples import test_samples
from intent_detection.approach1_nllb_llm import detect_intent_approach1
from intent_detection.approach2_direct_llm import detect_intent_approach2
from intent_detection.approach3_trained_classifier import detect_intent_approach3, detect_intent_hybrid


def _matches(intent, expected):
    # test_samples.py has one label ("Simplify") in different case than
    # the trained classes ("SIMPLIFY") — compare case-insensitively so
    # accuracy numbers aren't skewed by that inconsistency.
    return str(intent).strip().upper() == str(expected).strip().upper()


def run_evaluation():
    print("=" * 70)
    print("  INTENT DETECTION — FULL COMPARATIVE EVALUATION")
    print("  Approach 1 (NLLB+Llama) vs Approach 2 (Qwen direct)")
    print("  vs Approach 3 (trained classifier) vs Hybrid")
    print("  Component 4 — Adaptive Conversational Personalization")
    print("=" * 70)

    results = {"a1": [], "a2": [], "a3": [], "hybrid": []}

    for sample in test_samples:
        print(f"\n{'─'*70}")
        print(f"Sample ID    : {sample['id']}")
        print(f"STT Input    : {sample['stt_output']}")
        print(f"Expected     : {sample['expected_intent']}")

        # ── Approach 1 ──────────────────────────────
        try:
            r1 = detect_intent_approach1(sample["stt_output"])
            c1 = _matches(r1["intent"], sample["expected_intent"])
            print(f"  [A1 NLLB+Llama]     {r1['intent']:<20} {'✅' if c1 else '❌'}  {r1['total_time_sec']}s")
            results["a1"].append({**r1, "correct": c1})
        except Exception as e:
            print(f"  [A1 NLLB+Llama]     ❌ Error: {e}")
            results["a1"].append({"correct": False, "total_time_sec": 0, "intent": "ERROR"})

        # ── Approach 2 ──────────────────────────────
        try:
            r2 = detect_intent_approach2(sample["stt_output"])
            c2 = _matches(r2["intent"], sample["expected_intent"])
            print(f"  [A2 Qwen direct]    {r2['intent']:<20} {'✅' if c2 else '❌'}  {r2['total_time_sec']}s")
            results["a2"].append({**r2, "correct": c2})
        except Exception as e:
            print(f"  [A2 Qwen direct]    ❌ Error: {e}")
            results["a2"].append({"correct": False, "total_time_sec": 0, "intent": "ERROR"})

        # ── Approach 3 (trained classifier only) ────
        try:
            r3 = detect_intent_approach3(sample["stt_output"])
            c3 = _matches(r3["intent"], sample["expected_intent"])
            print(f"  [A3 Trained model]  {r3['intent']:<20} {'✅' if c3 else '❌'}  {r3['total_time_sec']}s  (conf: {r3['confidence']})")
            results["a3"].append({**r3, "correct": c3})
        except Exception as e:
            print(f"  [A3 Trained model]  ❌ Error: {e}")
            results["a3"].append({"correct": False, "total_time_sec": 0, "intent": "ERROR"})

        # ── Hybrid (trained model + Llama fallback) ─
        try:
            rh = detect_intent_hybrid(sample["stt_output"])
            ch = _matches(rh["intent"], sample["expected_intent"])
            print(f"  [Hybrid]            {rh['intent']:<20} {'✅' if ch else '❌'}  {rh['total_time_sec']}s  (route: {rh['route']})")
            results["hybrid"].append({**rh, "correct": ch})
        except Exception as e:
            print(f"  [Hybrid]            ❌ Error: {e}")
            results["hybrid"].append({"correct": False, "total_time_sec": 0, "intent": "ERROR"})

    # ── Summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    summary = {}
    for key, label in [("a1", "Approach 1 (NLLB+Llama)"), ("a2", "Approach 2 (Qwen direct)"),
                        ("a3", "Approach 3 (trained)"), ("hybrid", "Hybrid")]:
        rows = results[key]
        acc = sum(1 for r in rows if r["correct"]) / len(rows) * 100
        avg_t = sum(r["total_time_sec"] for r in rows) / len(rows)
        summary[key] = {"label": label, "accuracy": round(acc, 2), "avg_latency": round(avg_t, 3)}
        print(f"  {label:<28} accuracy: {acc:>6.1f}%   avg latency: {avg_t:>7.3f}s")

    best = max(summary.items(), key=lambda kv: kv[1]["accuracy"])
    fastest = min(summary.items(), key=lambda kv: kv[1]["avg_latency"])
    print(f"\n  🏆 Best Accuracy : {best[1]['label']} ({best[1]['accuracy']}%)")
    print(f"  ⚡ Fastest       : {fastest[1]['label']} ({fastest[1]['avg_latency']}s)")

    with open("intent_detection_results_all.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": results}, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Results saved to intent_detection_results_all.json")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
