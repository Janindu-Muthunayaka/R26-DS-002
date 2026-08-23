# Evaluation Script — Compares both approaches on all 6 test samples
# Measures: accuracy, latency, personalization extraction

import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.test_samples import test_samples
from intent_detection.approach1_nllb_llm import detect_intent_approach1
from intent_detection.approach2_direct_llm import detect_intent_approach2


def run_evaluation():
    print("=" * 70)
    print("  INTENT DETECTION — COMPARATIVE EVALUATION")
    print("  Component 4 — Adaptive Conversational Personalization")
    print("  Student: Sathsara S.P.Y.B | IT22004468 | R26-DS-002")
    print("=" * 70)

    results_a1 = []
    results_a2 = []

    for sample in test_samples:
        print(f"\n{'─'*70}")
        print(f"Sample ID    : {sample['id']}")
        print(f"Description  : {sample['description']}")
        print(f"STT Input    : {sample['stt_output']}")
        print(f"Expected     : {sample['expected_intent']}")

        # ── Approach 1 ──────────────────────────────
        print(f"\n  [Approach 1 — NLLB + Llama3.2:1b]")
        try:
            r1 = detect_intent_approach1(sample["stt_output"])
            correct1 = r1["intent"] == sample["expected_intent"]
            print(f"  Translation  : {r1['english_translation']}")
            print(f"  Intent       : {r1['intent']} {'✅' if correct1 else '❌'}")
            print(f"  Flags        : {r1['personalization_flags']}")
            print(f"  Total Time   : {r1['total_time_sec']}s")
            results_a1.append({**r1, "correct": correct1,
                                "expected": sample["expected_intent"]})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results_a1.append({"correct": False, "total_time_sec": 0,
                                "intent": "ERROR"})

        # ── Approach 2 ──────────────────────────────
        print(f"\n  [Approach 2 — Direct Qwen2.5:1.5b]")
        try:
            r2 = detect_intent_approach2(sample["stt_output"])
            correct2 = r2["intent"] == sample["expected_intent"]
            print(f"  Intent       : {r2['intent']} {'✅' if correct2 else '❌'}")
            print(f"  Flags        : {r2['personalization_flags']}")
            print(f"  Total Time   : {r2['total_time_sec']}s")
            results_a2.append({**r2, "correct": correct2,
                                "expected": sample["expected_intent"]})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results_a2.append({"correct": False, "total_time_sec": 0,
                                "intent": "ERROR"})

    # ── Summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    acc1 = sum(1 for r in results_a1 if r["correct"]) / len(results_a1) * 100
    acc2 = sum(1 for r in results_a2 if r["correct"]) / len(results_a2) * 100

    avg_t1 = sum(r["total_time_sec"] for r in results_a1) / len(results_a1)
    avg_t2 = sum(r["total_time_sec"] for r in results_a2) / len(results_a2)

    print(f"\n  {'Metric':<30} {'Approach 1':>15} {'Approach 2':>15}")
    print(f"  {'─'*60}")
    print(f"  {'Model':<30} {'NLLB+Llama3.2:1b':>15} {'Qwen2.5:1.5b':>15}")
    print(f"  {'Accuracy':<30} {acc1:>14.1f}% {acc2:>14.1f}%")
    print(f"  {'Avg Latency (sec)':<30} {avg_t1:>15.3f} {avg_t2:>15.3f}")
    print(f"  {'Translation Step':<30} {'Yes':>15} {'No':>15}")

    winner = "Approach 1" if acc1 > acc2 else \
             "Approach 2" if acc2 > acc1 else "Tie"
    faster = "Approach 1" if avg_t1 < avg_t2 else "Approach 2"

    print(f"\n  🏆 Best Accuracy : {winner}")
    print(f"  ⚡ Fastest       : {faster}")

    # Save results
    final = {
        "approach1_accuracy":     round(acc1, 2),
        "approach2_accuracy":     round(acc2, 2),
        "approach1_avg_latency":  round(avg_t1, 3),
        "approach2_avg_latency":  round(avg_t2, 3),
        "detailed_results_a1":    results_a1,
        "detailed_results_a2":    results_a2
    }

    with open("intent_detection_results.json", "w",
              encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Results saved to intent_detection_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()