# Phase 2b experiment (ADAPTIVE_VOICE_PLAN.md): Option A vs Option B.
#
#   original - SYSTEM_PROMPT_ORIGINAL (the thesis benchmark prompt; called
#             "current" in the first run, before production switched to A)
#             + modifiers from intent_detection/modifiers.py
#   A       - SYSTEM_PROMPT_OPTION_A (CHANGE_SPEED known) + modifiers from rules
#   B       - SYSTEM_PROMPT_OPTION_B: the LLM returns intent AND modifiers
#
# Each sentence is translated by NLLB ONCE; all three prompts get the same
# English. Intent accuracy is reported without and with the fusion rule
# (modifiers.fuse_intent: a Sinhala repeat word turns PREVIOUS / NEXT /
# UNKNOWN into REPEAT). The three sets are reported separately:
#   45     data/test_samples.py          (+ EXISTING_MODIFIER_LABELS)
#   multi  data/test_samples_multi.py    (16)
#   fresh  data/test_samples_fresh.py    (17, held-out)
#
# Run from voice_interaction/:
#   venv\Scripts\python.exe -m intent_detection.evaluate_phase2b [output.json]
# Default output: phase2b_results_after_fix.json. The first-run file
# (phase2b_results.json) is the held-out result: the script never overwrites
# an existing results file.
#
# Timing run (latency of the production prompt only):
#   venv\Scripts\python.exe -m intent_detection.evaluate_phase2b --timing
# Option A alone on the 45 commands, one prompt for the whole run, so
# llama.cpp's prompt-prefix cache works as it does in production. One warm-up
# call (not a test sentence) first, so model loading is not counted.
# Output: phase2b_timing_option_a.json (never overwritten).

import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.test_samples import test_samples
from data.test_samples_multi import EXISTING_MODIFIER_LABELS, NONE, multi_samples
from data.test_samples_fresh import fresh_samples
from intent_detection.approach1_nllb_llm import (
    translate_to_english, normalize_intent, parse_option_b,
    SYSTEM_PROMPT_ORIGINAL, SYSTEM_PROMPT_OPTION_A, SYSTEM_PROMPT_OPTION_B,
    KNOWN_INTENTS, KNOWN_INTENTS_OPTION_A,
)
from intent_detection.llm_backend import chat, parse_json_reply, labels_match, get_active_backend
from intent_detection.modifiers import build_frame, fuse_intent

FIELDS = ("repeat", "speed", "style")
MARK = {"repeat": "R", "speed": "P", "style": "S"}   # wrong-field marks in the per-sample lines
VARIANTS = [
    ("original", SYSTEM_PROMPT_ORIGINAL, KNOWN_INTENTS),
    ("A", SYSTEM_PROMPT_OPTION_A, KNOWN_INTENTS_OPTION_A),
    ("B", SYSTEM_PROMPT_OPTION_B, KNOWN_INTENTS_OPTION_A),
]
SETS = [
    ("45", [{**s, "expected_modifiers": EXISTING_MODIFIER_LABELS.get(s["id"], NONE)}
            for s in test_samples]),
    ("multi", multi_samples),
    ("fresh", fresh_samples),
]


def run_variant(name, prompt, known, sinhala, english):
    t = time.time()
    raw = chat(prompt, f"User command: '{english}'")
    llm_time = time.time() - t

    llm_intent = normalize_intent(raw, known)
    if name == "B":
        modifiers, valid = parse_option_b(raw)
        fused, note = fuse_intent(sinhala, llm_intent)
        frame = {"intent": fused, "modifiers": modifiers,
                 "evidence": {"llm_intent": llm_intent, "fusion": note}}
    else:
        parsed = parse_json_reply(raw)
        valid = isinstance(parsed, dict) and "intent" in parsed
        frame = build_frame(sinhala, english, llm_intent)
    return {"raw": raw, "llm_intent": llm_intent, "frame": frame,
            "valid_json": valid, "llm_time_sec": round(llm_time, 3)}


def score(rows, variant):
    n = len(rows)
    out = {"n": n}
    out["intent_no_fusion"] = sum(r[variant]["intent_ok_no_fusion"] for r in rows)
    out["intent_fusion"] = sum(r[variant]["intent_ok_fusion"] for r in rows)
    for f in FIELDS:
        out[f] = sum(r[variant]["mod_ok"][f] for r in rows)
    out["all_modifiers"] = sum(all(r[variant]["mod_ok"].values()) for r in rows)
    out["invalid_json"] = sum(not r[variant]["valid_json"] for r in rows)
    out["mean_llm_sec"] = round(sum(r[variant]["llm_time_sec"] for r in rows) / n, 3)
    out["mean_total_sec"] = round(sum(r[variant]["llm_time_sec"] + r["translation_time_sec"]
                                      for r in rows) / n, 3)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_name = args[0] if args else "phase2b_results_after_fix.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
    if os.path.exists(out_path):
        sys.exit(f"{out_path} already exists — not overwriting earlier results. Give a new file name.")
    print(f"LLM backend: {get_active_backend()}")
    results, summary = {}, {}

    for set_name, samples in SETS:
        print(f"\n{'=' * 100}\n  SET: {set_name} ({len(samples)})\n{'=' * 100}")
        rows = []
        for s in samples:
            si, exp, exp_mod = s["stt_output"], s["expected_intent"], s["expected_modifiers"]
            t = time.time()
            en = translate_to_english(si)
            row = {"id": s["id"], "sinhala": si, "english": en, "expected_intent": exp,
                   "expected_modifiers": exp_mod, "translation_time_sec": round(time.time() - t, 3)}
            for name, prompt, known in VARIANTS:
                r = run_variant(name, prompt, known, si, en)
                r["intent_ok_no_fusion"] = labels_match(r["llm_intent"], exp)
                r["intent_ok_fusion"] = labels_match(r["frame"]["intent"], exp)
                r["mod_ok"] = {f: r["frame"]["modifiers"][f] == exp_mod[f] for f in FIELDS}
                row[name] = r
            rows.append(row)

            def cell(name):
                r = row[name]
                fused = r["frame"]["intent"]
                txt = r["llm_intent"] + ("" if fused == r["llm_intent"] else f"->{fused}")
                mark = "✅" if r["intent_ok_fusion"] else "❌"
                mods = "".join("·" if ok else MARK[f] for f, ok in r["mod_ok"].items())
                return f"{txt[:24]:<24}{mark} {mods}"
            print(f"{s['id']:<7} exp {exp:<16} | orig {cell('original')} | A {cell('A')} | B {cell('B')}")
            print(f"        {en}")

        results[set_name] = rows
        summary[set_name] = {name: score(rows, name) for name, _, _ in VARIANTS}

    # ── Summary tables ──
    print(f"\n{'=' * 100}\n  SUMMARY (modifier marks above: · = right, R / P / S = repeat / speed (pace) / style wrong)\n{'=' * 100}")
    for set_name, _ in SETS:
        print(f"\nSet {set_name}")
        print(f"  {'variant':<8} {'intent':>9} {'intent+fusion':>14} {'repeat':>7} {'speed':>6} "
              f"{'style':>6} {'all 3':>6} {'bad JSON':>9} {'LLM s':>6} {'total s':>8}")
        for name, _, _ in VARIANTS:
            m = summary[set_name][name]
            n = m["n"]
            print(f"  {name:<8} {m['intent_no_fusion']:>4}/{n:<4} {m['intent_fusion']:>8}/{n:<5} "
                  f"{m['repeat']:>4}/{n:<2}{m['speed']:>3}/{n:<2}{m['style']:>3}/{n:<2}"
                  f"{m['all_modifiers']:>3}/{n:<2}{m['invalid_json']:>6}/{n:<2}"
                  f"{m['mean_llm_sec']:>7}{m['mean_total_sec']:>8}")

    print("\nFusion rule changes (intent before -> after):")
    for set_name, rows in results.items():
        for row in rows:
            for name, _, _ in VARIANTS:
                r = row[name]
                if r["frame"]["intent"] != r["llm_intent"]:
                    effect = ("fixed" if r["intent_ok_fusion"] and not r["intent_ok_no_fusion"]
                              else "broke" if r["intent_ok_no_fusion"] and not r["intent_ok_fusion"]
                              else "no effect")
                    print(f"  {set_name:<6} {row['id']:<7} {name:<8} {r['llm_intent']} -> "
                          f"{r['frame']['intent']}  (expected {row['expected_intent']}: {effect})")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"backend": get_active_backend(), "summary": summary, "detailed_results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}")


def timing_run():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2b_timing_option_a.json")
    if os.path.exists(out_path):
        sys.exit(f"{out_path} already exists — not overwriting earlier results.")
    print(f"LLM backend: {get_active_backend()}  |  Option A only, 45 commands")

    # Warm-up: loads the GGUF model and fills the prompt cache (not timed)
    chat(SYSTEM_PROMPT_OPTION_A, f"User command: '{translate_to_english('ආයුබෝවන්')}'")

    rows = []
    for s in test_samples:
        t = time.time()
        en = translate_to_english(s["stt_output"])
        t_tr = time.time() - t
        t = time.time()
        raw = chat(SYSTEM_PROMPT_OPTION_A, f"User command: '{en}'")
        t_llm = time.time() - t
        frame = build_frame(s["stt_output"], en, normalize_intent(raw, KNOWN_INTENTS_OPTION_A))
        ok = labels_match(frame["intent"], s["expected_intent"])
        rows.append({"id": s["id"], "english": en, "intent": frame["intent"],
                     "expected_intent": s["expected_intent"], "intent_ok_fusion": ok,
                     "translation_time_sec": round(t_tr, 3), "llm_time_sec": round(t_llm, 3)})
        print(f"{s['id']:<7} translate {t_tr:5.2f}s  LLM {t_llm:5.2f}s  {frame['intent']:<16}{'✅' if ok else '❌'}")

    n = len(rows)
    tr = sorted(r["translation_time_sec"] for r in rows)
    llm = sorted(r["llm_time_sec"] for r in rows)
    summary = {
        "n": n,
        "mean_translation_sec": round(sum(tr) / n, 3),
        "mean_llm_sec": round(sum(llm) / n, 3),
        "mean_total_sec": round((sum(tr) + sum(llm)) / n, 3),
        "median_translation_sec": tr[n // 2],
        "median_llm_sec": llm[n // 2],
        "max_total_sec": round(max(r["translation_time_sec"] + r["llm_time_sec"] for r in rows), 3),
        "intent_fusion": sum(r["intent_ok_fusion"] for r in rows),
    }
    print(f"\nMean translation {summary['mean_translation_sec']}s | mean LLM {summary['mean_llm_sec']}s | "
          f"mean total {summary['mean_total_sec']}s | intent+fusion {summary['intent_fusion']}/{n}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"backend": get_active_backend(), "summary": summary, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if "--timing" in sys.argv:
        timing_run()
    else:
        main()
