# eval_perplexity.py
from dotenv import load_dotenv
load_dotenv()

import json
import statistics
from datetime import datetime
from collections import defaultdict

from tabulate import tabulate

from vectorstore import build_vectorstore
from pipeline import run_pipeline

FIXED_QUERY = "Can you explain this a little more?"

STYLE_VARIANTS = [
    {
        "style_class": "Detailed",
        "prompt_modifier": "Provide a thorough, comprehensive explanation with full context and detail.",
        "personalization_flags": {"language_style": "simple"},
    },
    {
        "style_class": "Simple",
        "prompt_modifier": "Use very simple everyday words, short sentences, avoid technical terms.",
        "personalization_flags": {"language_style": "simple"},
    },
    {
        "style_class": "StepByStep",
        "prompt_modifier": "Break the explanation into numbered steps.",
        "personalization_flags": {"detail_level": "structured"},
    },
]

def load_query_set(path="queries_corpus.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_voice_input(variant):
    return {
        "route": "GENERATE",
        "intent": "EXPLAIN",
        "english_translation": FIXED_QUERY,
        "style_class": variant["style_class"],
        "prompt_modifier": variant["prompt_modifier"],
        "personalization_flags": variant["personalization_flags"],
        "retrieved_chunk_id": None,
        "correction_applied": None,
    }

def main():
    vs = build_vectorstore()
    query_set = load_query_set()  # full 25 corpus entries — no slice

    raw_runs = []
    total = len(query_set) * len(STYLE_VARIANTS)
    done = 0
    raw_runs_path = "perplexity_raw_runs.json"

    for item in query_set:
        ocr_input = {"corrected_text": item["corrected_text"]}

        for variant in STYLE_VARIANTS:
            voice_input = build_voice_input(variant)
            try:
                result = run_pipeline(
                    vs, ocr_input, voice_input,
                    raw_request_context={"age": 25},
                )
                raw_runs.append({
                    "corpus_id": item["id"],
                    "style_class": variant["style_class"],
                    "perplexity": result["perplexity"],
                    "word_count": len(result["answer_si"].split()),
                    "answer_full": result["answer_si"],
                })
            except Exception as e:
                print(f"  [ERROR] corpus #{item['id']} - {variant['style_class']}: {e}")
                raw_runs.append({
                    "corpus_id": item["id"],
                    "style_class": variant["style_class"],
                    "perplexity": None,
                    "word_count": 0,
                    "answer_full": None,
                    "error": str(e),
                })

            done += 1
            print(f"  [{done}/{total}] corpus #{item['id']} - {variant['style_class']} done")

            # Save progress after every single run so a crash never loses work
            with open(raw_runs_path, "w", encoding="utf-8") as f:
                json.dump({"raw_runs": raw_runs}, f, ensure_ascii=False, indent=2)

    # --- aggregate per style: mean + std dev ---
    by_style = defaultdict(list)
    for r in raw_runs:
        if r["perplexity"] is not None:
            by_style[r["style_class"]].append(r["perplexity"])

    summary_rows = []
    for style in [v["style_class"] for v in STYLE_VARIANTS]:
        values = by_style[style]
        n = len(values)
        mean = statistics.mean(values) if n > 0 else float("nan")
        std = statistics.stdev(values) if n > 1 else 0.0
        summary_rows.append([style, n, f"{mean:.4f}", f"{std:.4f}"])

    summary_table = tabulate(
        summary_rows,
        headers=["Style", "N", "Mean Perplexity", "Std Dev"],
        tablefmt="grid",
    )

    print("\n" + summary_table)

    # --- write summary to file ---
    timestamp = datetime.now().isoformat()

    with open("perplexity_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Fixed query: {FIXED_QUERY}\n")
        f.write(f"Number of corpus texts tested: {len(query_set)}\n")
        f.write(f"Total generations: {total} ({len(query_set)} texts x {len(STYLE_VARIANTS)} styles)\n\n")
        f.write(summary_table)
        f.write("\n")

    print(f"\nSummary saved to: perplexity_summary.txt")
    print(f"Full raw data saved to: {raw_runs_path}")

if __name__ == "__main__":
    main()