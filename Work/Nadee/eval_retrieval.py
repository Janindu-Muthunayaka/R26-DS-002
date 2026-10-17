# eval_retrieval.py
# Retrieval-only evaluation (RQ1): compares dense-only, BM25-only and hybrid
# on the answerable queries of your eval set, at article level and evidence level.
#
#   python eval_retrieval.py                       # uses eval/eval_set.jsonl
#   python eval_retrieval.py path/to/eval.jsonl
#   python eval_retrieval.py --si gold             # use the gold Sinhala `question`
#                                                  #   as the BM25 query (upper bound)
#   python eval_retrieval.py --si llm              # (default) Sinhala query produced
#                                                  #   by the LLM from question_en,
#                                                  #   exactly like the real pipeline
#
# Each query is run WITHOUT a pre-selected passage and without the OCR chunk,
# over the whole corpus - same protocol as thesis Section 6.
from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import os
from collections import OrderedDict

from vectorstore import build_vectorstore
from hybrid import hybrid_search
from bm25_index import normalize
from query_si import get_sinhala_query

KS = (1, 3, 5)
MAX_RANK = 10          # how deep each ranking is read (Hit@k and MRR@10)
OUT_DIR = "eval_results"


def match_norm(text: str) -> str:
    """Same normalisation for key facts and passages (invisible joiners,
    case, whitespace), so 'ඊබ්රාහිම්' matches 'ඊබ්‍රාහිම්ට'."""
    return " ".join(normalize(text).split())


def load_queries(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if r.get("type") == "answerable"]


def first_rank(docs, predicate):
    for i, d in enumerate(docs, start=1):
        if predicate(d):
            return i
    return None


def run_all_configs(vs, q_en, q_si):
    return OrderedDict([
        # what the pipeline did before (English query, dense only)
        ("dense_en",   hybrid_search(vs, q_en, None, MAX_RANK, "dense")),
        ("dense_si",   hybrid_search(vs, q_si, None, MAX_RANK, "dense")),
        ("bm25_word",  hybrid_search(vs, None, q_si, MAX_RANK, "bm25", use_ngrams=False)),
        ("bm25_ngram", hybrid_search(vs, None, q_si, MAX_RANK, "bm25")),
        # what the pipeline does now (English dense + Sinhala BM25 + RRF)
        ("hybrid",     hybrid_search(vs, q_en, q_si, MAX_RANK, "hybrid")),
        ("hybrid_si",  hybrid_search(vs, q_si, q_si, MAX_RANK, "hybrid")),
    ])


def summarise(ranks):
    n = len(ranks)
    out = {}
    for k in KS:
        out[f"Hit@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    out["MRR"] = sum((1.0 / r) if r else 0.0 for r in ranks) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="eval/eval_set.jsonl")
    ap.add_argument("--si", choices=["llm", "gold"], default="llm")
    args = ap.parse_args()

    queries = load_queries(args.path)
    print(f"{len(queries)} answerable queries from {args.path}  (Sinhala query source: {args.si})")

    vs = build_vectorstore()

    per_query = []
    art_ranks = {}
    evi_ranks = {}
    missing_si = 0

    for q in queries:
        q_en = q["question_en"]
        if args.si == "gold":
            q_si = q["question"]
        else:
            q_si = get_sinhala_query(q_en) or ""
            if not q_si:
                missing_si += 1

        gold_article = str(q["article_id"])
        key_facts = [match_norm(k) for k in q.get("key_facts", [])]

        def is_article(d):
            return d.metadata.get("article_id") == gold_article

        def is_evidence(d):
            if not is_article(d):
                return False
            text = match_norm(d.page_content)
            return any(kf and kf in text for kf in key_facts)

        results = run_all_configs(vs, q_en, q_si)
        row = {"qid": q["qid"], "q_si_used": q_si}
        for name, docs in results.items():
            a = first_rank(docs, is_article)
            e = first_rank(docs, is_evidence)
            art_ranks.setdefault(name, []).append(a)
            evi_ranks.setdefault(name, []).append(e)
            row[f"{name}_article_rank"] = a
            row[f"{name}_evidence_rank"] = e
        per_query.append(row)

    if missing_si:
        print(f"WARNING: no Sinhala query for {missing_si} queries (translation failed).")

    summary = {"article": {}, "evidence": {}}
    for level, store in (("article", art_ranks), ("evidence", evi_ranks)):
        print(f"\n=== {level.upper()} level ===")
        header = f"{'config':<12}" + "".join(f"{('Hit@' + str(k)):>9}" for k in KS) + f"{'MRR@10':>9}"
        print(header)
        for name, ranks in store.items():
            s = summarise(ranks)
            summary[level][name] = s
            print(f"{name:<12}" + "".join(f"{s[f'Hit@{k}']*100:>8.1f}%" for k in KS) + f"{s['MRR']:>9.3f}")

    # Paired comparison: hybrid vs the old dense-only behaviour (evidence rank)
    better = worse = tie = 0
    for h, d in zip(evi_ranks["hybrid"], evi_ranks["dense_en"]):
        hv = h if h is not None else 10**6
        dv = d if d is not None else 10**6
        if hv < dv:
            better += 1
        elif hv > dv:
            worse += 1
        else:
            tie += 1
    print(f"\nhybrid vs dense_en (evidence rank): better={better}  worse={worse}  same={tie}")
    print(f"(one query = {100 / len(queries):.1f} percentage points - small differences are noise)")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "retrieval_per_query.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
        w.writeheader()
        w.writerows(per_query)
    with open(os.path.join(OUT_DIR, "retrieval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"si_source": args.si, "n_queries": len(queries), "summary": summary,
                   "hybrid_vs_dense_evidence": {"better": better, "worse": worse, "same": tie}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_DIR}/retrieval_summary.json and {OUT_DIR}/retrieval_per_query.csv")


if __name__ == "__main__":
    main()
