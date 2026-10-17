"""
Evaluation runner for the Sinhala Reading Assistant RAG (Component 3).

Run from the project root (same folder as pipeline.py), with OPENAI_API_KEY set:
    python eval_rag.py --data eval_set.jsonl --out eval_out

Outputs:
    eval_out/per_question.csv   one row per question (read this first when debugging)
    eval_out/summary.json       aggregate numbers for the thesis tables
"""
import argparse
import copy
import csv
import json
import os
import statistics
import unicodedata

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from vectorstore import build_vectorstore
from pipeline import run_pipeline
from adapters import parse_voice_input, parse_request_context
from prompt import resolve_age_profile, resolve_max_words
from generate import _sinhala_purity
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT

# ----------------------------------------------------------------------------
# CONFIG - adjust to match contracts.py if your intent names / modifiers differ
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "corpus", "articles_fixed.jsonl")

# The contract's query field is "english_translation" (Component 4 sends the translated question).
QUERY_FIELD = "english_translation"

# eval_set labels -> exact strings used in contracts.py / retrieve.py
INTENT_MAP = {
    "direct_answer": "DIRECT_ANSWER",   # <- CHECK this exact spelling in adapters.py / retrieve.py
    "summarize": "SUMMARIZE",
    "simplify": "SIMPLIFY",
    "elaborate": "ELABORATE",
}
STYLE_MAP = {"simple": "Simple", "moderate": "Moderate", "detailed": "Detailed"}

# Prompt modifiers follow style_class in your contract. "Simple" is copied from contracts.py;
# replace the others with the exact text from sample_personalization_outputs.json.
MODIFIER_BY_STYLE = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Moderate": "Use clear, balanced language with the key details.",
    "Detailed": "Give a thorough, well-organised explanation with all relevant details.",
}

REFUSAL_PHRASE = "ප්‍රමාණවත් තොරතුරු නොමැත"
EVAL_AGE = 25  # -> "18+" band; change to test other age bands


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def norm(s: str) -> str:
    """Remove invisible Sinhala joiners so string matching is reliable.
    The corpus `clean_body` has no ZWJ, but typed questions/answers usually do."""
    s = unicodedata.normalize("NFC", s or "")
    for ch in ("\u200d", "\u200c", "\u200b", "\u00ad"):
        s = s.replace(ch, "")
    return " ".join(s.split())


def set_key(d, key, value) -> bool:
    """Overwrite `key` wherever it appears in a nested dict/list."""
    found = False
    if isinstance(d, dict):
        for k in list(d):
            if k == key:
                d[k] = value
                found = True
            elif isinstance(d[k], (dict, list)):
                found = set_key(d[k], key, value) or found
    elif isinstance(d, list):
        for x in d:
            found = set_key(x, key, value) or found
    return found


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def build_voice_input(row: dict, query: str) -> dict:
    # copy everything except correction_applied (contracts.py uses sqlalchemy's `null` there)
    v = {k: copy.deepcopy(val) for k, val in SAMPLE_VOICE_INPUT.items() if k != "correction_applied"}
    v["correction_applied"] = None
    style = STYLE_MAP[row["style_class"]]
    for key, val in [
        (QUERY_FIELD, query),
        ("intent", INTENT_MAP[row["intent"]]),
        ("style_class", style),
        ("prompt_modifier", MODIFIER_BY_STYLE[style]),
        ("retrieved_chunk_id", None),  # IMPORTANT: no anchor -> tests real retrieval
    ]:
        if not set_key(v, key, val):
            raise RuntimeError(f"key '{key}' not found in SAMPLE_VOICE_INPUT - check contracts.py")
    v.setdefault("personalization_flags", {})["detail_level"] = row["detail_level"]
    return v


def check_corpus(rows):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        ids = {json.loads(line)["article_id"] for line in f if line.strip()}
    needed = {r["article_id"] for r in rows if r.get("article_id")}
    needed |= {r["related_article_id"] for r in rows if r.get("related_article_id")}
    missing = needed - ids
    if missing:
        raise SystemExit(f"These article_ids are NOT in {CORPUS_PATH}: {sorted(missing)}")
    print(f"[ok] all {len(needed)} gold article_ids exist in the knowledge base ({len(ids)} articles)")


JUDGE_TEMPLATE = """You are a strict evaluator of a Sinhala question-answering system.
You get: a question, the retrieved CONTEXT, a REFERENCE answer, and the system ANSWER (all in Sinhala).

Score the ANSWER:
- "faithfulness": 1 if EVERY factual claim in the ANSWER is supported by the CONTEXT, else 0.
  (Ignore style. Unsupported names, numbers or dates -> 0.)
- "correctness": 2 = fully agrees with the REFERENCE on the key facts,
                 1 = partly correct or incomplete,
                 0 = wrong, off-topic or missing the key fact.

Return ONLY a JSON object: {{"faithfulness": 0 or 1, "correctness": 0, 1 or 2, "reason": "<one short English sentence>"}}

QUESTION:
{question}

CONTEXT:
{context}

REFERENCE:
{reference}

ANSWER:
{answer}
"""


def judge(llm, question, contexts, reference, answer):
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        context="\n---\n".join(contexts),
        reference=reference,
        answer=answer,
    )
    raw = llm.invoke(prompt).content
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


# ----------------------------------------------------------------------------
# one question
# ----------------------------------------------------------------------------
def evaluate_row(vs, row, judge_llm, query_lang="en"):
    answerable = row["type"] == "answerable"
    query = row["question_en"] if query_lang == "en" else row["question"]
    voice_raw = build_voice_input(row, query)
    ctx_raw = {"age": EVAL_AGE}

    # sanity check: confirm the adapter really received our question
    voice = parse_voice_input(copy.deepcopy(voice_raw))
    if query not in json.dumps(voice, ensure_ascii=False, default=str):
        raise RuntimeError("parse_voice_input dropped/changed the query - check QUERY_FIELD and adapters.py")

    # word cap exactly as pipeline.py computes it
    ctx = parse_request_context(ctx_raw)
    age_profile = resolve_age_profile(ctx["age_group"] or voice["personalization_flags"].get("age_group"))
    max_words = resolve_max_words(voice["style_class"], voice["personalization_flags"], age_profile["word_cap"])

    res = run_pipeline(vs, copy.deepcopy(SAMPLE_OCR_INPUT), voice_raw, ctx_raw, skip_ocr=True)
    answer = res["answer_si"] or ""
    contexts = res["contexts"]
    sources = res["retrieved_sources"]

    # ---- retrieval: rank of gold article / of a chunk that contains the key fact
    art_rank = ev_rank = None
    pos = 0
    for src, text in zip(sources, contexts):
        if src.get("source_type", "article") != "article":
            continue  # ignore OCR / non-corpus chunks
        pos += 1
        if answerable and src.get("article_id") == row["article_id"]:
            art_rank = art_rank or pos
            if ev_rank is None and any(norm(k) in norm(text) for k in row["key_facts"]):
                ev_rank = pos

    refused = norm(REFUSAL_PHRASE) in norm(answer)
    out = {
        "qid": row["qid"], "type": row["type"], "intent": row["intent"], "query_lang": query_lang,
        "detail_level": row["detail_level"], "article_id": row.get("article_id"),
        "question": query, "answer": answer, "refused": refused,
        "n_retrieved": pos, "article_rank": art_rank, "evidence_rank": ev_rank,
        "key_fact_in_answer": (
            any(norm(k) in norm(answer) for k in row["key_facts"]) if answerable else None
        ),
        "purity": round(_sinhala_purity(answer), 3),
        "max_words": max_words, "n_words": len(answer.split()),
        "within_word_limit": len(answer.split()) <= max_words,
        "perplexity": res["perplexity"],
        "faithfulness": None, "correctness": None, "judge_reason": None, "error": None,
    }

    if answerable and judge_llm is not None:
        if refused:
            out["correctness"] = 0  # false refusal; faithfulness not applicable
        else:
            try:
                j = judge(judge_llm, row["question"], contexts, row["reference_answer"], answer)
                out["faithfulness"] = j.get("faithfulness")
                out["correctness"] = j.get("correctness")
                out["judge_reason"] = j.get("reason")
            except Exception as e:  # judge failure must not kill the run
                out["judge_reason"] = f"judge_error: {e!r}"
    return out


# ----------------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------------
def summarize(results):
    ok = [r for r in results if not r.get("error")]
    ans = [r for r in ok if r["type"] == "answerable"]
    una = [r for r in ok if r["type"].startswith("unanswerable")]
    gen = [r for r in ans if not r["refused"]]  # answered answerable questions

    def hit(field, k):
        return mean([1.0 if r[field] and r[field] <= k else 0.0 for r in ans])

    def mrr(field):
        return mean([1.0 / r[field] if r[field] else 0.0 for r in ans])

    s = {
        "n_total": len(results), "n_errors": len(results) - len(ok),
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "retrieval": {
            **{f"article_hit@{k}": hit("article_rank", k) for k in (1, 3, 5)},
            "article_hit@any": hit("article_rank", 10**6),
            **{f"evidence_hit@{k}": hit("evidence_rank", k) for k in (1, 3, 5)},
            "evidence_hit@any": hit("evidence_rank", 10**6),
            "article_MRR": mrr("article_rank"),
            "evidence_MRR": mrr("evidence_rank"),
        },
        "generation": {
            "key_fact_in_answer_rate": mean([1.0 if r["key_fact_in_answer"] else 0.0 for r in ans]),
            "correctness_mean_0to2": mean([r["correctness"] for r in ans]),
            "faithfulness_rate": mean([r["faithfulness"] for r in gen]),
            "false_refusal_rate": mean([1.0 if r["refused"] else 0.0 for r in ans]),
            "correct_abstention_rate": mean([1.0 if r["refused"] else 0.0 for r in una]),
            "correct_abstention_far": mean([1.0 if r["refused"] else 0.0 for r in una if r["type"] == "unanswerable_far"]),
            "correct_abstention_near": mean([1.0 if r["refused"] else 0.0 for r in una if r["type"] == "unanswerable_near"]),
        },
        "format": {
            "sinhala_purity_mean": mean([r["purity"] for r in ok]),
            "purity_ge_0.85_rate": mean([1.0 if r["purity"] >= 0.85 else 0.0 for r in ok]),
            "within_word_limit_rate": mean([1.0 if r["within_word_limit"] else 0.0 for r in ok]),
        },
        # perplexity only on real answers (refusal sentence is a fixed phrase -> artificially low)
        "perplexity_answered_questions": {
            "mean": mean([r["perplexity"] for r in gen]),
            "median": median([r["perplexity"] for r in gen]),
            "by_intent": {
                i: mean([r["perplexity"] for r in gen if r["intent"] == i])
                for i in sorted({r["intent"] for r in gen})
            },
            "correct_answers_only": mean([r["perplexity"] for r in gen if r["correctness"] == 2]),
            "wrong_or_partial_only": mean([r["perplexity"] for r in gen if r["correctness"] in (0, 1)]),
        },
        "by_intent_correctness": {
            i: mean([r["correctness"] for r in ans if r["intent"] == i])
            for i in sorted({r["intent"] for r in ans})
        },
    }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval_set.jsonl")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5.4-mini"))
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judge (retrieval + rule-based metrics only)")
    ap.add_argument("--query-lang", choices=["en", "si"], default="en",
                    help="en = English translation (real pipeline input); si = original Sinhala question")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    check_corpus(rows)
    vs = build_vectorstore()
    judge_llm = None if args.no_judge else ChatOpenAI(model=args.judge_model, temperature=0)

    results = []
    for i, row in enumerate(rows, 1):
        try:
            r = evaluate_row(vs, row, judge_llm, args.query_lang)
        except Exception as e:
            r = {"qid": row["qid"], "type": row["type"], "intent": row["intent"], "error": repr(e)}
        results.append(r)
        print(f"[{i}/{len(rows)}] {row['qid']} art_rank={r.get('article_rank')} "
              f"ev_rank={r.get('evidence_rank')} refused={r.get('refused')} err={r.get('error')}")

    fields = sorted({k for r in results for k in r}, key=lambda k: (k != "qid", k))
    with open(os.path.join(args.out, "per_question.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    summary = summarize(results)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()