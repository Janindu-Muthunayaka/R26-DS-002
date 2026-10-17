# test_history.py  — run with:  python test_history.py
from dotenv import load_dotenv
load_dotenv()

import copy
import history
from history import add_turn, get_history, clear_history
from rewrite import rewrite_query
from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT


def test_a_store():
    print("\n=== A: history store (no API calls) ===")
    clear_history()
    for i in range(history.MAX_TURNS + 2):
        add_turn(f"q{i}", f"standalone q{i}", f"a{i}")
    turns = get_history()
    assert len(turns) == history.MAX_TURNS, "should keep only MAX_TURNS turns"
    assert turns[0]["question"] == "q2", "oldest turns should be dropped"
    clear_history()
    assert get_history() == [], "clear_history should empty the store"
    print("PASS")


def test_b_rewrite():
    print("\n=== B: rewrite_query (1 LLM call) ===")
    fake_history = [{
        "question": "What did Namal Rajapaksa say about the Easter attack?",
        "standalone_question": "What did Namal Rajapaksa say about the Easter attack?",
        "answer": "He said the Easter attack has been completely politicised.",
    }]
    out = rewrite_query("Who did he say was arrested?", fake_history)
    print("rewritten:", out)
    assert "Rajapaksa" in out or "Namal" in out, "rewrite should add the missing subject"

    same = rewrite_query("Who is the president of Sri Lanka?", fake_history)
    print("standalone question (should be unchanged):", same)

    assert rewrite_query("Who did he say was arrested?", []) == "Who did he say was arrested?"
    print("PASS (check the printed lines by eye)")


def _voice(question):
    v = copy.deepcopy(SAMPLE_VOICE_INPUT)
    v.update({
        "route": "GENERATE",
        "english_translation": question,
        "style_class": "Moderate",
        "prompt_modifier": "Answer clearly.",
        "personalization_flags": {"detail_level": "moderate"},
        "retrieved_chunk_id": None,   # no anchor, so we test real retrieval
        "correction_applied": None,
    })
    return v


def test_c_pipeline():
    print("\n=== C: full pipeline, 3 turns (uses OpenAI) ===")
    vs = build_vectorstore()
    clear_history()
    questions = [
        "What did Namal Rajapaksa say about the Easter Sunday attack?",
        "Who did he say was arrested?",           # follow-up: "he"
        "Why does he think it is political?",     # follow-up: "he", "it"
    ]
    for q in questions:
        res = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q),
                           {"age": 25}, skip_ocr=True, use_history=True)
        print(f"\nUSER      : {q}")
        print(f"STANDALONE: {res['standalone_query']}")
        print(f"SOURCES   : {[s.get('headline', '')[:40] for s in res['retrieved_sources']]}")
        print(f"ANSWER    : {res['answer_si'][:200]}")
    assert len(get_history()) == 3

    print("\n--- Same follow-up WITHOUT history (for comparison) ---")
    clear_history()
    res = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice("Who did he say was arrested?"),
                       {"age": 25}, skip_ocr=True, use_history=True)
    print(f"STANDALONE: {res['standalone_query']}   (should be unchanged)")
    print(f"ANSWER    : {res['answer_si'][:200]}")

    print("\n--- use_history=False must not touch the store (eval safety) ---")
    clear_history()
    run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(questions[0]),
                 {"age": 25}, skip_ocr=True)
    assert get_history() == [], "default call must not save history"
    print("PASS")


if __name__ == "__main__":
    test_a_store()
    test_b_rewrite()
    test_c_pipeline()