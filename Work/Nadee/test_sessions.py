# test_sessions.py  — run with:  python test_sessions.py
from dotenv import load_dotenv
load_dotenv()

import copy
import os

import history
from history import add_turn, get_history, clear_history, history_path, HISTORY_DIR
from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT

SA, SB = "test_a", "test_b"


def _cleanup():
    for sid in (SA, SB, "../../evil", None):
        clear_history(sid)


def test_a_store():
    print("\n=== A: per-session store + JSON files (no API calls) ===")
    _cleanup()

    # isolation
    add_turn("q-a", "standalone q-a", "පාස්කු ප්‍රහාරය ගැන පිළිතුර", SA)
    add_turn("q-b", "standalone q-b", "answer b", SB)
    assert [t["question"] for t in get_history(SA)] == ["q-a"]
    assert [t["question"] for t in get_history(SB)] == ["q-b"]

    # trimming
    for i in range(history.MAX_TURNS + 2):
        add_turn(f"q{i}", f"s{i}", f"a{i}", SA)
    turns = get_history(SA)
    assert len(turns) == history.MAX_TURNS
    assert turns[0]["question"] == "q2", "oldest turns should be dropped"

    # JSON file exists and Sinhala is stored as real characters (not \\uXXXX)
    add_turn("q", "s", "පාස්කු", SB)
    path = history_path(SB)
    assert os.path.exists(path), "session file was not created"
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    assert "පාස්කු" in raw, "Sinhala should be stored unescaped"
    print("session file:", os.path.relpath(path))

    # unsafe ids can never escape the chat_sessions folder
    evil = os.path.abspath(history_path("../../evil"))
    assert os.path.dirname(evil) == os.path.abspath(HISTORY_DIR)

    # clear affects only one session
    clear_history(SA)
    assert get_history(SA) == []
    assert len(get_history(SB)) == 2, "clearing session A must not touch session B"

    _cleanup()
    print("PASS")


def _voice(question, session_id):
    v = copy.deepcopy(SAMPLE_VOICE_INPUT)
    v.update({
        "route": "GENERATE",
        "intent": "DIRECT_ANSWER",
        "english_translation": question,
        "style_class": "Moderate",
        "prompt_modifier": "Answer clearly.",
        "personalization_flags": {"detail_level": "moderate"},
        "retrieved_chunk_id": None,
        "correction_applied": None,
    })
    if session_id is not None:
        v["session_id"] = session_id
    return v


def test_b_pipeline_sessions():
    print("\n=== B: two sessions through the pipeline (uses OpenAI) ===")
    vs = build_vectorstore()
    _cleanup()

    q1 = "What did Namal Rajapaksa say about the Easter Sunday attack?"
    q2 = "Who did he say was arrested?"   # follow-up with "he"

    run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q1, SA), {"age": 25},
                 skip_ocr=True, use_history=True)
    res_a = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q2, SA), {"age": 25},
                         skip_ocr=True, use_history=True)
    res_b = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q2, SB), {"age": 25},
                         skip_ocr=True, use_history=True)

    print("\nsession A follow-up STANDALONE:", res_a["standalone_query"])
    print("session B follow-up STANDALONE:", res_b["standalone_query"])

    assert len(get_history(SA)) == 2, "session A should have 2 turns"
    assert len(get_history(SB)) == 1, "session B should only have its own 1 turn"
    assert res_b["standalone_query"] == q2, "session B has no history -> query unchanged"
    if "Namal" in res_a["standalone_query"] or "Rajapaksa" in res_a["standalone_query"]:
        print("PASS (session A expanded 'he', session B did not)")
    else:
        print("CHECK: session A follow-up was not expanded - look at the line above")

    _cleanup()


def test_c_missing_session_id():
    print("\n=== C: request without session_id -> shared 'default' session ===")
    vs = build_vectorstore()
    _cleanup()
    run_pipeline(vs, SAMPLE_OCR_INPUT,
                 _voice("What did Namal Rajapaksa say about the Easter Sunday attack?", None),
                 {"age": 25}, skip_ocr=True, use_history=True)
    assert len(get_history(None)) == 1, "turn should be stored in the default session"
    print("default session file:", os.path.relpath(history_path(None)))
    _cleanup()
    print("PASS (you should also see the one-time '[history] WARNING' line above)")


if __name__ == "__main__":
    test_a_store()
    test_b_pipeline_sessions()
    test_c_missing_session_id()