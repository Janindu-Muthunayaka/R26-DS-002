# test_prompt_history.py  — run with:  python test_prompt_history.py
from dotenv import load_dotenv
load_dotenv()

import copy

from history import clear_history
from history_prompt import format_history_for_prompt
from prompt import PROMPT
from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT


def test_a_formatter():
    print("\n=== A: format_history_for_prompt (no API calls) ===")
    refusal = "කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත."
    turns = [
        {"question": "q1", "standalone_question": "s1", "answer": "a1"},
        {"question": "q2", "standalone_question": "s2", "answer": refusal},
        {"question": "q3", "standalone_question": "s3", "answer": "x" * 1000},
        {"question": "q4", "standalone_question": "s4", "answer": "a4"},
    ]
    out = format_history_for_prompt(turns)
    print(out[:300])
    assert "s2" not in out, "refusal turns must be skipped"
    assert "s1" not in out, "only the last 2 usable turns should be kept"
    assert "s3" in out and "s4" in out
    assert "x" * 401 not in out, "long answers must be truncated"
    assert format_history_for_prompt([]) == ""
    print("PASS")


def test_b_prompt_template():
    print("\n=== B: prompt template has {chat_history} (no API calls) ===")
    assert "chat_history" in PROMPT.input_variables
    text = PROMPT.format(
        prompt_modifier="m", age_instruction="a", max_words=100,
        intent="EXPLAIN", style_class="Simple",
        chat_history="MARKER_HISTORY", evidence="EV", query_text="Q",
    )
    assert "MARKER_HISTORY" in text
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
        "session_id": session_id,
    })
    return v


def test_c_ab_comparison():
    print("\n=== C: history in prompt OFF vs ON (uses OpenAI) ===")
    vs = build_vectorstore()
    questions = [
        "What did Namal Rajapaksa say about the Easter Sunday attack?",
        "Tell me more about it.",
        "Say that again in simpler words.",
    ]
    for label, in_prompt in (("OFF", False), ("ON", True)):
        sid = f"ph_{label.lower()}"
        clear_history(sid)
        print(f"\n################ history_in_prompt = {label} ################")
        for q in questions:
            res = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q, sid), {"age": 25},
                               skip_ocr=True, use_history=True,
                               history_in_prompt=in_prompt)
            print(f"\nUSER      : {q}")
            print(f"STANDALONE: {res['standalone_query']}")
            print(f"IN PROMPT : {len(res['chat_history_text'])} chars of history")
            print(f"ANSWER    : {res['answer_si'][:350]}")
        clear_history(sid)
    print("\nCHECK by eye: in the ON run, turn 2 should add NEW facts instead of "
          "repeating turn 1, and turn 3 should be a simpler version of the earlier answer.")


if __name__ == "__main__":
    test_a_formatter()
    test_b_prompt_template()
    test_c_ab_comparison()