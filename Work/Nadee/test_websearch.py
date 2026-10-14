# test_websearch.py  — run with:  python test_websearch.py
from dotenv import load_dotenv
load_dotenv()

import copy
from websearch import is_refusal, web_search
from history import clear_history
from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT


def test_a_refusal_detection():
    print("\n=== A: is_refusal (no API calls) ===")
    assert is_refusal("කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත.")
    assert is_refusal("කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත")
    assert not is_refusal("නාමල් රාජපක්ෂ යනු ශ්‍රී ලංකා පොදුජන පෙරමුණේ ජාතික සංවිධායකයා ය.")
    print("PASS")


def test_b_web_search():
    print("\n=== B: web_search (1 Tavily call) ===")
    docs = web_search("Who is Namal Rajapaksa?")
    assert docs, "no web results (check TAVILY_API_KEY in .env)"
    for d in docs:
        print("-", d.metadata["title"], "|", d.metadata["source_url"])
        print("  ", d.page_content[:150].replace("\n", " "))
    print("PASS (check by eye that the results are about Namal Rajapaksa)")


def _voice(question):
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
    return v


def test_c_pipeline():
    print("\n=== C: pipeline with / without web fallback (uses OpenAI + Tavily) ===")
    vs = build_vectorstore()
    clear_history()

    # A question the newspaper corpus cannot answer.
    q = "What is the capital city of Japan?"

    off = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q), {"age": 25}, skip_ocr=True)
    print("\n[web OFF] searched:", off["web_searched"], "| used:", off["web_used"])
    print("ANSWER:", off["answer_si"][:200])
    assert off["web_searched"] is False and off["web_used"] is False

    on = run_pipeline(vs, SAMPLE_OCR_INPUT, _voice(q), {"age": 25},
                      skip_ocr=True, use_web_search=True)
    print("\n[web ON ] searched:", on["web_searched"], "| used:", on["web_used"])
    print("ANSWER:", on["answer_si"][:300])
    print("WEB SOURCES:", [s.get("source_url") for s in on["retrieved_sources"]
                           if s.get("source_type") == "web"])
    print("PASS (OFF should be the refusal sentence; ON should answer about Tokyo "
          "and start by saying the info came from the web)")


if __name__ == "__main__":
    test_a_refusal_detection()
    test_b_web_search()
    test_c_pipeline()