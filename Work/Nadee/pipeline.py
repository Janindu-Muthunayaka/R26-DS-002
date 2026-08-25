# pipeline.py
from adapters import parse_voice_input, parse_ocr_input
from vectorstore import build_vectorstore, add_current_ocr_chunk
from retrieve import retrieve_context
from generate import generate_answer
from prompt import resolve_max_words

def run_pipeline(vectorstore, raw_ocr_input: dict, raw_voice_input: dict):
    voice = parse_voice_input(raw_voice_input)
    ocr = parse_ocr_input(raw_ocr_input)

    # route != GENERATE means the Voice module isn't asking RAG to answer at all
    if voice["route"] != "GENERATE":
        return {
            "intent": voice["intent"],
            "answer_si": f"'{voice['route']}' route සඳහා RAG පිළිතුරු ජනනය නොකරයි.",
            "retrieved_sources": [],
            "speakable_text": None,
        }

    add_current_ocr_chunk(vectorstore, ocr)

    docs = retrieve_context(
        vectorstore,
        query_text=voice["query_text"],
        intent=voice["intent"],
        retrieved_chunk_id=voice.get("retrieved_chunk_id"),
    )

    max_words = resolve_max_words(voice["style_class"], voice["personalization_flags"])

    try:
        answer = generate_answer(
            query_text=voice["query_text"],
            intent=voice["intent"],
            style_class=voice["style_class"],
            prompt_modifier=voice["prompt_modifier"],
            max_words=max_words,
            evidence_docs=docs,
        )
    except Exception:
        answer = "පිළිතුර ලබාගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න."

    return {
        "intent": voice["intent"],
        "answer_si": answer,
        "retrieved_sources": [d.metadata for d in docs],
        "speakable_text": answer,
    }