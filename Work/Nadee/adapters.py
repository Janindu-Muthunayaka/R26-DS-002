# adapters.py

def parse_voice_input(raw: dict) -> dict:
    required = ["route", "intent", "english_translation", "style_class",
                "prompt_modifier", "personalization_flags"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Voice input missing required fields: {missing}")
    return {
        "route": raw["route"],
        "intent": raw["intent"],
        "query_text": raw["english_translation"],       # renamed for internal use
        "style_class": raw["style_class"],
        "prompt_modifier": raw["prompt_modifier"],
        "personalization_flags": raw.get("personalization_flags", {}),
        "retrieved_chunk_id": raw.get("retrieved_chunk_id"),  # may be None/missing
        "correction_applied": raw.get("correction_applied"),
    }

def parse_ocr_input(raw: dict) -> dict:
    if "corrected_text" not in raw:
        raise ValueError("OCR input missing corrected_text")
    return {
        "corrected_text": raw["corrected_text"],
        "tokens": raw.get("tokens", []),
    }