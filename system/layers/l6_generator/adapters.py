# adapters.py
try:
    from .prompt import age_to_group
except (ImportError, ValueError):
    from prompt import age_to_group


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
        "session_id": raw.get("session_id"),            # optional, may be None/missing
        "original_text": raw.get("original_text") or raw.get("sinhala_input"),  # optional Sinhala query text
    }


def parse_ocr_input(raw: dict) -> dict:
    if "corrected_text" not in raw:
        raise ValueError("OCR input missing corrected_text")
    return {
        "corrected_text": raw["corrected_text"],
        "tokens": raw.get("tokens", []),
    }


def parse_request_context(raw: dict) -> dict:
    """Accepts either an explicit age_group (e.g. "14-16") or a raw numeric
    age (e.g. 15) from the client. If both are given, age_group wins.
    If neither is given, age_group is None and resolve_age_profile()
    falls back to DEFAULT_AGE_BAND ("18+")."""
    raw = raw or {}

    age_group = raw.get("age_group")
    raw_age = raw.get("age")

    if not age_group and raw_age is not None:
        age_group = age_to_group(raw_age)

    return {
        "age_group": age_group,
    }
