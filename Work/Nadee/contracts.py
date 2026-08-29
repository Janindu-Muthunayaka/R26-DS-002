# contracts.py

# Real OCR shape — from your OCR_INPUT.txt, the "council agreement" sentence
SAMPLE_OCR_INPUT = {
    "corrected_text": "ඉංග්‍රීසි ජාතික ආක්‍රමණිකයන් හා ලක්දිව එකල පැවති එකම නිදහස් රට වූ සිංහලේ රදළ වරුන් අතර ඇති කර ගත් අවබෝධතා ගිවිසුමකි.",
    "tokens": [
        {"original": "ආක්‍රමණීකයන්", "corrected": "ආක්‍රමණිකයන්", "label": "ERROR", "confidence": 0.42, "was_changed": True},
        {"original": "රදල", "corrected": "රදළ", "label": "ERROR", "confidence": 0.51, "was_changed": True},
        {"original": "ඇවබෝධතා", "corrected": "අවබෝධතා", "label": "ERROR", "confidence": 0.38, "was_changed": True},
        {"original": "ගිවිසුමක්", "corrected": "ගිවිසුමකි", "label": "ERROR", "confidence": 0.47, "was_changed": True},
    ],
}

# Case 7 from sample_personalization_outputs.json — ELABORATE / Detailed / full.
# Should resolve to max_words=500 via resolve_max_words(), and anchor on chunk_article_2.
SAMPLE_VOICE_INPUT = {
    "route": "GENERATE",
    "intent": "ELABORATE",
    "english_translation": "Give me the full details, don't leave anything out.",
    "style_class": "Detailed",
    "prompt_modifier": "Provide a thorough, comprehensive explanation with full context and detail.",
    "personalization_flags": {"detail_level": "full"},
    "retrieved_chunk_id": "chunk_article_2",
    "correction_applied": None,
}

RAG_OUTPUT_SCHEMA = {
    "intent": None,
    "answer_si": None,
    "retrieved_sources": [],
    "speakable_text": None,
}