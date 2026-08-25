from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template("""
ඔබ දෘෂ්ටි විනාශයට ලක් වූ පුද්ගලයින් සඳහා සිංහල කියවීමේ සහායකයෙකි.

උපදෙස්:
- පිළිතුර සිංහල භාෂාවෙන් පමණක් ලබා දෙන්න.
- පහත දක්වා ඇති සාක්ෂි (evidence) මත පමණක් පදනම් වන්න.
- ප්‍රමාණවත් තොරතුරු නොමැති නම්, "කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත." ලෙස පිළිතුරු දෙන්න.
- {prompt_modifier}
- උපරිම වචන ගණන: {max_words}.

Intent: {intent}
Style: {style_class}

Evidence:
{evidence}

User question (translated): {query_text}

පිළිතුර:
""")

# Voice's personalization_flags SOMETIMES sends detail_level, but the values used are not
# limited to brief/moderate/detailed — "structured" and "full" have also been observed.
# personalization_flags without detail_level at all (e.g. {"language_style": "simple"}) is
# also a valid shape. style_class is always present, so it's the reliable fallback signal —
# detail_level (if sent and recognized) overrides it. Both maps are matched case-insensitively
# since style_class has been observed as "StepByStep" (mixed case).
DETAIL_LEVEL_WORD_LIMITS = {
    "brief": 80,
    "moderate": 200,
    "detailed": 400,
    "structured": 300,   # step-by-step breakdowns need room for numbered steps
    "full": 500,          # "don't leave anything out" — the fullest tier
}

STYLE_CLASS_WORD_LIMITS = {
    "simple": 80,
    "moderate": 200,
    "detailed": 400,
    "stepbystep": 300,
}

def resolve_max_words(style_class: str, personalization_flags: dict) -> int:
    detail_level = personalization_flags.get("detail_level")
    if detail_level in DETAIL_LEVEL_WORD_LIMITS:
        return DETAIL_LEVEL_WORD_LIMITS[detail_level]
    key = (style_class or "").lower().replace(" ", "").replace("_", "")
    return STYLE_CLASS_WORD_LIMITS.get(key, STYLE_CLASS_WORD_LIMITS["moderate"])