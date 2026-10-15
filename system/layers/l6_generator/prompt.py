# prompt.py
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template("""
ඔබ දෘෂ්ටි විනාශයට ලක් වූ පුද්ගලයින් සඳහා සිංහල කියවීමේ සහායකයෙකි.

උපදෙස්:
- පිළිතුර සිංහල භාෂාවෙන් පමණක් ලබා දෙන්න.
- පහත දක්වා ඇති සාක්ෂි (Evidence) හි සඳහන් කරුණු ඇසුරෙන් අදාළ පිළිතුර ලබා දෙන්න. සාක්ෂි මගින් පිළිතුරු දිය නොහැකි නම් පමණක් "කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත." ලෙස පිළිතුරු දෙන්න.
- ප්‍රශ්නයේ "මේක", "මෙය", "එය" වැනි සර්වනාම තිබේ නම්, පෙර සංවාදය හෝ සාක්ෂි (Evidence) හි සඳහන් කරුණු ඇසුරෙන් එය කුමක්දැයි තේරුම් ගෙන අදාළ පිළිතුර ලබා දෙන්න.
- "පෙර සංවාදය" කොටස ඇත්තේ පරිශීලකයා කලින් ඇසූ දේ තේරුම් ගැනීමට පමණි; කරුණු සඳහා Evidence කොටස පමණක් භාවිතා කරන්න.
- පරිශීලකයා "තව කියන්න" වැනි ලෙස වැඩිදුර විස්තර ඉල්ලා ඇත්නම්, පෙර පිළිතුරේ දැනටමත් සඳහන් කළ කරුණු නැවත නොකියා, Evidence හි ඇති අලුත් කරුණු ලබා දෙන්න.
- {prompt_modifier}
- {age_instruction}
- උපරිම වචන ගණන: {max_words}.

Intent: {intent}
Style: {style_class}

පෙර සංවාදය (සන්දර්භය පමණි, සාක්ෂියක් නොවේ):
{chat_history}

Evidence:
{evidence}

පරිශීලකයාගේ ප්‍රශ්නය: {query_text}

පිළිතුර:
""")

DETAIL_LEVEL_WORD_LIMITS = {
    "brief": 80,
    "moderate": 200,
    "detailed": 400,
    "structured": 300,
    "full": 500,
}

STYLE_CLASS_WORD_LIMITS = {
    "simple": 80,
    "moderate": 200,
    "detailed": 400,
    "stepbystep": 300,
}

AGE_BAND_PROFILES = {
    "10-13": {
        "word_cap": 100,
        "instruction": (
            "පිළිතුර 10-13 හැවිරිදි දරුවෙකුට තේරෙන ඉතා සරල වචන වලින් ලියන්න. "
            "කෙටි වාක්‍ය භාවිතා කරන්න, සංකීර්ණ පදවි මගහරින්න, අවශ්‍ය නම් සරල උදාහරණයක් දෙන්න."
        ),
    },
    "14-16": {
        "word_cap": 250,
        "instruction": (
            "පිළිතුර 14-16 හැවිරිදි සිසුවෙකුට තේරෙන පැහැදිලි භාෂාවෙන් ලියන්න. "
            "අවශ්‍ය නම් සරල පදවි කෙටියෙන් පැහැදිලි කර භාවිතා කළ හැක."
        ),
    },
    "17-18": {
        "word_cap": 400,
        "instruction": (
            "පිළිතුර 17-18 හැවිරිදි උසස් පෙළ සිසුවෙකුට සුදුසු මට්ටමින්, "
            "තරමක් විස්තරාත්මකව හා නිවැරදි පදවි සමඟ ලියන්න."
        ),
    },
    "18+": {
        "word_cap": 500,
        "instruction": "පිළිතුර වැඩිහිටි පාඨකයෙකුට සුදුසු සම්පූර්ණ, විස්තරාත්මක මට්ටමින් ලියන්න.",
    },
}
DEFAULT_AGE_BAND = "18+"


def age_to_group(age) -> str:
    """Maps a raw user age (int) to one of the 4 supported age bands.
    Returns DEFAULT_AGE_BAND if age is None or not a valid number."""
    if age is None:
        return DEFAULT_AGE_BAND
    try:
        age = int(age)
    except (TypeError, ValueError):
        return DEFAULT_AGE_BAND

    if age < 10:
        return "10-13"
    elif 10 <= age <= 13:
        return "10-13"
    elif 14 <= age <= 16:
        return "14-16"
    elif 17 <= age <= 18:
        return "17-18"
    else:  # age > 18
        return "18+"


def resolve_age_profile(age_group: str) -> dict:
    key = (age_group or "").strip()
    if key not in AGE_BAND_PROFILES:
        key = key.replace("_", "-")
    return AGE_BAND_PROFILES.get(key, AGE_BAND_PROFILES[DEFAULT_AGE_BAND])


def resolve_max_words(style_class: str, personalization_flags: dict, age_cap: int | None = None) -> int:
    detail_level = (personalization_flags or {}).get("detail_level")
    if detail_level in DETAIL_LEVEL_WORD_LIMITS:
        base = DETAIL_LEVEL_WORD_LIMITS[detail_level]
    else:
        key = (style_class or "").lower().replace(" ", "").replace("_", "")
        base = STYLE_CLASS_WORD_LIMITS.get(key, STYLE_CLASS_WORD_LIMITS["moderate"])

    if age_cap is not None:
        return min(base, age_cap)
    return base
