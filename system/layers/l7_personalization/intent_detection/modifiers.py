# intent_detection/modifiers.py
# Phase 2b (ADAPTIVE_VOICE_PLAN.md): one sentence = ONE main action + MODIFIERS.
#   "ආයෙත් හෙමින් කියන්න" (say it again, slowly) -> REPEAT + slower
#   "කෙටියෙන් කියන්න කවුද දිනුවේ" (briefly, who won) -> ASK_QUESTION + Simple
#
# The LLM gives the main action; the modifiers come from keyword checks here:
#   repeat (bool), speed ("slower" | "faster" | None), style ("Simple" |
#   "Detailed" | "StepByStep" | None).
# build_frame() joins them into {intent, modifiers, evidence}, the one object
# that later stages (delivery/feedback.py) read.
#
# Keyword checks look at the SINHALA text first and the NLLB English second:
# NLLB often drops the very word we need ("ආයෙ" -> "I'm going back."), and
# no check on the English can recover a word that is already gone.
#
# Pure functions only — importing this file loads no model.

import re

# ── Text helpers ───────────────────────────────────────────────────────────

def _clean(text):
    """Lower-case, punctuation to spaces, single spaces."""
    text = re.sub(r"[,.?!;:\"'“”‘’()…]+", " ", str(text or "").lower())
    return " ".join(text.split())


def _words(text):
    return _clean(text).split()


def _any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def _label(intent):
    """Same canonical form as llm_backend.normalize_label; '' -> 'UNKNOWN'."""
    return re.sub(r"[\s\-]+", "_", str(intent or "").strip().upper()).strip("_") or "UNKNOWN"


# ── REPEAT ─────────────────────────────────────────────────────────────────
# Whole words only for නැවත, so නැවතී ("stopped") is not "again". ආපසු is
# not in this list: it also means "back" (PREVIOUS) — see APASU_* below.
REPEAT_WORDS_SI = {"ආයෙ", "ආයෙත්", "ආයේ", "ආයෙම", "නැවත", "නැවතත්"}
REPEAT_PHRASES_SI = [
    r"තව(ත්)? (පාරක්|සැරයක්|වතාවක්|වරක්)",       # one more time
    r"(ඇහු|ඇසු)[නණ][ේෙ] (නෑ|නැහැ|නැත)",           # didn't hear
]
# ආපසු / ආපහු (Bumal, 14 Oct): directly followed by කියන්න / කියවන්න = "say /
# read it again" (repeat). With යන්න (go) or පිටුව (page) anywhere = "go back"
# (PREVIOUS), never a repeat: "කලින් පිටුවට ආපහු යන්න". යන්න must start a
# word: කියන්න ends in the same letters.
APASU_REPEAT_SI = r"(ආපසු|ආපහු) (කියන්න|කියවන්න)"
APASU_BACK_SI = r"(^| )(යන්න|පිටුව)"
REPEAT_EN = [r"\bagain\b", r"\brepeat\b", r"\bone more time\b", r"\bonce more\b",
             r"\b(didn't|did not|couldn't|could not|can't|cannot) hear\b"]

# "Again" next to one of these is a change/style request (REPHRASE, SIMPLIFY,
# EXPLAIN ...) that needs a new answer, not a replay — e.g. sam_24, m_13.
NOT_REPEAT_SI = ["වෙනස්", "වෙනත්", "වෙන විදිහ", "වෙන වචන", "වෙන ක්‍රම", "සරල",
                 "ලේසි", "කෙටියෙන්", "විස්තර", "පැහැදිලි", "පියවරෙන්"]
NOT_REPEAT_EN = [r"\bchange", r"\bsimpl", r"\beas(y|ier|ily)\b", r"\banother way\b",
                 r"\bother words\b", r"\bdifferent", r"\bexplain", r"\bdetail",
                 r"\bstep by step\b", r"\bbrief", r"\bsummar"]


def _apasu_repeat(si):
    return bool(re.search(APASU_REPEAT_SI, si)) and not re.search(APASU_BACK_SI, si)


def _repeat_signal_si(si):
    si = _clean(si)
    return bool(REPEAT_WORDS_SI & set(si.split())) or _any(REPEAT_PHRASES_SI, si) or _apasu_repeat(si)


def _not_repeat_si(si):
    return any(w in _clean(si) for w in NOT_REPEAT_SI)


def repeat_in_sinhala(sinhala):
    """True if the Sinhala text alone says "again" with no change/style word."""
    si = _clean(sinhala)
    return bool(si) and not _not_repeat_si(si) and _repeat_signal_si(si)


def _repeat_with_source(sinhala, english, intent):
    si, en = _clean(sinhala), _clean(english)
    if si and _not_repeat_si(si):
        return False, "sinhala_change_or_style_word"
    if si and _repeat_signal_si(si):
        return True, "sinhala"
    if not si and _any(NOT_REPEAT_EN, en):
        return False, "english_change_or_style_word"
    if _label(intent) == "REPEAT":
        return True, "intent"
    if _any(REPEAT_EN, en) and not _any(NOT_REPEAT_EN, en):
        return True, "english"
    return False, None


def is_repeat(sinhala, english, intent=None):
    """
    True if the user wants the last spoken text again (same text, no new
    Component 3 answer).

    Sinhala first: a change/style word blocks it; a repeat word confirms it.
    Then the intent label, then the English (whose own change/style words
    block it, e.g. "Change the sentence a little and say it again").
    """
    return _repeat_with_source(sinhala, english, intent)[0]


# ── Slower / faster ────────────────────────────────────────────────────────
_FAST_SI = r"(වේගෙන්|වේගයෙන්|ඉක්මනින්|ඉක්මනට|ඉක්මණින්|ඉක්මණට)"
_SLOW_SI = r"(හෙමින්|සෙමින්|හෙමිහිට)"
_SPEED_SI = r"(වේගය|වේගෙ|වේගේ)"
_VERY_SI = r"(ගොඩක්|හුඟක්|හුගක්|වැඩියෙන්|වැඩිපුර|ඕනවට වඩා|ඕනෑවට වඩා)"
_COMMAND_SI = r"(කියවන්න|කියන්න|කරන්න)"
_DESCRIBING_SI = r"(කියවන්නේ|කියන්නේ|කතා කරන්නේ)"

# Complaints mean the OPPOSITE of the bare word, so they are checked first:
# "ඔයා කියවන්නේ ගොඩක් වේගෙන්" ("you read too fast") wants SLOWER. "Very fast"
# followed by a command ("ගොඩක් වේගෙන් කියවන්න") is a real "faster".
TOO_FAST_SI = [_VERY_SI + " " + _FAST_SI + "(?! " + _COMMAND_SI + ")",
               _SPEED_SI + " වැඩියි",
               _DESCRIBING_SI + " .*" + _FAST_SI]
TOO_SLOW_SI = [_VERY_SI + " " + _SLOW_SI + "(?! " + _COMMAND_SI + ")",
               _SPEED_SI + " අඩුයි", _SLOW_SI + " වැඩියි",
               _DESCRIBING_SI + " .*" + _SLOW_SI]
SLOWER_SI = [_SLOW_SI, _SPEED_SI + " අඩු(?!යි)"]
FASTER_SI = [_FAST_SI, _SPEED_SI + " වැඩි(?!යි)"]

TOO_FAST_EN = [r"\btoo (fast|quick)", r"\bso fast\b"]
TOO_SLOW_EN = [r"\btoo slow", r"\bso slow\b"]
SLOWER_EN = [r"\bslow(er|ly)?\b", r"\bslow down\b"]
FASTER_EN = [r"\bfast(er)?\b", r"\bquick(er|ly)?\b", r"\bspeed up\b", r"\bhurry\b"]


def _speed_from(text, too_fast, too_slow, slower, faster):
    if not text:
        return None
    if _any(too_fast, text):
        return "slower"
    if _any(too_slow, text):
        return "faster"
    want_slower, want_faster = _any(slower, text), _any(faster, text)
    if want_slower and want_faster:
        return None          # contradictory — change nothing
    return "slower" if want_slower else "faster" if want_faster else None


def _speed_with_source(sinhala, english):
    si = _speed_from(_clean(sinhala), TOO_FAST_SI, TOO_SLOW_SI, SLOWER_SI, FASTER_SI)
    if si:
        return si, "sinhala"
    en = _speed_from(_clean(english), TOO_FAST_EN, TOO_SLOW_EN, SLOWER_EN, FASTER_EN)
    return (en, "english") if en else (None, None)


def speed_request(sinhala, english):
    """'slower' | 'faster' | None. Sinhala decides if it says anything."""
    return _speed_with_source(sinhala, english)[0]


# ── Pauses / part size / volume (TURN_CONDITIONS.xlsx D8-D11) ──────────────
# "නවත්තලා නවත්තලා" (stopping now and then) = more pauses; "නොනවත්වා" (without
# stopping) = fewer. නොනවත්වා is checked first: it contains නවත්.
LESS_PAUSE_SI = [r"නොනවත්ව", r"නොනැවතී", r"එක දිගට", r"පරතරය අඩු", r"නැවතීම් අඩු"]
MORE_PAUSE_SI = [r"නවත්තලා", r"නවත්වලා", r"නවත්තමින්", r"ඉඩ තියලා", r"ඉඩ තබා",
                 r"පරතරය වැඩි", r"නැවතීම් වැඩි", r"නැවතී නැවතී"]
LESS_PAUSE_EN = [r"\b(fewer|less|shorter|no) pauses?\b", r"\bwithout (stopping|pausing|pauses)\b"]
MORE_PAUSE_EN = [r"\bmore pauses?\b", r"\blonger pauses?\b", r"\bpause (more|between)\b"]
SMALLER_CHUNK_SI = [r"කෑලි කෑලි", r"කොටසින් කොටස", r"ටික ටික", r"පොඩි කොටස්"]
SMALLER_CHUNK_EN = [r"\bsmaller (parts|pieces|chunks)\b", r"\bbit by bit\b", r"\bpiece by piece\b"]
VOLUME_SI = [r"ශබ්දය", r"සද්දෙ", r"සද්දය", r"හයියෙන්"]
VOLUME_EN = [r"\bvolume\b", r"\blouder\b", r"\bquieter\b"]


def pause_request(sinhala, english):
    """'more' | 'less' | None — an explicit request about pauses."""
    si, en = _clean(sinhala), _clean(english)
    if si:
        if _any(LESS_PAUSE_SI, si):
            return "less"
        if _any(MORE_PAUSE_SI, si):
            return "more"
        return None
    if _any(LESS_PAUSE_EN, en):
        return "less"
    if _any(MORE_PAUSE_EN, en):
        return "more"
    return None


def chunk_request(sinhala, english):
    """'smaller' | None — "say it in smaller parts"."""
    si, en = _clean(sinhala), _clean(english)
    if si:
        return "smaller" if _any(SMALLER_CHUNK_SI, si) else None
    return "smaller" if _any(SMALLER_CHUNK_EN, en) else None


SPEED_MENTION_EN = [r"\bspeed\b", r"\bpace\b"]
SUMMARY_SI = [r"සාරාංශ", r"සංක්ෂේප"]
SUMMARY_EN = [r"\bsummar", r"\bin short\b", r"\bgist\b"]


def speed_mentioned(sinhala, english):
    """True if the sentence talks about speed at all ("වේගය වෙනස් කරන්න",
    change the speed) — with or without saying slower or faster."""
    si, en = _clean(sinhala), _clean(english)
    return bool(re.search(_SPEED_SI, si)) if si else _any(SPEED_MENTION_EN, en)


def summary_request(sinhala, english):
    """True if the user asks for a summary ("සාරාංශයක්", summarise)."""
    si, en = _clean(sinhala), _clean(english)
    return _any(SUMMARY_SI, si) if si else _any(SUMMARY_EN, en)


def volume_request(sinhala, english):
    """True if the user is asking about loudness (a phone setting)."""
    si, en = _clean(sinhala), _clean(english)
    return _any(VOLUME_SI, si) if si else _any(VOLUME_EN, en)


# ── Style (English keywords, moved from personalization/diagnostic.py) ─────
# Lexical grounding: words in the English translation that genuinely
# indicate each style. diagnostic.py imports these back unchanged — the style
# model's evidence rules use them to corroborate/reject a flag and as a
# direct evidence source when the flag is missing or untrustworthy.
STYLE_KEYWORDS = {
    "Simple": [
        "simple", "simply", "simpler", "simplify", "easy", "easier",
        "plain", "basic", "brief", "briefly", "short", "shorter",
        "summary", "summarize", "summarise", "concise", "quick",
    ],
    "Detailed": [
        "detail", "details", "detailed", "elaborate", "more", "full",
        "fully", "thorough", "deep", "deeply", "depth", "deeper", "expand",
        "comprehensive", "longer", "complete", "everything", "technical",
    ],
    "StepByStep": [
        "step", "steps", "stepwise", "one by one", "sequentially",
        "in order", "numbered", "walk me through", "stages",
    ],
}


def _keyword_scores(english_text):
    """Counts style-indicating keywords in the translated sentence."""
    if not english_text:
        return {}
    text = str(english_text).lower()
    scores = {}
    for style, words in STYLE_KEYWORDS.items():
        hits = 0
        for w in words:
            if " " in w:
                if w in text:
                    hits += 1
            elif re.search(rf"\b{re.escape(w)}\b", text):
                hits += 1
        if hits:
            scores[style] = hits
    return scores


def _winner(scores):
    """The single top style, or None on no hits / a tie (don't guess)."""
    if not scores:
        return None
    best = max(scores.values())
    winners = [s for s, n in scores.items() if n == best]
    return winners[0] if len(winners) == 1 else None


def style_from_keywords(english_text):
    """The style indicated by the words of the sentence itself, or None.
    Ties (a sentence containing both 'simple' and 'more') return None so we
    don't guess — the model handles it instead."""
    return _winner(_keyword_scores(english_text))


# ── Style modifier (Sinhala) ───────────────────────────────────────────────
# The style words already in NOT_REPEAT_SI, each mapped to its style. The
# English list above is too broad for a modifier ("one more time" -> "more"
# -> Detailed, "summary" -> Simple), so it is used only when there is no
# Sinhala text at all. "විස්තර නැතුව" (without details) means Simple (sam_5).
_WITHOUT_SI = r"(නැතුව|නැතිව|නොමැතිව)"
STYLE_SI = {
    "Simple": [r"සරල", r"ලේසි", r"කෙටියෙන්", r"විස්තර " + _WITHOUT_SI],
    "Detailed": [r"විස්තර(?! " + _WITHOUT_SI + ")"],
    "StepByStep": [r"පියවරෙන්"],
}


def _style_with_source(sinhala, english):
    si = _clean(sinhala)
    if si:
        scores = {style: sum(bool(re.search(p, si)) for p in pats)
                  for style, pats in STYLE_SI.items()}
        style = _winner({s: n for s, n in scores.items() if n})
        return (style, "sinhala") if style else (None, None)
    style = style_from_keywords(english)
    return (style, "english") if style else (None, None)


def style_modifier(sinhala, english):
    """'Simple' | 'Detailed' | 'StepByStep' | None — an explicit manner word
    (simply, briefly, in detail, step by step), not the intent itself."""
    return _style_with_source(sinhala, english)[0]


# ── Fusion: Sinhala repeat word vs. a wrong LLM label ──────────────────────
# NLLB turns "ආයෙ" into "I'm going back." and Llama then says PREVIOUS; "තව
# පාරක්" can come out as "Previously on ..." -> PREVIOUS. When the Sinhala
# itself says "again" and the LLM's label is one of these, REPEAT wins.
FUSION_OVERRIDABLE = {"PREVIOUS", "NEXT", "UNKNOWN"}


def fuse_intent(sinhala, llm_intent):
    """(final_intent, note). note is None when the LLM label is kept."""
    label = _label(llm_intent)
    if label in FUSION_OVERRIDABLE and repeat_in_sinhala(sinhala):
        return "REPEAT", f"Sinhala repeat word overrides LLM {label}"
    return label, None


# ── Modifiers the words did not find ───────────────────────────────────────
# The LLM label implies a modifier, but no Sinhala or English word matched it.
# These sentences are what the word lists are missing; main_flow.py saves
# them to the "unmatched_modifiers" table (personalization/logger.py).
STYLE_INTENT_TO_STYLE = {"SIMPLIFY": "Simple", "ELABORATE": "Detailed", "STEP_BY_STEP": "StepByStep"}


def unmatched_modifiers(intent, evidence):
    """
    Fields whose modifier comes only from the LLM intent:
      "repeat" - repeat is True only because the label is REPEAT
      "speed"  - the label is CHANGE_SPEED but no slower/faster word
      "style"  - the label is SIMPLIFY / ELABORATE / STEP_BY_STEP but no style word
    """
    out = []
    if evidence["repeat_from"] == "intent":
        out.append("repeat")
    if intent == "CHANGE_SPEED" and evidence["speed_from"] is None:
        out.append("speed")
    if intent in STYLE_INTENT_TO_STYLE and evidence["style_from"] is None:
        out.append("style")
    return out


# ── The frame ──────────────────────────────────────────────────────────────

def build_frame(sinhala, english, llm_intent):
    """
    {
      "intent":    final main action (LLM label, or REPEAT by the fusion rule),
      "modifiers": {"repeat": bool, "speed": "slower"|"faster"|None,
                    "style": "Simple"|"Detailed"|"StepByStep"|None,
                    "pause": "more"|"less"|None, "chunk": "smaller"|None,
                    "volume": bool},
      "evidence":  {"llm_intent": label as the LLM gave it,
                    "intent_from": "llm" | "fusion",
                    "fusion": note | None,
                    "repeat_from" / "speed_from" / "style_from": which check
                    decided it ("sinhala", "english", "intent", ...) or None,
                    "unmatched": unmatched_modifiers() — [] when the words
                    found every modifier the label implies}
    }
    """
    intent, fusion = fuse_intent(sinhala, llm_intent)
    repeat, repeat_from = _repeat_with_source(sinhala, english, intent)
    speed, speed_from = _speed_with_source(sinhala, english)
    style, style_from = _style_with_source(sinhala, english)
    if repeat:
        # A repeat replays the same text, so no style applies. (A Sinhala
        # style word would already have blocked the repeat.)
        style, style_from = None, None
    evidence = {
        "llm_intent": _label(llm_intent),
        "intent_from": "fusion" if fusion else "llm",
        "fusion": fusion,
        "repeat_from": repeat_from,
        "speed_from": speed_from,
        "style_from": style_from,
    }
    evidence["unmatched"] = unmatched_modifiers(intent, evidence)
    return {
        "intent": intent,
        "modifiers": {"repeat": repeat, "speed": speed, "style": style,
                      "pause": pause_request(sinhala, english),
                      "chunk": chunk_request(sinhala, english),
                      "volume": volume_request(sinhala, english),
                      "speed_mentioned": speed_mentioned(sinhala, english),
                      "summary": summary_request(sinhala, english)},
        "evidence": evidence,
    }
