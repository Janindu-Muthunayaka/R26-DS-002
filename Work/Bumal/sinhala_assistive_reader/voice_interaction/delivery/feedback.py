# delivery/feedback.py
# Reads what the user says after an answer (and how playback ended) and
# decides whether it was a DELIVERY problem (didn't hear / too fast / too
# slow -> change rate, pauses, chunking) or a CONTENT problem (wants it
# simpler / more detailed -> the existing style channel, delivery unchanged).
#
# Phase 2b: apply_turn() reads ONLY the frame from
# intent_detection/modifiers.build_frame() — {intent, modifiers, evidence}.
# The REPEAT / slower / faster / style keyword checks live there now.
#
# Everything here is a pure function on a profile dict (delivery/voice_profile.py);
# storage and playback stay with the caller (app.py in Phase 3).

import copy

from delivery.voice_profile import count_turn, is_next_turn, repeatable
from intent_detection.modifiers import _words, repeat_in_sinhala, speed_request, style_modifier

# ── Tunable rule constants (ADAPTIVE_VOICE_PLAN.md, "Feedback rules") ─────
RATE_MIN, RATE_MAX = 0.6, 1.8
CHUNK_MIN, CHUNK_MAX = 1, 6
PAUSE_SCALE_MAX = 2.0
SLOWER_FACTOR, FASTER_FACTOR = 0.90, 1.10
REPEAT_RATE_FACTOR, REPEAT_PAUSE_STEP = 0.95, 0.1
DRIFT_FACTOR, CLEAN_ANSWERS_FOR_DRIFT = 1.02, 3
LONG_ANSWER_SEGMENTS = 3       # an answer this long counts for chunk_size rules
STYLE_INTENTS = {"SIMPLIFY", "ELABORATE", "STEP_BY_STEP"}
YES_NO_MAX_WORDS = 4
SPEED_QUESTION_SI = "හෙමින්ද ඉක්මනින්ද?"   # asked on CHANGE_SPEED with no direction

# ── Yes / no after a "continue?" check-in ─────────────────────────────────
# ONLY call this on the turn right after a check-in question. Elsewhere හරි
# is just "okay" ("හරි හරි නවත්තන්න" = STOP). "No" wins when both appear:
# "එපා කියවන්න" = don't read. (Phase 3) A reply with a repeat, speed or
# style word is a new request, not a yes, although කියන්න is a yes word:
# "කෙටියෙන් කියන්න", "හෙමින් කියන්න", "ආයෙ කියන්න".
YES_SI = {"ඔව්", "ඔව", "ඔවු", "හරි", "කියවන්න", "කියන්න", "දිගටම"}
NO_SI = {"නැහැ", "නෑ", "එපා", "ඇති", "නවත්වන්න", "නවත්තන්න", "නතර"}
YES_EN = {"yes", "yeah", "yep", "ok", "okay", "continue", "sure"}
NO_EN = {"no", "nope", "stop", "enough", "don't"}


def _yes_no_from(words, yes, no):
    if not words or len(words) > YES_NO_MAX_WORDS:
        return None          # long reply = a new request, not an answer
    if no & set(words):
        return "no"
    if yes & set(words):
        return "yes"
    return None


def yes_no(sinhala, english=None):
    """'yes' | 'no' | None (None = treat it as a normal new request)."""
    si_words = _words(sinhala)
    if si_words:
        if (repeat_in_sinhala(sinhala) or speed_request(sinhala, "")
                or style_modifier(sinhala, "")):
            return None
        return _yes_no_from(si_words, YES_SI, NO_SI)
    return _yes_no_from(_words(english), YES_EN, NO_EN)


# ── Answer to "හෙමින්ද ඉක්මනින්ද?" (slower or faster?) ─────────────────────
# Same idea as yes/no: ONLY call this on the turn right after the speed
# question, and only a short reply counts (හෙමින් / ඉක්මනින් / "ටිකක් හෙමින්").

def speed_answer(sinhala, english=None):
    """'slower' | 'faster' | None (None = treat it as a normal new request)."""
    si_words = _words(sinhala)
    if si_words:
        return speed_request(sinhala, "") if len(si_words) <= YES_NO_MAX_WORDS else None
    en_words = _words(english)
    if not en_words or len(en_words) > YES_NO_MAX_WORDS:
        return None
    return speed_request("", english)


# ── Feedback rules ─────────────────────────────────────────────────────────

def _decision():
    return {"channels": [], "repeat": False, "replay": None, "speed": None,
            "ask": None, "changes": {}, "reasons": []}


def _set(profile, decision, key, value):
    old = profile[key]
    if value != old:
        profile[key] = value
        decision["changes"][key] = [old, value]


def _scale_rate(profile, decision, factor):
    rate = round(min(RATE_MAX, max(RATE_MIN, profile["rate"] * factor)), 3)
    _set(profile, decision, "rate", rate)


def _add_channel(decision, channel):
    if channel not in decision["channels"]:
        decision["channels"].append(channel)


def apply_turn(profile, frame, now):
    """
    Applies the rules for one user turn, described by its frame
    (intent_detection/modifiers.build_frame). Returns (new_profile, decision);
    the input profile is not modified. Counts the turn (last_spoken.turns_since),
    so call it once per user turn.

    REPEAT learns ("didn't hear": rate x0.95, pause +0.1) only on the very next
    user turn after something was spoken. A later REPEAT replays without
    learning. Either way nothing older than 30 minutes is replayed.

    decision = {
      "channels": [] | ["delivery"] | ["content"] | both,
      "repeat":   True if this turn is a REPEAT,
      "replay":   a copy of last_spoken to replay, or None ("nothing to repeat"),
      "speed":    "slower" | "faster" | None,
      "ask":      None, or {"about": "speed", "question": SPEED_QUESTION_SI} when
                  the intent is CHANGE_SPEED but no slower/faster word was found;
                  the caller speaks the question and passes the next turn to
                  apply_speed_answer(),
      "changes":  {"rate": [old, new], ...},
      "reasons":  [str, ...]   # for the delivery log
    }
    """
    p, d = copy.deepcopy(profile), _decision()
    label = frame["intent"]
    speed = frame["modifiers"]["speed"]
    d["speed"] = speed

    if frame["modifiers"]["repeat"]:
        d["repeat"] = True
        p["counts"]["repeat"] += 1
        last = repeatable(p, now)
        d["replay"] = dict(last) if last else None
        if last is None:
            d["reasons"].append("repeat: nothing to repeat (none spoken, or older than 30 min)")
        elif is_next_turn(p):
            _add_channel(d, "delivery")
            _set(p, d, "pause_scale", round(min(PAUSE_SCALE_MAX, p["pause_scale"] + REPEAT_PAUSE_STEP), 2))
            if not speed:   # an explicit slower/faster below replaces the ×0.95
                _scale_rate(p, d, REPEAT_RATE_FACTOR)
            p["clean_streak"] = 0
            d["reasons"].append("repeat on the very next turn: didn't hear")
        else:
            d["reasons"].append(f"repeat {last.get('turns_since', '?')} turns later: replay only, no learning")

    if speed:
        _add_channel(d, "delivery")
        _scale_rate(p, d, SLOWER_FACTOR if speed == "slower" else FASTER_FACTOR)
        p["counts"][speed] += 1
        p["clean_streak"] = 0
        d["reasons"].append(f"asked {speed}")
    elif label == "CHANGE_SPEED":
        d["ask"] = {"about": "speed", "question": SPEED_QUESTION_SI}
        d["reasons"].append("CHANGE_SPEED without a slower/faster word: ask slower or faster")

    if label in STYLE_INTENTS and not d["repeat"]:
        _add_channel(d, "content")
        d["reasons"].append(f"{label}: content (style) channel, delivery unchanged")

    count_turn(p)
    return p, d


def apply_speed_answer(profile, sinhala, english=None, now=None):
    """
    The turn right after SPEED_QUESTION_SI was asked. A short slower/faster
    reply sets the speed exactly like an explicit request (rate x0.90 / x1.10).
    Returns (new_profile, decision), or (profile, None) when the reply is not
    a slower/faster answer — then it is a normal new request.
    """
    speed = speed_answer(sinhala, english)
    if speed is None:
        return profile, None
    frame = {"intent": "CHANGE_SPEED",
             "modifiers": {"repeat": False, "speed": speed, "style": None},
             "evidence": {"speed_from": "speed_question_answer"}}
    p, d = apply_turn(profile, frame, now)
    d["reasons"][-1] += " (answer to the speed question)"
    return p, d


def apply_playback(profile, playback, now=None):
    """
    Applies the rules for how an answer's playback ended (laptop_player
    result: completed, n_segments, ...). Returns (new_profile, decision).
    """
    p, d = copy.deepcopy(profile), _decision()
    completed = bool(playback.get("completed"))
    is_long = playback.get("n_segments", 0) >= LONG_ANSWER_SEGMENTS

    if not completed:
        p["counts"]["stopped"] += 1
        p["clean_streak"] = 0

    if is_long:
        history = (p["long_history"] + [not completed])[-3:]
        p["long_history"] = history
        if sum(history) >= 2:
            _add_channel(d, "delivery")
            _set(p, d, "chunk_size", max(CHUNK_MIN, p["chunk_size"] - 1))
            p["long_history"] = []   # start counting again at the new size
            d["reasons"].append("stopped in 2 of the last 3 long answers: smaller chunks")
        elif len(history) == 3 and not any(history):
            _set(p, d, "chunk_size", min(CHUNK_MAX, p["chunk_size"] + 1))
            p["long_history"] = []
            d["reasons"].append("3 long answers listened to the end: bigger chunks")

    if completed:
        p["clean_streak"] += 1
        if p["clean_streak"] >= CLEAN_ANSWERS_FOR_DRIFT:
            _scale_rate(p, d, DRIFT_FACTOR)
            p["clean_streak"] = 0
            d["reasons"].append(f"{CLEAN_ANSWERS_FOR_DRIFT} answers with no complaint: slightly faster")

    return p, d
