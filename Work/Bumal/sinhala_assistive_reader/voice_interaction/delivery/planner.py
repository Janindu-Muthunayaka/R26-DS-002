# delivery/planner.py
# Turns a Sinhala answer into a speech plan: the segments to speak, the pause
# after each, where to check in with the user, and the speech rate.
#
# Pure function — no files, network or audio — so it can run unchanged in the
# laptop player now and behind svc-voice /deliver later (ADAPTIVE_VOICE_PLAN.md,
# Phase 6). The returned dict is the contract the phone app will read:
#
#   {"rate": 0.85, "pitch": 1.0, "preview": "කරුණු 3ක් තියෙනවා",
#    "segments": [{"text": "...", "pause_after_ms": 700}, ...],
#    "checkpoints": [2], "replay": false}
#
# "rate" means the same as Android TextToSpeech.setSpeechRate() (1.0 = normal).

import re

DEFAULT_PROFILE = {"rate": 1.0, "pause_scale": 1.0, "chunk_size": 3}

RATE_MIN, RATE_MAX = 0.6, 1.8
CHUNK_MIN, CHUNK_MAX = 1, 6

# Pause after each segment, before pause_scale. StepByStep gets the longest
# pause so each step can sink in.
BASE_PAUSE_MS = {"Simple": 400, "Detailed": 600, "StepByStep": 800}
FALLBACK_PAUSE_MS = 600

# A new segment starts before each spoken ordinal, even without a full stop,
# so every step of a StepByStep answer is its own segment.
ORDINALS = ["පළමුව", "දෙවනුව", "තෙවනුව", "සිව්වනුව", "පස්වනුව",
            "හයවනුව", "හත්වනුව", "අටවනුව", "නවවනුව", "දහවනුව"]

# Abbreviations that end in "." but do not end a sentence. Compared against
# the whitespace-separated token the full stop belongs to.
ABBREVIATIONS = {
    "ප.ව.", "පෙ.ව.", "ක්‍රි.ව.", "ක්‍රි.පූ.", "අ.පො.ස.", "ශ්‍රී.ල.නි.",
    "dr.", "mr.", "mrs.", "ms.", "prof.", "no.", "st.", "etc.", "e.g.", "i.e.",
}

# Preview spoken before a long answer. Counts the numbered points when the
# answer has spoken ordinals (an intro sentence is not a point), otherwise
# the segments.
PREVIEW_POINTS = "කරුණු {n}ක් තියෙනවා"   # "there are {n} points"
PREVIEW_PARTS = "කොටස් {n}ක් තියෙනවා"    # "there are {n} parts"

_ORDINAL_RE = re.compile(r"\s+(?=(?:" + "|".join(ORDINALS) + r")(?:\s|,|$))")


def _clamp(value, low, high):
    return max(low, min(high, value))


def _ends_with_abbreviation(text):
    tokens = text.split()
    return bool(tokens) and tokens[-1].lower() in ABBREVIATIONS


def split_segments(text):
    """
    Splits an answer into speakable segments.

    Breaks after ".", "?" or "!" when followed by whitespace or the end of the
    text — so "25.50" never splits — unless the full stop closes a known
    abbreviation ("ප.ව."). Then breaks again before each spoken ordinal.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return []

    sentences, start = [], 0
    for match in re.finditer(r"[.?!](?=\s|$)", text):
        end = match.end()
        piece = text[start:end]
        if match.group() == "." and _ends_with_abbreviation(piece):
            continue
        sentences.append(piece.strip())
        start = end
    if text[start:].strip():
        sentences.append(text[start:].strip())

    segments = []
    for sentence in sentences:
        segments.extend(p.strip() for p in _ORDINAL_RE.split(sentence) if p.strip())
    return segments


def _preview(texts, chunk_size):
    """
    Spoken overview, only when the answer is longer than one chunk — and not
    when the answer already opens by announcing its points ("ප්‍රධාන කරුණු
    තුනක් තියෙනවා"), which would make the preview say it twice.
    """
    if len(texts) <= chunk_size or "කරුණු" in texts[0]:
        return None
    n_points = sum(1 for t in texts if t.startswith(tuple(ORDINALS)))
    if n_points >= 2:
        return PREVIEW_POINTS.format(n=n_points)
    return PREVIEW_PARTS.format(n=len(texts))


def _checkpoints(n_segments, chunk_size):
    """Check in after every chunk_size segments, never after the last one."""
    return list(range(chunk_size, n_segments, chunk_size))


def plan_speech(answer_si, style, profile=None, replay=False):
    """
    Builds the speech plan for one answer.

    answer_si — the Sinhala answer text (from Component 3, or the last spoken
                text when replaying)
    style     — "Simple" | "Detailed" | "StepByStep" (unknown -> 600 ms pauses)
    profile   — the user's delivery settings; missing keys use DEFAULT_PROFILE
    replay    — True when this is a REPEAT of something already spoken
    """
    settings = {**DEFAULT_PROFILE, **(profile or {})}
    rate = round(_clamp(float(settings["rate"]), RATE_MIN, RATE_MAX), 3)
    pause_scale = max(0.0, float(settings["pause_scale"]))
    chunk_size = int(_clamp(int(settings["chunk_size"]), CHUNK_MIN, CHUNK_MAX))

    pause_ms = int(round(BASE_PAUSE_MS.get(style, FALLBACK_PAUSE_MS) * pause_scale))
    texts = split_segments(answer_si)
    segments = [
        {"text": t, "pause_after_ms": pause_ms if i < len(texts) - 1 else 0}
        for i, t in enumerate(texts)
    ]

    return {
        "rate": rate,
        "pitch": 1.0,
        "preview": _preview(texts, chunk_size),
        "segments": segments,
        "checkpoints": _checkpoints(len(segments), chunk_size),
        "replay": bool(replay),
    }
