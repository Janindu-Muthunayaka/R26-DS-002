# Tests for delivery/feedback.py. Run from voice_interaction/:
#   venv\Scripts\python.exe -m pytest tests -q
#
# Part 1: one test per row of the feedback rules table (ADAPTIVE_VOICE_PLAN.md).
# Part 3: yes/no after a check-in.
# Part 4: the speed question (CHANGE_SPEED with no slower/faster word).
# (Part 2, the keyword checks on the test sets, moved to tests/test_modifiers.py
# with the checks themselves.)
#
# apply_turn() reads only the frame, so each turn is built with build_frame().

from datetime import datetime, timedelta

import pytest

from data.test_samples import test_samples
from data.test_samples_multi import multi_samples
from delivery.feedback import (
    SPEED_QUESTION_SI, apply_playback, apply_speed_answer, apply_turn as _apply_frame,
    speed_answer, yes_no,
)
from delivery.voice_profile import new_profile, set_last_spoken
from intent_detection.modifiers import build_frame

NOW = datetime(2026, 10, 13, 12, 0, 0)
ANSWER = "පළමුව එක. දෙවනුව දෙක. තෙවනුව තුන."


def spoken(seconds_ago, profile=None):
    """A profile whose last answer finished `seconds_ago` before NOW; the
    next apply_turn() is the very next user turn after it."""
    p = profile or new_profile()
    return set_last_spoken(p, ANSWER, "answer", "StepByStep", NOW - timedelta(seconds=seconds_ago))


def apply_turn(profile, sinhala, english, intent, now):
    return _apply_frame(profile, build_frame(sinhala, english, intent), now)


def played(completed, n_segments=4):
    return {"completed": completed, "n_segments": n_segments}


# ── Part 1: the rules table ────────────────────────────────────────────────

def test_row_repeat_on_the_very_next_turn_replays_and_slows_down():
    p, d = apply_turn(spoken(10), "ආයෙ කියන්න", "Say it again", "REPEAT", NOW)
    assert d["repeat"] and d["replay"]["text"] == ANSWER
    assert d["channels"] == ["delivery"]
    assert p["rate"] == 0.95 and p["pause_scale"] == 1.1


def test_row_repeat_on_the_next_turn_learns_even_minutes_later():
    # No 20 s limit any more: only "is it the very next turn?" matters.
    p, d = apply_turn(spoken(10 * 60), "ආයෙ කියන්න", "Say it again", "REPEAT", NOW)
    assert d["channels"] == ["delivery"] and p["rate"] == 0.95
    assert "very next turn" in d["reasons"][0]


def test_row_repeat_on_a_later_turn_replays_only_no_learning():
    p, _ = apply_turn(spoken(5), "මේකේ මොකක්ද තියෙන්නේ", "What's in this?", "IDENTIFY_CONTENT", NOW)
    p, d = apply_turn(p, "ආයෙ කියන්න", "Say it again", "REPEAT", NOW)
    assert d["repeat"] and d["replay"]["text"] == ANSWER
    assert d["channels"] == [] and d["changes"] == {}
    assert p["rate"] == 1.0 and p["pause_scale"] == 1.0
    assert "1 turns later" in d["reasons"][0]


def test_second_repeat_in_a_row_does_not_learn_again_by_itself():
    # Learning needs a fresh last_spoken; the caller sets it after replaying.
    p, _ = apply_turn(spoken(5), "ආයෙ කියන්න", "Say it again", "REPEAT", NOW)
    p, d = apply_turn(p, "ආයෙ කියන්න", "Say it again", "REPEAT", NOW)
    assert d["changes"] == {} and p["rate"] == 0.95


def test_every_turn_counts_and_the_replay_is_a_copy():
    p, d = apply_turn(spoken(5), "ආයෙ", "Again", "REPEAT", NOW)
    assert p["last_spoken"]["turns_since"] == 1
    assert d["replay"]["turns_since"] == 0          # as it was when the turn started


def test_row_repeat_with_nothing_recent_says_nothing_to_repeat():
    _, d = apply_turn(new_profile(), "ආයෙ", "Again", "REPEAT", NOW)
    assert d["repeat"] and d["replay"] is None
    _, d = apply_turn(spoken(31 * 60), "ආයෙ", "Again", "REPEAT", NOW)
    assert d["replay"] is None and d["changes"] == {}   # 30-minute expiry kept


def test_row_slower_reduces_rate_by_10_percent():
    p, d = apply_turn(spoken(5), "ටිකක් හෙමින් කියවන්න", "Read a little slowly", "CHANGE_SPEED", NOW)
    assert d["speed"] == "slower" and d["channels"] == ["delivery"]
    assert p["rate"] == 0.9


def test_row_faster_increases_rate_by_10_percent():
    p, d = apply_turn(spoken(5), "ටිකක් ඉක්මනට කියවන්න", "Read a little faster", "CHANGE_SPEED", NOW)
    assert d["speed"] == "faster" and p["rate"] == 1.1


def test_row_speed_word_counts_even_when_intent_is_read_aloud():
    # "read the article slowly" -> READ_ALOUD intent, but still a delivery change
    p, d = apply_turn(new_profile(), "හෙමින් ලිපිය කියවන්න", "Read the article slowly", "READ_ALOUD", NOW)
    assert d["speed"] == "slower" and p["rate"] == 0.9


def test_row_stopped_in_2_of_last_3_long_answers_lowers_chunk_size():
    p = new_profile()
    p, d1 = apply_playback(p, played(False))
    p, d2 = apply_playback(p, played(True))
    assert p["chunk_size"] == 3                       # only 1 stop so far
    p, d3 = apply_playback(p, played(False))
    assert d3["changes"]["chunk_size"] == [3, 2]
    assert p["long_history"] == []                    # starts counting again


def test_row_short_answers_do_not_count_for_chunk_size():
    p = new_profile()
    for _ in range(3):
        p, _ = apply_playback(p, played(False, n_segments=2))
    assert p["chunk_size"] == 3


def test_row_style_correction_is_content_channel_delivery_unchanged():
    p, d = apply_turn(spoken(5), "සරල භාෂාවෙන් කියන්න", "Say it in simple language", "SIMPLIFY", NOW)
    assert d["channels"] == ["content"]
    assert d["changes"] == {} and not d["repeat"]
    assert p["rate"] == 1.0


def test_row_3_clean_answers_drift_rate_up_2_percent():
    p = new_profile()
    p, _ = apply_playback(p, played(True))
    p, _ = apply_playback(p, played(True))
    assert p["rate"] == 1.0
    p, d = apply_playback(p, played(True))
    assert d["changes"]["rate"] == [1.0, 1.02]
    assert p["clean_streak"] == 0


def test_row_complaint_resets_the_clean_streak():
    p = new_profile()
    p, _ = apply_playback(p, played(True))
    p, _ = apply_playback(p, played(True))
    p, _ = apply_turn(p, "හෙමින් කියන්න", "Say it slowly", "CHANGE_SPEED", NOW)
    p, d = apply_playback(p, played(True))
    assert "rate" not in d["changes"]                 # streak restarted at the complaint


def test_row_change_speed_without_keyword_asks_and_changes_nothing():
    p, d = apply_turn(new_profile(), "වේගය වෙනස් කරන්න", "Change the speed", "CHANGE_SPEED", NOW)
    assert d["ask"] == {"about": "speed", "question": "හෙමින්ද ඉක්මනින්ද?"}
    assert d["speed"] is None and d["changes"] == {} and d["channels"] == []
    assert p == new_profile()
    assert any("CHANGE_SPEED" in r for r in d["reasons"])


def test_no_question_when_the_direction_is_known():
    _, d = apply_turn(new_profile(), "හෙමින් කියන්න", "Say it slowly", "CHANGE_SPEED", NOW)
    assert d["ask"] is None and d["speed"] == "slower"
    _, d = apply_turn(new_profile(), "ලිපිය කියවන්න", "Read the article", "READ_ALOUD", NOW)
    assert d["ask"] is None


# ── Part 1b: combinations and limits ───────────────────────────────────────

def test_repeat_and_slower_change_rate_once_not_twice():
    p, d = apply_turn(spoken(5), "ආයෙත් හෙමින් කියන්න", "Say it again slowly", "REPEAT", NOW)
    assert d["repeat"] and d["speed"] == "slower"
    assert p["rate"] == 0.9                           # explicit x0.90, not x0.90 x0.95
    assert p["pause_scale"] == 1.1                    # repeat-window pause still applies


def test_rate_and_chunk_size_stay_within_limits():
    p = new_profile()
    for _ in range(20):
        p, _ = apply_turn(p, "හෙමින් කියන්න", "", "CHANGE_SPEED", NOW)
    assert p["rate"] == 0.6
    for _ in range(20):
        p, _ = apply_turn(p, "ඉක්මනට කියන්න", "", "CHANGE_SPEED", NOW)
    assert p["rate"] == 1.8
    for _ in range(10):
        p, _ = apply_playback(p, played(False))
    assert p["chunk_size"] == 1


def test_long_clean_listening_raises_chunk_size():
    p = new_profile()
    for _ in range(3):
        p, d = apply_playback(p, played(True))
    assert d["changes"]["chunk_size"] == [3, 4]


def test_input_profile_is_not_modified():
    p = spoken(5)
    apply_turn(p, "ආයෙ", "Again", "REPEAT", NOW)
    apply_playback(p, played(False))
    assert p["rate"] == 1.0 and p["counts"]["repeat"] == 0 and p["long_history"] == []
    assert p["last_spoken"]["turns_since"] == 0


def test_feedback_reads_only_the_frame():
    # A hand-made frame with no text at all still drives the rules.
    frame = {"intent": "REPEAT", "modifiers": {"repeat": True, "speed": "slower", "style": None},
             "evidence": {}}
    p, d = _apply_frame(spoken(5), frame, NOW)
    assert d["repeat"] and d["speed"] == "slower" and p["rate"] == 0.9


def test_fused_repeat_reaches_feedback():
    # ආයෙ -> NLLB "I'm going back." -> Llama PREVIOUS; the frame says REPEAT
    p, d = apply_turn(spoken(10), "ආයෙ", "I'm going back.", "PREVIOUS", NOW)
    assert d["repeat"] and d["replay"]["text"] == ANSWER and p["rate"] == 0.95


# ── Part 3: yes / no after a check-in ──────────────────────────────────────

@pytest.mark.parametrize("sinhala, expected", [
    ("ඔව්", "yes"), ("හරි", "yes"), ("ඔව් කියවන්න", "yes"), ("හරි දිගටම කියවන්න", "yes"),
    ("නෑ", "no"), ("එපා", "no"), ("ඇති", "no"), ("කියවන්න එපා", "no"),
    ("හරි හරි නවත්තන්න", "no"),                     # sam_37: "okay okay stop" = no
    ("ඊළඟ පිටුවට යන්න", None),                       # a new request, not an answer
    ("කෙටියෙන් කියන්න", None),                       # style word: a new request (Phase 3)
    ("හෙමින් කියන්න", None),                         # speed word: a new request
    ("ආයෙ කියන්න", None),                            # repeat word: a new request
    ("හරි ආයෙ කියන්න", None),
    ("මේ ලිපියේ තියෙන දේ කෙටියෙන් සාරාංශයක් විදිහට දෙන්න", None),   # too long
])
def test_yes_no_sinhala(sinhala, expected):
    assert yes_no(sinhala) == expected


def test_yes_no_uses_english_only_without_sinhala():
    assert yes_no("", "Yes, continue") == "yes"
    assert yes_no("", "No, stop") == "no"
    assert yes_no("ඔව්", "No") == "yes"                # Sinhala decides


def test_yes_no_ignores_long_sentences_from_the_test_sets():
    for s in test_samples + multi_samples:
        if len(s["stt_output"].split()) > 4:
            assert yes_no(s["stt_output"]) is None, s["id"]


# ── Part 4: the speed question ─────────────────────────────────────────────

@pytest.mark.parametrize("sinhala, expected", [
    ("හෙමින්", "slower"), ("සෙමින්", "slower"), ("ටිකක් හෙමින්", "slower"), ("හෙමින් කියන්න", "slower"),
    ("ඉක්මනින්", "faster"), ("ඉක්මනට", "faster"), ("වේගෙන් කියවන්න", "faster"),
    ("හෙමින්ද ඉක්මනින්ද", None),                    # the question echoed back: no direction
    ("ඔව්", None),                                   # yes is not an answer to "which one?"
    ("ඊළඟ පිටුවට යන්න", None),                       # a new request
    ("මට ලිපිය ටිකක් හෙමින් පැහැදිලි කරලා දෙන්න", None),   # too long: a new request
])
def test_speed_answer_sinhala(sinhala, expected):
    assert speed_answer(sinhala) == expected


def test_speed_answer_uses_english_only_without_sinhala():
    assert speed_answer("", "Slower") == "slower"
    assert speed_answer("", "Faster please") == "faster"
    assert speed_answer("හෙමින්", "Faster") == "slower"   # Sinhala decides


def test_speed_question_then_slower_reply_sets_the_rate():
    p, d = apply_turn(new_profile(), "වේගය වෙනස් කරන්න", "Change the speed", "CHANGE_SPEED", NOW)
    assert d["ask"]["question"] == SPEED_QUESTION_SI
    p, d = apply_speed_answer(p, "හෙමින්", now=NOW)
    assert d["speed"] == "slower" and d["channels"] == ["delivery"]
    assert d["changes"] == {"rate": [1.0, 0.9]} and p["counts"]["slower"] == 1
    assert d["ask"] is None and "speed question" in d["reasons"][-1]


def test_speed_question_then_faster_reply_sets_the_rate():
    p, _ = apply_turn(new_profile(), "වේගය වෙනස් කරන්න", "", "CHANGE_SPEED", NOW)
    p, d = apply_speed_answer(p, "ඉක්මනින්", now=NOW)
    assert d["speed"] == "faster" and p["rate"] == 1.1


def test_speed_question_then_other_reply_is_a_new_request():
    p, _ = apply_turn(new_profile(), "වේගය වෙනස් කරන්න", "", "CHANGE_SPEED", NOW)
    p2, d = apply_speed_answer(p, "ඊළඟ පිටුවට යන්න", now=NOW)
    assert d is None and p2 is p and p["rate"] == 1.0


def test_repeat_after_the_speed_question_is_not_the_next_turn():
    # answer -> "change the speed" -> question -> "ආයෙ": two turns after the answer
    p, _ = apply_turn(spoken(5), "වේගය වෙනස් කරන්න", "", "CHANGE_SPEED", NOW)
    p, d = apply_turn(p, "ආයෙ", "", "UNKNOWN", NOW)
    assert d["repeat"] and d["replay"]["text"] == ANSWER and d["changes"] == {}
