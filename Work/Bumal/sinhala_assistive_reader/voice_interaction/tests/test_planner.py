# Tests for delivery/planner.py. Run from voice_interaction/:
#   venv\Scripts\python.exe -m pytest tests -q

import json

from delivery.planner import plan_speech, split_segments, DEFAULT_PROFILE

STEPS = ("මෙම ලිපියේ ප්‍රධාන කරුණු තුනක් තියෙනවා. "
         "පළමුව, ඉන්ධන මිල රුපියල් 25.50 කින් අඩු කළා. "
         "දෙවනුව, නව මිල අද රාත්‍රියේ සිට ක්‍රියාත්මක වෙනවා. "
         "තෙවනුව, බස් ගාස්තු අඩු කිරීමට සාකච්ඡා කෙරෙනවා.")


# ── Segmenting ──────────────────────────────────────────

def test_splits_on_full_stop_question_and_exclamation():
    assert split_segments("එක. දෙක? තුන! හතර") == ["එක.", "දෙක?", "තුන!", "හතර"]


def test_never_splits_inside_a_number():
    segs = split_segments("මිල රුපියල් 25.50 කින් අඩු වුණා. ඊළඟ කරුණ.")
    assert segs == ["මිල රුපියල් 25.50 කින් අඩු වුණා.", "ඊළඟ කරුණ."]


def test_never_splits_after_an_abbreviation():
    segs = split_segments("රැස්වීම 4.30 ප.ව. ආරම්භ වෙනවා. සියලු දෙනා එන්න.")
    assert segs == ["රැස්වීම 4.30 ප.ව. ආරම්භ වෙනවා.", "සියලු දෙනා එන්න."]


def test_splits_before_spoken_ordinals_without_full_stops():
    segs = split_segments("පියවර දෙකයි පළමුව යතුර ඔබන්න දෙවනුව දොර අරින්න")
    assert segs == ["පියවර දෙකයි", "පළමුව යතුර ඔබන්න", "දෙවනුව දොර අරින්න"]


def test_ordinal_answer_gives_one_segment_per_point():
    segs = split_segments(STEPS)
    assert len(segs) == 4
    assert segs[1].startswith("පළමුව") and "25.50" in segs[1]
    assert segs[2].startswith("දෙවනුව")
    assert segs[3].startswith("තෙවනුව")


def test_empty_or_blank_text_gives_no_segments():
    assert split_segments("") == []
    assert split_segments("   ") == []
    assert split_segments(None) == []
    assert plan_speech("", "Simple")["segments"] == []


# ── Pauses ──────────────────────────────────────────────

def test_pause_depends_on_style_and_last_segment_has_none():
    for style, ms in [("Simple", 400), ("Detailed", 600), ("StepByStep", 800)]:
        pauses = [s["pause_after_ms"] for s in plan_speech(STEPS, style)["segments"]]
        assert pauses == [ms, ms, ms, 0]


def test_pause_scale_multiplies_the_base_pause():
    plan = plan_speech(STEPS, "Detailed", {"pause_scale": 1.5})
    assert plan["segments"][0]["pause_after_ms"] == 900


def test_unknown_style_uses_fallback_pause():
    assert plan_speech("එක. දෙක.", "Weird")["segments"][0]["pause_after_ms"] == 600


# ── Check-ins and preview ───────────────────────────────

def test_checkpoints_every_chunk_never_after_last_segment():
    text = "එක. දෙක. තුන. හතර. පහ."
    assert plan_speech(text, "Simple", {"chunk_size": 2})["checkpoints"] == [2, 4]
    assert plan_speech(text, "Simple", {"chunk_size": 5})["checkpoints"] == []
    assert plan_speech(text, "Simple", {"chunk_size": 1})["checkpoints"] == [1, 2, 3, 4]


# Same points as STEPS, but the intro sentence does not announce them.
STEPS_NO_ANNOUNCE = STEPS.replace("මෙම ලිපියේ ප්‍රධාන කරුණු තුනක් තියෙනවා.", "මෙන්න විස්තරය.")


def test_preview_only_when_more_segments_than_chunk_size():
    assert plan_speech(STEPS_NO_ANNOUNCE, "StepByStep", {"chunk_size": 3})["preview"] is not None
    assert plan_speech(STEPS_NO_ANNOUNCE, "StepByStep", {"chunk_size": 4})["preview"] is None


def test_preview_counts_numbered_points_not_the_intro_sentence():
    # intro sentence + 3 ordinal points -> "3 points", not 4
    plan = plan_speech(STEPS_NO_ANNOUNCE, "StepByStep", {"chunk_size": 2})
    assert plan["preview"] == "කරුණු 3ක් තියෙනවා"


def test_no_preview_when_first_segment_already_announces_points():
    # STEPS opens with "ප්‍රධාන කරුණු තුනක් තියෙනවා" — a preview would repeat it
    assert plan_speech(STEPS, "StepByStep", {"chunk_size": 2})["preview"] is None


def test_preview_counts_parts_when_there_are_no_ordinals():
    text = "එක. දෙක. තුන. හතර. පහ."
    assert plan_speech(text, "Detailed", {"chunk_size": 3})["preview"] == "කොටස් 5ක් තියෙනවා"


# ── Rate, pitch, defaults, contract ─────────────────────

def test_rate_is_clamped_and_pitch_fixed():
    assert plan_speech(STEPS, "Simple", {"rate": 0.2})["rate"] == 0.6
    assert plan_speech(STEPS, "Simple", {"rate": 3.0})["rate"] == 1.8
    assert plan_speech(STEPS, "Simple", {"rate": 0.85})["rate"] == 0.85
    assert plan_speech(STEPS, "Simple", {"rate": 3.0})["pitch"] == 1.0


def test_chunk_size_is_clamped():
    text = "එක. දෙක. තුන. හතර. පහ. හය. හත. අට."
    assert plan_speech(text, "Simple", {"chunk_size": 0})["checkpoints"][:2] == [1, 2]
    assert plan_speech(text, "Simple", {"chunk_size": 20})["checkpoints"] == [6]


def test_missing_profile_uses_defaults():
    plan = plan_speech(STEPS, "Detailed")
    assert plan["rate"] == DEFAULT_PROFILE["rate"]
    assert plan["checkpoints"] == [3]
    assert plan["replay"] is False


def test_plan_matches_contract_and_is_json_serializable():
    plan = plan_speech(STEPS, "StepByStep", {"rate": 0.85}, replay=True)
    assert set(plan) == {"rate", "pitch", "preview", "segments", "checkpoints", "replay"}
    assert all(set(s) == {"text", "pause_after_ms"} for s in plan["segments"])
    assert plan["replay"] is True
    assert json.loads(json.dumps(plan, ensure_ascii=False)) == plan
