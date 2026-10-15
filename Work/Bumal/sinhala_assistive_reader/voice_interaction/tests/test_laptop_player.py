# Tests for delivery/laptop_player.py: which segments play_plan() plays for a
# start/end range and what it reports. Synthesis, ffplay and the pauses are
# replaced by fakes, so no audio, network or FFmpeg is needed.

import pytest

from delivery import laptop_player
from delivery.laptop_player import play_plan, play_plan_silent
from delivery.planner import plan_speech

ANSWER = ("ලිපිය ගැන කියන්නම්. පළමුව එක. දෙවනුව දෙක. "
          "තෙවනුව තුන. සිව්වනුව හතර.")


@pytest.fixture
def fake_audio(monkeypatch):
    """Records the texts 'played'. stop_on = text whose playback the user stops."""
    state = {"played": [], "stop_on": None}
    monkeypatch.setattr(laptop_player, "synthesize", lambda text: text)

    def play_file(ffplay, path, rate):
        state["played"].append(path)
        return path == state["stop_on"]

    monkeypatch.setattr(laptop_player, "_play_file", play_file)
    monkeypatch.setattr(laptop_player, "_wait", lambda ms: False)
    return state


def plan(chunk_size=6):
    return plan_speech(ANSWER, "StepByStep", {"chunk_size": chunk_size})


def test_whole_plan_plays_every_segment(fake_audio):
    p = plan()
    r = play_plan(p, ffplay="ffplay")
    assert r["completed"] and r["segments_played"] == 5 and r["stopped_in"] is None
    assert fake_audio["played"] == [s["text"] for s in p["segments"]]


def test_end_stops_at_the_checkpoint_without_a_user_stop(fake_audio):
    p = plan(chunk_size=2)
    assert p["checkpoints"] == [2, 4] and p["preview"]
    r = play_plan(p, ffplay="ffplay", end=2)
    assert not r["completed"] and r["stopped_in"] == "checkpoint" and r["stopped_at"] is None
    assert r["segments_played"] == 2
    assert fake_audio["played"] == [p["preview"]] + [s["text"] for s in p["segments"][:2]]


def test_start_continues_without_the_preview(fake_audio):
    p = plan(chunk_size=2)
    r = play_plan(p, ffplay="ffplay", start=2, end=4)
    assert fake_audio["played"] == [s["text"] for s in p["segments"][2:4]]
    assert r["start"] == 2 and r["segments_played"] == 4 and r["stopped_in"] == "checkpoint"
    r = play_plan(p, ffplay="ffplay", start=4)
    assert r["completed"] and r["segments_played"] == 5


def test_user_stop_reports_the_absolute_segment_index(fake_audio):
    p = plan()
    fake_audio["stop_on"] = p["segments"][3]["text"]
    r = play_plan(p, ffplay="ffplay", start=2)
    assert r["stopped_in"] == "segment" and r["stopped_at"] == 3
    assert r["segments_played"] == 3          # resume from segment 3
    assert r["finished_at"]


def test_silent_player_gives_the_same_result_shape(fake_audio):
    p = plan(chunk_size=2)
    for start, end in [(0, None), (0, 2), (2, 4), (4, None)]:
        loud = play_plan(p, ffplay="ffplay", start=start, end=end)
        quiet = play_plan_silent(p, start=start, end=end)
        loud.pop("finished_at"), quiet.pop("finished_at")
        assert loud == quiet
