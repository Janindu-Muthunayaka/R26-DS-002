# Tests for delivery/voice_profile.py. Uses a temporary TinyDB file, never data/db.json.

from datetime import datetime, timedelta

from delivery.voice_profile import (
    DEFAULTS, count_turn, get_profile, is_next_turn, save_profile, set_last_spoken,
    repeatable, new_profile,
)

NOW = datetime(2026, 10, 13, 12, 0, 0)


def test_new_user_gets_defaults(tmp_path):
    db = str(tmp_path / "db.json")
    profile = get_profile("nobody", db_path=db)
    assert profile["rate"] == DEFAULTS["rate"]
    assert profile["chunk_size"] == DEFAULTS["chunk_size"]
    assert profile["last_spoken"] is None


def test_profile_round_trips_and_users_are_separate(tmp_path):
    db = str(tmp_path / "db.json")
    p = get_profile("user_a", db_path=db)
    p["rate"], p["counts"]["repeat"] = 0.85, 2
    set_last_spoken(p, "මෙය පරීක්ෂණයකි.", "answer", "Simple", NOW)
    save_profile("user_a", p, db_path=db)

    again = get_profile("user_a", db_path=db)
    assert again["rate"] == 0.85
    assert again["counts"]["repeat"] == 2
    assert again["last_spoken"]["text"] == "මෙය පරීක්ෂණයකි."
    assert again["last_spoken"]["finished_at"] == "2026-10-13T12:00:00"
    assert get_profile("user_b", db_path=db)["rate"] == 1.0


def test_default_profiles_do_not_share_mutable_state():
    a, b = new_profile(), new_profile()
    a["counts"]["repeat"] += 1
    a["long_history"].append(True)
    assert b["counts"]["repeat"] == 0 and b["long_history"] == []


def test_last_spoken_older_than_30_minutes_is_nothing_to_repeat():
    p = set_last_spoken(new_profile(), "text", "answer", "Detailed", NOW)
    assert repeatable(p, NOW + timedelta(minutes=29))["text"] == "text"
    assert repeatable(p, NOW + timedelta(minutes=31)) is None


def test_nothing_spoken_yet_is_nothing_to_repeat():
    assert repeatable(new_profile(), NOW) is None


def test_next_turn_is_only_the_first_turn_after_speaking():
    p = set_last_spoken(new_profile(), "text", "answer", "Simple", NOW)
    assert is_next_turn(p)
    count_turn(p)
    assert not is_next_turn(p) and p["last_spoken"]["turns_since"] == 1
    count_turn(p)
    assert p["last_spoken"]["turns_since"] == 2
    assert not is_next_turn(count_turn(new_profile()))     # nothing spoken: no-op


def test_old_profiles_without_turns_since_are_never_the_next_turn():
    p = new_profile()
    p["last_spoken"] = {"text": "t", "kind": "answer", "style": "Simple",
                        "finished_at": "2026-10-13T12:00:00"}
    assert not is_next_turn(p)


def test_stopped_answer_keeps_where_to_resume():
    p = set_last_spoken(new_profile(), "t", "answer", "Simple", NOW, completed=False, resume_at=2)
    assert p["last_spoken"]["completed"] is False and p["last_spoken"]["resume_at"] == 2
    p = set_last_spoken(p, "t", "answer", "Simple", NOW, completed=True, resume_at=2)
    assert p["last_spoken"]["resume_at"] is None            # finished: nothing to resume
