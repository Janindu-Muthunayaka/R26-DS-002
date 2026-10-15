# Tests for delivery/turn.py: one user turn from Sinhala text to what is
# spoken. The intent pipeline is a fake (no NLLB / Llama), the player is
# laptop_player.play_plan_silent (no audio), and the database is a temp file.

from datetime import datetime, timedelta

import pytest

from delivery.laptop_player import play_plan_silent
from delivery.mock_c3 import MockComponent3
from delivery.turn import (
    CHECKIN_NO_SI, CONTINUE_QUESTION_SI, NOTHING_TO_REPEAT_SI, SPEED_OK_SI, TurnRunner,
)
from delivery.voice_profile import TABLE, LOG_TABLE, _db_cache, get_profile
from intent_detection.modifiers import build_frame

START = datetime(2026, 10, 15, 10, 0, 0)

# sinhala -> (english, LLM intent, route, style or command), as the real
# pipeline would return them.
FAKE_PIPELINE = {
    "මේකේ මොකක්ද තියෙන්නේ": ("What's in this?", "IDENTIFY_CONTENT", "GENERATE", "Detailed"),
    "කෙටියෙන් කියන්න": ("Tell me briefly", "SUMMARIZE", "GENERATE", "Simple"),
    "පියවරෙන් පියවර කියන්න": ("Tell me step by step", "STEP_BY_STEP", "GENERATE", "StepByStep"),
    "සරලව හෙමින් කියන්න": ("Say it simply and slowly", "SIMPLIFY", "GENERATE", "Simple"),
    "වේගය වෙනස් කරන්න": ("Change the speed", "CHANGE_SPEED", "GENERATE", "Detailed"),
    "මට මේ ලිපිය ටිකක් හෙමින් කියවන්න": ("Read this article to me a little slowly",
                                          "READ_ALOUD", "GENERATE", "Detailed"),
    "ඊළඟ ලිපියට යන්න": ("Go to the next article", "NEXT", "SYSTEM_COMMAND", "GO_FORWARD"),
    "කලින් ලිපියට යන්න": ("Go to the previous article", "PREVIOUS", "SYSTEM_COMMAND", "GO_BACK"),
}


def fake_handle_command(calls):
    def handle(sinhala, user_id):
        calls.append(sinhala)
        english, intent, route, extra = FAKE_PIPELINE[sinhala]
        if route == "SYSTEM_COMMAND":
            final = {"route": route, "command": extra, "intent": intent, "english_translation": english}
        else:
            final = {"route": route, "intent": intent, "english_translation": english,
                     "style_class": extra, "prompt_modifier": "...",
                     "personalization_flags": {}, "correction_applied": None}
        return {"final_prompt": final, "frame": build_frame(sinhala, english, intent)}
    return handle


class FakePlayer:
    """play_plan_silent, recording each call; stop_at = a user stop (Enter)
    during that segment index on the next answer played."""

    def __init__(self):
        self.calls, self.stop_at = [], None

    def __call__(self, plan, start=0, end=None, on_segment=None):
        texts = [s["text"] for s in plan["segments"]]
        self.calls.append({"texts": texts, "start": start, "end": end, "rate": plan["rate"]})
        if self.stop_at is not None and len(texts) > 1:
            i, self.stop_at = self.stop_at, None
            r = play_plan_silent(plan, start=start, end=i)
            r.update(completed=False, stopped_at=i, stopped_in="segment", segments_played=i)
            return r
        return play_plan_silent(plan, start=start, end=end, on_segment=on_segment)

    def answers(self):
        """Calls that played an answer (system messages are one segment)."""
        return [c for c in self.calls if len(c["texts"]) > 1]


class Clock:
    def __init__(self):
        self.t = START

    def __call__(self):
        self.t += timedelta(seconds=5)
        return self.t


@pytest.fixture
def run(tmp_path):
    calls, player = [], FakePlayer()
    db = str(tmp_path / "db.json")
    runner = TurnRunner("tester", fake_handle_command(calls), MockComponent3(), player,
                        now=Clock(), db_path=db)
    runner.calls, runner.player, runner.db = calls, player, db
    yield runner
    _db_cache.pop(db).close()


C3 = MockComponent3()
FUEL = C3.articles[0]["answers"]       # Detailed: 6 segments, check-in after 3


# ── Content route ──────────────────────────────────────────────────────────

def test_question_plays_a_new_answer_and_saves_last_spoken(run):
    rec = run.turn("කෙටියෙන් කියන්න")
    assert rec["route"] == "content" and rec["action"] == "answer"
    assert run.calls == ["කෙටියෙන් කියන්න"]
    assert rec["style"] == "Simple" and rec["completed"] and rec["rate_used"] == 1.0
    last = run.profile["last_spoken"]
    assert last["text"] == FUEL["Simple"] and last["completed"] and last["turns_since"] == 0


def test_long_answer_pauses_at_the_check_in_and_yes_plays_the_rest(run):
    rec = run.turn("මේකේ මොකක්ද තියෙන්නේ")
    assert rec["stopped_in"] == "checkpoint" and rec["said"] == [CONTINUE_QUESTION_SI]
    assert run.player.answers()[0]["end"] == 3
    rec = run.turn("ඔව්")
    assert rec["action"] == "checkin_yes" and rec["completed"] and rec["start"] == 3
    assert run.calls == ["මේකේ මොකක්ද තියෙන්නේ"]           # "ඔව්" never reaches the pipeline
    assert run.profile["last_spoken"]["completed"]


def test_check_in_no_stops_without_learning(run):
    run.turn("මේකේ මොකක්ද තියෙන්නේ")
    rec = run.turn("එපා")
    assert rec["action"] == "checkin_no" and rec["said"] == [CHECKIN_NO_SI]
    assert len(run.player.answers()) == 1 and rec["changes"] == {}
    assert run.profile["counts"]["stopped"] == 0
    assert run.profile["last_spoken"]["resume_at"] == 3


def test_check_in_dropped_by_a_new_request(run):
    run.turn("මේකේ මොකක්ද තියෙන්නේ")
    rec = run.turn("කෙටියෙන් කියන්න")
    assert rec["route"] == "content" and rec["style"] == "Simple"
    assert "check-in dropped" in rec["reasons"][0]


# ── REPEAT ─────────────────────────────────────────────────────────────────

def test_repeat_on_the_very_next_turn_replays_whole_answer_and_learns(run):
    run.turn("කෙටියෙන් කියන්න")
    rec = run.turn("ආයෙ කියන්න")
    assert rec["fast_path"] and len(run.calls) == 1        # no NLLB + Llama call
    assert rec["route"] == "delivery" and rec["action"] == "replay"
    assert run.player.answers()[-1]["texts"] == run.player.answers()[0]["texts"]
    assert rec["changes"]["rate"] == [1.0, 0.95] and rec["rate_used"] == 0.95


def test_repeat_on_a_later_turn_replays_without_learning(run):
    run.turn("කෙටියෙන් කියන්න")
    run.turn("ඊළඟ ලිපියට යන්න")
    rec = run.turn("ආයෙ කියන්න")
    assert rec["action"] == "replay" and rec["changes"] == {} and rec["rate_used"] == 1.0
    assert run.player.answers()[-1]["texts"] == run.player.answers()[0]["texts"]


def test_repeat_with_nothing_spoken_says_so(run):
    rec = run.turn("ආයෙ කියන්න")
    assert rec["action"] == "nothing_to_repeat" and rec["said"] == [NOTHING_TO_REPEAT_SI]


def test_repeat_after_30_minutes_says_nothing_to_repeat(run):
    run.turn("කෙටියෙන් කියන්න")
    run.now.t += timedelta(minutes=31)
    rec = run.turn("ආයෙ කියන්න")
    assert rec["action"] == "nothing_to_repeat"


# ── Slower / faster (decision 15 Oct) ─────────────────────────────────────

def test_slower_after_a_finished_answer_confirms_and_does_not_replay(run):
    run.turn("කෙටියෙන් කියන්න")
    rec = run.turn("හෙමින් කියන්න")
    assert rec["fast_path"] and rec["route"] == "delivery" and rec["action"] == "confirm"
    assert rec["said"] == [SPEED_OK_SI["slower"]] and len(run.player.answers()) == 1
    assert rec["changes"]["rate"] == [1.0, 0.9]
    run.turn("කෙටියෙන් කියන්න")
    assert run.player.answers()[-1]["rate"] == 0.9            # new speed from the next answer


def test_faster_confirmation(run):
    run.turn("කෙටියෙන් කියන්න")
    rec = run.turn("ඉක්මනින් කියන්න")
    assert rec["said"] == [SPEED_OK_SI["faster"]] and run.profile["rate"] == 1.1


def test_slower_after_a_stopped_answer_continues_from_the_stopped_segment(run):
    run.player.stop_at = 1
    rec = run.turn("පියවරෙන් පියවර කියන්න")
    assert rec["stopped_in"] == "segment" and run.profile["last_spoken"]["resume_at"] == 1
    rec = run.turn("හෙමින්")
    assert rec["action"] == "resume" and rec["start"] == 1
    assert rec["said"] == [CONTINUE_QUESTION_SI]              # no confirmation; check-in at 3
    last_call = run.player.answers()[-1]
    assert last_call["start"] == 1 and last_call["rate"] == 0.9
    assert last_call["texts"] == run.player.answers()[0]["texts"]


def test_slower_at_a_check_in_continues_from_the_check_in(run):
    run.turn("මේකේ මොකක්ද තියෙන්නේ")
    rec = run.turn("හෙමින් කියන්න")                         # not yes/no: check-in dropped
    assert rec["action"] == "resume" and rec["start"] == 3 and rec["rate_used"] == 0.9


def test_again_and_slower_replays_the_whole_answer_at_the_new_speed(run):
    run.player.stop_at = 1
    run.turn("පියවරෙන් පියවර කියන්න")
    rec = run.turn("ආයෙත් හෙමින් කියන්න")
    assert rec["action"] == "replay" and rec["start"] == 0 and rec["rate_used"] == 0.9


def test_speed_word_with_read_aloud_is_delivery_only(run):
    run.turn("කෙටියෙන් කියන්න")
    rec = run.turn("මට මේ ලිපිය ටිකක් හෙමින් කියවන්න")
    assert not rec["fast_path"] and rec["route"] == "delivery" and rec["action"] == "confirm"


def test_style_and_speed_together_is_both(run):
    rec = run.turn("සරලව හෙමින් කියන්න")
    assert not rec["fast_path"]                               # style word: needs the pipeline
    assert rec["route"] == "both" and rec["action"] == "answer"
    assert rec["style"] == "Simple" and rec["rate_used"] == 0.9 and rec["said"] == []


def test_speed_question_then_answer(run):
    run.turn("කෙටියෙන් කියන්න")
    rec = run.turn("වේගය වෙනස් කරන්න")
    assert rec["action"] == "ask_speed" and rec["said"] == ["හෙමින්ද ඉක්මනින්ද?"]
    assert run.profile["rate"] == 1.0
    rec = run.turn("ඉක්මනින්")
    assert rec["action"] == "confirm" and run.profile["rate"] == 1.1
    assert "speed question" in " ".join(rec["reasons"])


# ── System commands ────────────────────────────────────────────────────────

def test_next_and_previous_article_change_the_mock_article(run):
    rec = run.turn("ඊළඟ ලිපියට යන්න")
    assert rec["route"] == "system" and rec["action"] == "GO_FORWARD"
    assert rec["said"] == [f"ඊළඟ ලිපිය. {C3.articles[1]['title']}"]
    rec = run.turn("කෙටියෙන් කියන්න")
    assert rec["article_id"] == C3.articles[1]["id"]
    run.turn("කලින් ලිපියට යන්න")
    assert run.turn("කෙටියෙන් කියන්න")["article_id"] == C3.articles[0]["id"]


def test_system_messages_are_not_saved_as_last_spoken(run):
    run.turn("කෙටියෙන් කියන්න")
    run.turn("ඊළඟ ලිපියට යන්න")
    assert run.profile["last_spoken"]["text"] == FUEL["Simple"]


# ── Feedback after each answer, storage, errors ────────────────────────────

def test_three_clean_answers_drift_the_rate_up(run):
    for _ in range(3):
        rec = run.turn("කෙටියෙන් කියන්න")
    assert rec["changes"]["rate"] == [1.0, 1.02]


def test_profile_and_delivery_log_are_saved(run):
    run.turn("කෙටියෙන් කියන්න")
    run.turn("හෙමින් කියන්න")
    assert get_profile("tester", run.db)["rate"] == 0.9
    log = _db_cache[run.db].table(LOG_TABLE).all()
    assert [r["action"] for r in log] == ["answer", "confirm"]
    assert log[0]["rate_used"] == 1.0 and log[0]["completed"] is True
    assert log[1]["profile_after"]["rate"] == 0.9
    assert len(_db_cache[run.db].table(TABLE).all()) == 1


def test_a_playback_error_is_logged_and_the_turn_still_finishes(run):
    def broken(plan, **kwargs):
        raise ConnectionError("no internet")
    run.play = broken
    rec = run.turn("කෙටියෙන් කියන්න")
    assert rec["errors"] == ["ConnectionError: no internet"]
    assert run.profile["last_spoken"] is None                 # nothing was spoken
