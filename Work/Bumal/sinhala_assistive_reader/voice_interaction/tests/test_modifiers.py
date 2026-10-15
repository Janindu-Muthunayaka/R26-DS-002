# Tests for intent_detection/modifiers.py. Run from voice_interaction/:
#   venv\Scripts\python.exe -m pytest tests -q
#
# Part 1: keyword checks driven by data/test_samples.py (45 commands, with the
#         real NLLB English and Approach 1 intent saved in
#         intent_detection/intent_detection_results_all.json — broken
#         translations included) and data/test_samples_multi.py (16 commands).
#         (Moved here from tests/test_feedback.py with the checks themselves.)
# Part 2: style modifier.
# Part 3: build_frame() and the fusion rule.
# Part 4: modifiers that come only from the LLM intent (unmatched_modifiers log).
#
# data/test_samples_fresh.py is the held-out set: it is NOT used here.

import ast
import json
import os

import pytest

from data.test_samples import test_samples
from data.test_samples_multi import EXISTING_MODIFIER_LABELS, NONE, multi_samples
from intent_detection.modifiers import (
    build_frame, fuse_intent, is_repeat, speed_request, style_from_keywords, style_modifier,
)
from personalization import diagnostic
from personalization.logger import UNMATCHED_TABLE, log_unmatched_modifiers

# ── Part 1: keyword checks on the test sets ────────────────────────────────

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "intent_detection", "intent_detection_results_all.json")
with open(_RESULTS, encoding="utf-8") as _f:
    _NLLB = {r["sinhala_input"]: r for r in json.load(_f)["detailed_results"]["a1"]}

CASES_45 = [
    pytest.param(s["stt_output"], _NLLB[s["stt_output"]]["english_translation"],
                 _NLLB[s["stt_output"]]["intent"], EXISTING_MODIFIER_LABELS.get(s["id"], NONE),
                 id=s["id"])
    for s in test_samples
]
CASES_16 = [pytest.param(s["stt_output"], "", None, s["expected_modifiers"], id=s["id"])
            for s in multi_samples]


def test_every_one_of_the_45_has_saved_nllb_output():
    assert all(s["stt_output"] in _NLLB for s in test_samples)


@pytest.mark.parametrize("sinhala, english, intent, labels", CASES_45 + CASES_16)
def test_is_repeat_matches_labels(sinhala, english, intent, labels):
    assert is_repeat(sinhala, english, intent) == labels["repeat"]


@pytest.mark.parametrize("sinhala, english, intent, labels", CASES_45 + CASES_16)
def test_speed_request_matches_labels(sinhala, english, intent, labels):
    assert speed_request(sinhala, english) == labels["speed"]


def test_sinhala_wins_where_nllb_dropped_the_word():
    # ආයෙ -> "I'm going back." and Llama said PREVIOUS, yet it is a REPEAT
    assert is_repeat("ආයෙ", "I'm going back.", "PREVIOUS")
    # "briefly, fast" -> NLLB lost "fast" entirely
    assert speed_request("කෙටියෙන් වේගයෙන් පහදන්න", "I want to get this down without any details .") == "faster"


def test_too_fast_and_too_slow_mean_the_opposite_word():
    assert speed_request("ඔයා කියවන්නේ ගොඩක් වේගෙන්", "") == "slower"
    assert speed_request("වේගය වැඩියි", "") == "slower"
    assert speed_request("ගොඩක් වේගෙන් කියවන්න", "") == "faster"   # "read very fast" is a real request
    assert speed_request("", "You are reading too fast") == "slower"
    assert speed_request("", "It's too slow") == "faster"


def test_again_next_to_a_change_or_style_word_is_not_repeat():
    assert not is_repeat("මේ වාක්‍යය ටිකක් වෙනස් කරලා ආයෙ කියන්න", "", None)      # sam_24
    assert not is_repeat("සරල වචන වලින් ආයෙත් පැහැදිලි කරන්න", "", None)          # m_13
    assert not is_repeat("", "Change the sentence a little and say it again .", None)


def test_navatha_stopped_is_not_navatha_again():
    assert not is_repeat("කියවීම නැවතී තියෙනවා", "", None)


# ආපසු / ආපහු + කියන්න / කියවන්න = repeat (Bumal, 14 Oct). The first three are
# the fresh-set sentences f_03, f_04, f_12 that the first run missed.
@pytest.mark.parametrize("sinhala", [
    "ආපසු කියන්න",
    "සමාවෙන්න ආපහු කියන්න පුළුවන්ද",
    "ආපසු කියන්න පොඩ්ඩක් හෙමින්",
    "ආපහු කියවන්න",
    "ඒ කොටස ආපසු කියවන්න",
])
def test_apasu_with_say_or_read_is_repeat(sinhala):
    assert is_repeat(sinhala, "", None)
    assert fuse_intent(sinhala, "PREVIOUS") == ("REPEAT", "Sinhala repeat word overrides LLM PREVIOUS")


@pytest.mark.parametrize("sinhala", [
    "ආපසු යන්න",
    "ආපහු යන්න",
    "කලින් පිටුවට ආපසු යන්න",
    "ආපසු පිටුවට",
    "ආපසු පිටුව කියවන්න",
    "ආපහු කියන්න පිටුවට යන්න",
])
def test_apasu_with_go_or_page_is_not_repeat(sinhala):
    assert not is_repeat(sinhala, "", "PREVIOUS")
    assert fuse_intent(sinhala, "PREVIOUS") == ("PREVIOUS", None)


def test_apasu_go_back_sentences_from_training_data_are_not_repeat():
    # Every "go back" sentence with ආපසු / ආපහු in data/generate_samples.py
    path = os.path.join(os.path.dirname(_RESULTS), os.pardir, "data", "generate_samples.py")
    with open(path, encoding="utf-8") as f:
        samples = ast.literal_eval(f.read())
    back = [s["stt_output"] for s in samples if "ආපසු" in s["stt_output"] or "ආපහු" in s["stt_output"]]
    assert len(back) >= 10
    assert not [si for si in back if is_repeat(si, "", None)]


# ── Part 2: style modifier ─────────────────────────────────────────────────

# sam_2 "තව ටිකක් විස්තර කරන්න" (Detailed) and sam_16 "ලේසියෙන් කියන්න"
# (Simple) were known failures until Bumal corrected their labels (14 Oct).
@pytest.mark.parametrize("sinhala, english, intent, labels", CASES_45 + CASES_16)
def test_style_modifier_matches_labels(sinhala, english, intent, labels):
    frame = build_frame(sinhala, english, intent)
    assert frame["modifiers"]["style"] == labels["style"]


def test_style_from_sinhala_words():
    assert style_modifier("කෙටියෙන් කියන්න කවුද දිනුවේ", "") == "Simple"            # m_12
    assert style_modifier("පියවරෙන් පියවර හෙමින් පැහැදිලි කරන්න", "") == "StepByStep"  # m_15
    assert style_modifier("දිගු විස්තර නොමැතිව කෙටියෙන් පහදන්න", "") == "Simple"      # without details
    assert style_modifier("ලිපිය කියවන්න", "Read it in full detail") is None          # Sinhala decides


def test_style_uses_english_only_without_sinhala():
    assert style_modifier("", "Explain it step by step") == "StepByStep"
    assert style_modifier("", "") is None


def test_repeat_frame_has_no_style():
    # "one more time" -> English "more" would say Detailed; a repeat has no style
    assert build_frame("තව පාරක් කියන්න", "Tell me one more time", None)["modifiers"]["style"] is None


def test_diagnostic_still_uses_the_same_style_keywords():
    # style_from_keywords moved here; the style model's import must be the same function
    assert diagnostic.style_from_keywords is style_from_keywords
    assert diagnostic.style_from_keywords("Give me the full details") == "Detailed"


# ── Part 3: build_frame and the fusion rule ────────────────────────────────

def test_frame_shape():
    f = build_frame("ආයෙත් හෙමින් කියන්න", "Say it again slowly", "REPEAT")
    assert f["intent"] == "REPEAT"
    assert f["modifiers"] == {"repeat": True, "speed": "slower", "style": None}
    assert f["evidence"]["intent_from"] == "llm" and f["evidence"]["fusion"] is None
    assert f["evidence"]["repeat_from"] == "sinhala" and f["evidence"]["speed_from"] == "sinhala"


@pytest.mark.parametrize("llm_intent", ["PREVIOUS", "NEXT", "UNKNOWN", None, "previous"])
def test_fusion_sinhala_repeat_overrides_previous_next_unknown(llm_intent):
    f = build_frame("ආයෙ", "I'm going back.", llm_intent)
    assert f["intent"] == "REPEAT" and f["modifiers"]["repeat"]
    assert f["evidence"]["intent_from"] == "fusion"
    assert f["evidence"]["llm_intent"] == str(llm_intent or "UNKNOWN").upper()
    assert "overrides" in f["evidence"]["fusion"]


@pytest.mark.parametrize("llm_intent", ["STOP", "READ_ALOUD", "REPHRASE", "SIMPLIFY", "CHANGE_SPEED"])
def test_fusion_leaves_other_labels_alone(llm_intent):
    f = build_frame("ආයෙ", "Again", llm_intent)
    assert f["intent"] == llm_intent and f["evidence"]["intent_from"] == "llm"


def test_fusion_needs_the_sinhala_word():
    # English "again" alone does not override; a style word blocks the Sinhala repeat
    assert fuse_intent("", "PREVIOUS") == ("PREVIOUS", None)
    assert fuse_intent("සරල වචන වලින් ආයෙත් පැහැදිලි කරන්න", "UNKNOWN")[0] == "UNKNOWN"   # m_13
    assert fuse_intent("ඊළඟ පිටුවට යන්න", "NEXT")[0] == "NEXT"


def test_fusion_on_the_saved_45():
    # With the saved thesis-run output, fusion fixes exactly sam_31 and sam_33
    # (both PREVIOUS) and changes nothing else.
    changed = {}
    for s in test_samples:
        r = _NLLB[s["stt_output"]]
        f = build_frame(s["stt_output"], r["english_translation"], r["intent"])
        if f["intent"] != r["intent"]:
            changed[s["id"]] = (r["intent"], f["intent"], s["expected_intent"])
    assert changed == {"sam_31": ("PREVIOUS", "REPEAT", "REPEAT"),
                       "sam_33": ("PREVIOUS", "REPEAT", "REPEAT")}


# ── Part 4: modifiers that come only from the LLM intent ───────────────────

@pytest.mark.parametrize("sinhala, english, intent, unmatched", [
    ("මට තේරුනේ නෑ", "I didn't understand", "REPEAT", ["repeat"]),     # repeat only from the label
    ("වේගය වෙනස් කරන්න", "Change the speed", "CHANGE_SPEED", ["speed"]),  # no slower/faster word
    ("මේක තේරෙන්නේ නෑ", "I don't understand this", "SIMPLIFY", ["style"]),  # no style word
    ("ආයෙත් හෙමින් කියන්න", "Say it again slowly", "REPEAT", []),      # words found everything
    ("හෙමින් කියන්න", "Say it slowly", "CHANGE_SPEED", []),
    ("සරල වචන වලින් කියන්න", "Say it in simple words", "SIMPLIFY", []),
    ("ආයෙ", "I'm going back.", "PREVIOUS", []),                         # fusion = a Sinhala word
    ("ලිපිය කියවන්න", "Read the article", "READ_ALOUD", []),            # label implies no modifier
])
def test_unmatched_lists_modifiers_only_the_label_gave(sinhala, english, intent, unmatched):
    assert build_frame(sinhala, english, intent)["evidence"]["unmatched"] == unmatched


def test_the_saved_45_have_no_unmatched_modifiers():
    for s in test_samples:
        r = _NLLB[s["stt_output"]]
        assert build_frame(s["stt_output"], r["english_translation"], r["intent"])["evidence"]["unmatched"] == [], s["id"]


def test_unmatched_sentence_and_frame_are_saved(tmp_path):
    db = str(tmp_path / "db.json")
    frame = build_frame("වේගය වෙනස් කරන්න", "Change the speed", "CHANGE_SPEED")
    rec = log_unmatched_modifiers("u1", "වේගය වෙනස් කරන්න", "Change the speed", frame, db_path=db)
    assert rec["unmatched"] == ["speed"] and rec["frame"] == frame
    with open(db, encoding="utf-8") as f:
        rows = list(json.load(f)[UNMATCHED_TABLE].values())
    assert len(rows) == 1
    assert rows[0]["sinhala_input"] == "වේගය වෙනස් කරන්න" and rows[0]["frame"]["intent"] == "CHANGE_SPEED"


def test_matched_sentence_is_not_saved(tmp_path):
    db = str(tmp_path / "db.json")
    frame = build_frame("හෙමින් කියන්න", "Say it slowly", "CHANGE_SPEED")
    assert log_unmatched_modifiers("u1", "හෙමින් කියන්න", "Say it slowly", frame, db_path=db) is None
    assert not os.path.exists(db)
