# Tests for delivery/mock_c3.py and data/mock_answers.json.

import pytest

from delivery.mock_c3 import STYLES, MockComponent3, load_articles
from delivery.planner import plan_speech, split_segments


def prompt(style, route="GENERATE"):
    # The same shape handle_voice_command() sends to Component 3.
    return {"route": route, "intent": "SUMMARIZE", "english_translation": "Summarize this",
            "style_class": style, "prompt_modifier": "...", "personalization_flags": {},
            "correction_applied": None}


def test_every_article_has_all_three_styles_in_sinhala():
    articles = load_articles()
    assert len(articles) >= 2
    for a in articles:
        assert a["id"] and a["title"]
        for style in STYLES:
            text = a["answers"][style]
            assert any("\u0d80" <= ch <= "\u0dff" for ch in text), (a["id"], style)


def test_styles_differ_in_the_way_the_planner_sees_them():
    for a in load_articles():
        simple = split_segments(a["answers"]["Simple"])
        detailed = split_segments(a["answers"]["Detailed"])
        steps = split_segments(a["answers"]["StepByStep"])
        assert len(simple) < len(detailed), a["id"]           # Simple is short
        assert sum(s.startswith(("පළමුව", "දෙවනුව", "තෙවනුව")) for s in steps) >= 3, a["id"]


def test_long_answers_get_a_check_in_with_default_settings():
    for a in load_articles():
        assert plan_speech(a["answers"]["Detailed"], "Detailed")["checkpoints"], a["id"]


def test_generate_returns_the_answer_in_the_requested_style():
    c3 = MockComponent3()
    for style in STYLES:
        r = c3.generate(prompt(style), article_index=1)
        assert r["style_class"] == style and r["source"] == "mock_c3"
        assert r["answer_si"] == c3.articles[1]["answers"][style]
        assert r["article_id"] == c3.articles[1]["id"]


def test_unknown_style_falls_back_to_detailed():
    r = MockComponent3().generate(prompt(None))
    assert r["style_class"] == "Detailed"


def test_article_index_wraps_around():
    c3 = MockComponent3()
    n = len(c3.articles)
    assert c3.article(n)["id"] == c3.articles[0]["id"]
    assert c3.article(-1)["id"] == c3.articles[-1]["id"]


def test_system_commands_are_not_sent_to_component_3():
    with pytest.raises(ValueError):
        MockComponent3().generate(prompt(None, route="SYSTEM_COMMAND"))
