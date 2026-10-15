# delivery/mock_c3.py
# A stand-in for Component 3 (RAG, Nadee) so the whole Component 4 turn can
# run on the laptop: it takes the same payload handle_voice_command() builds
# for Component 3 ("final_prompt", route GENERATE) and returns a canned
# Sinhala answer for the current article in the requested style.
#
# Answers live in data/mock_answers.json: articles, each with one answer per
# style (Simple / Detailed / StepByStep). The intent does not pick a
# different answer: a mock cannot answer free questions, and the style is
# what Component 4 personalizes.
#
# In the full system Component 3's reply goes to the reader, not through
# svc-voice (ADAPTIVE_VOICE_PLAN.md, "Where things stand"), so this file is
# only for the standalone demo and is never integrated.

import json
import os

MOCK_ANSWERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mock_answers.json"
)
STYLES = ("Simple", "Detailed", "StepByStep")
FALLBACK_STYLE = "Detailed"


def load_articles(path=MOCK_ANSWERS_PATH):
    """The list of mock articles: [{id, title, answers: {style: text}}, ...]."""
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)["articles"]
    for a in articles:
        missing = [s for s in STYLES if not a["answers"].get(s)]
        if missing:
            raise ValueError(f"mock article {a['id']} has no answer for {missing}")
    return articles


class MockComponent3:
    def __init__(self, path=MOCK_ANSWERS_PATH):
        self.articles = load_articles(path)

    def article(self, index):
        """The article at `index`, wrapping around at both ends (next/previous)."""
        return self.articles[index % len(self.articles)]

    def generate(self, final_prompt, article_index=0):
        """
        Answers one Component 3 request.

        final_prompt — handle_voice_command()["final_prompt"] with route
                       "GENERATE" (style_class, intent, english_translation,
                       prompt_modifier, personalization_flags, ...)
        Returns {"answer_si", "style_class", "article_id", "title", "intent",
                 "source": "mock_c3"}.
        """
        if final_prompt.get("route") != "GENERATE":
            raise ValueError(f"Component 3 only answers GENERATE requests, got {final_prompt.get('route')}")
        style = final_prompt.get("style_class")
        if style not in STYLES:
            style = FALLBACK_STYLE
        article = self.article(article_index)
        return {
            "answer_si": article["answers"][style],
            "style_class": style,
            "article_id": article["id"],
            "title": article["title"],
            "intent": final_prompt.get("intent"),
            "source": "mock_c3",
        }
