# personalization/system_commands.py
#
# System / navigation commands — the fixed control vocabulary of the reader
# app (move forward, stop, etc.). These are NOT content requests, so they are
# kept completely out of the personalization path:
#
#   - NOT sent to the style model for prediction
#   - NEVER used as training evidence (a "stop" tells you nothing about
#     whether someone prefers simple or detailed explanations)
#   - still logged, but with style_class = None
#
# MATCHING IS DELIBERATELY REDUNDANT. Llama 3.2:1b invents its own intent
# names, so a registry keyed only on intent labels will always have gaps.
# We match on THREE signals:
#     1. the intent label
#     2. the whole normalized utterance (exact)
#     3. a distinctive multi-word phrase contained in the utterance
#
# Question guards protect explanatory / interrogative content requests
# from being hijacked by generic command words.

import re

SYSTEM_COMMANDS = {
    "GO_FORWARD": {
        "action": "Advance to the next item",
        "phrases": {
            "go forward", "go forwards", "forward", "move forward",
            "next", "go next", "continue", "go ahead", "go on",
            "proceed", "carry on", "keep going", "onwards",
        },
        "contains": {
            "go forward", "go ahead", "move forward", "carry on",
            "keep going", "next page", "next article", "go to the next",
        },
        "intents": {
            "GO_FORWARD", "GO_AHEAD", "GO_ON", "NEXT", "READ_NEXT",
            "NEXT_PAGE", "NEXT_ARTICLE", "CONTINUE", "PROCEED",
            "MOVE_FORWARD", "FORWARD", "ADVANCE",
        },
    },
    "STOP": {
        "action": "Stop playback",
        "phrases": {
            "stop", "stop it", "stop please", "please stop",
            "halt", "pause", "pause it", "quiet", "be quiet",
            "stop reading", "stop talking",
        },
        "contains": {
            "stop reading", "stop talking", "stop it",
        },
        "intents": {
            "STOP", "PAUSE", "STOP_READING", "STOP_AUDIO",
            "HALT", "CANCEL", "QUIET",
        },
    },
    "GO_BACK": {
        "action": "Go back to the previous item",
        "phrases": {
            "go back", "back", "previous", "previous one", "previous page",
            "previous article", "previous news", "the previous one",
        },
        "contains": {
            "previous page", "previous article", "previous news",
            "go back to the previous",
        },
        "intents": {
            "PREVIOUS", "GO_BACK", "BACK", "PREVIOUS_PAGE",
            "PREVIOUS_ARTICLE", "GO_PREVIOUS",
        },
    },
    "GO_TO_PAGE": {
        "action": "Jump to a specific page (number, first or last)",
        "phrases": {
            "first page", "go to the first page", "last page",
            "go to the last page",
        },
        "contains": set(),
        "intents": {
            "GO_TO_PAGE", "NAVIGATE_PAGE_NUMBER", "GO_TO_START", "GO_TO_END",
            "PAGE_NUMBER", "GO_TO_FIRST_PAGE", "GO_TO_LAST_PAGE",
        },
    },
    "GO_TO_SECTION": {
        "action": "Jump to a named section of the newspaper",
        "phrases": set(),
        "contains": set(),
        "intents": {
            "GO_TO_SECTION", "NAVIGATE_SECTION", "OPEN_SECTION",
        },
    },
    "READ_HEADLINES": {
        "action": "Read out the list of headlines",
        "phrases": {
            "headlines", "read the headlines", "read headlines",
            "what are the headlines",
        },
        "contains": set(),
        "intents": {
            "READ_HEADLINES", "HEADLINES", "LIST_HEADLINES",
        },
    },
}


def _normalize(text):
    """Lowercase, remove ALL punctuation, collapse whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(cleaned.split())


def _normalize_intent(intent):
    if not intent:
        return ""
    return re.sub(r"[\s\-]+", "_", str(intent).strip().upper())


def detect_system_command(intent, english_text):
    """
    Returns the canonical command name (e.g. "GO_FORWARD") if this turn is a
    system/navigation command, otherwise None.
    """
    intent_key = _normalize_intent(intent)
    phrase = _normalize(english_text)
    words = phrase.split() if phrase else []

    # If the user utterance is an explanatory request or interrogative question,
    # do NOT hijack it as a system navigation command.
    question_starters = {"what", "who", "where", "why", "when", "how", "which", "tell", "explain", "describe"}
    is_question = (
        ("?" in str(english_text or "")) or
        (bool(words) and words[0] in question_starters and intent_key not in {"READ_HEADLINES"}) or
        (len(words) >= 2 and words[0] in {"can", "could", "is", "are", "do", "does"} and words[1] in {"you", "it", "there", "this", "that"})
    )
    if is_question:
        return None

    for name, spec in SYSTEM_COMMANDS.items():
        if phrase and phrase in spec["phrases"]:
            return name
        if phrase:
            for fragment in spec.get("contains", ()):
                if fragment in phrase and len(words) <= 5:
                    return name
        if intent_key in spec["intents"]:
            # Generic navigation intents only apply if the utterance is short / imperative
            if intent_key in {"NEXT", "CONTINUE", "PROCEED", "ADVANCE", "FORWARD", "GO_ON", "GO_AHEAD"}:
                if len(words) > 3:
                    continue
            return name

    return None


def get_command_action(command_name):
    """Human-readable description of what the command does."""
    spec = SYSTEM_COMMANDS.get(command_name)
    return spec["action"] if spec else None


def is_system_command(intent, english_text):
    """Convenience boolean wrapper around detect_system_command()."""
    return detect_system_command(intent, english_text) is not None
