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
# Observed on the real stack: "ඉදිරියට යන්න" produced intent GO_AHEAD, which
# was not in the original registry, so a navigation command leaked through
# and received a full RAG generation prompt. We therefore match on THREE
# signals, any one of which is sufficient:
#     1. the intent label
#     2. the whole normalized utterance (exact)
#     3. a distinctive multi-word phrase contained in the utterance
#
# Single words like "next" or "continue" are matched ONLY as a whole
# utterance, never as substrings — otherwise "continue explaining this
# simply" would be misread as a navigation command.
#
# To add a command later, add one entry to SYSTEM_COMMANDS below.

import re

SYSTEM_COMMANDS = {
    "GO_FORWARD": {
        "action": "Advance to the next item",
        # Matched against the ENTIRE normalized utterance
        "phrases": {
            "go forward", "go forwards", "forward", "move forward",
            "next", "go next", "continue", "go ahead", "go on",
            "proceed", "carry on", "keep going", "onwards",
        },
        # Matched anywhere inside the utterance (multi-word only, to avoid
        # hijacking content requests that happen to contain one of these)
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
}


def _normalize(text):
    """Lowercase, remove ALL punctuation, collapse whitespace.

    Removing punctuation globally (not just trailing) matters because NLLB
    emits detached punctuation — "Go ahead ." — which a simple rstrip leaves
    as "go ahead " and fails to match.
    """
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

    for name, spec in SYSTEM_COMMANDS.items():
        if intent_key in spec["intents"]:
            return name
        if phrase and phrase in spec["phrases"]:
            return name
        if phrase:
            for fragment in spec.get("contains", ()):
                if fragment in phrase:
                    return name

    return None


def get_command_action(command_name):
    """Human-readable description of what the command does."""
    spec = SYSTEM_COMMANDS.get(command_name)
    return spec["action"] if spec else None


def is_system_command(intent, english_text):
    """Convenience boolean wrapper around detect_system_command()."""
    return detect_system_command(intent, english_text) is not None
