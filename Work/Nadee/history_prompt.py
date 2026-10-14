# history_prompt.py
# Turns the stored chat history into a short text block for the LLM prompt.
# The block is CONTEXT only (what was asked / answered before), never evidence.
from websearch import is_refusal

PROMPT_TURNS = 2     # how many recent turns go into the prompt
ANSWER_CHARS = 400   # each earlier answer is cut to this many characters


def format_history_for_prompt(turns: list[dict],
                              max_turns: int = PROMPT_TURNS,
                              answer_chars: int = ANSWER_CHARS) -> str:
    """Returns "" when there is nothing useful to include.
    - turns whose answer was a refusal are skipped (they would bias the model
      towards refusing again)
    - only the last `max_turns` usable turns are kept
    - earlier answers are truncated to keep the prompt short"""
    usable = [t for t in turns if not is_refusal(t.get("answer", ""))]
    usable = usable[-max_turns:]

    lines = []
    for t in usable:
        question = t.get("standalone_question") or t.get("question", "")
        answer = (t.get("answer") or "").strip()
        if len(answer) > answer_chars:
            answer = answer[:answer_chars].rstrip() + "…"
        lines.append(f"පරිශීලකයා: {question}")
        lines.append(f"ඔබේ පිළිතුර: {answer}")
    return "\n".join(lines)