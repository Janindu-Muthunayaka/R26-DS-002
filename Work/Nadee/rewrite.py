# rewrite.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Separate, deterministic LLM instance (temperature 0) just for rewriting.
_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

REWRITE_PROMPT = ChatPromptTemplate.from_template("""You rewrite follow-up questions so they can be understood without the conversation.

Conversation so far:
{history}

Latest question: {question}

Rules:
- If the latest question depends on earlier turns (pronouns such as he, she, it, they, that; or phrases such as "tell me more", "why?", "and the second one?"), rewrite it as ONE complete standalone question in English. Add the missing names and topics from the conversation.
- Keep any requested style or format (for example "in numbered steps", "in simple words").
- If the latest question is already standalone, return it exactly as written.
- Do NOT answer the question. Do NOT add facts that are not in the conversation.
- Return only the question text, nothing else.""")

ANSWER_PREVIEW_CHARS = 300  # answers are long Sinhala text; a preview is enough context


def _format_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        lines.append(f"User: {turn['standalone_question']}")
        lines.append(f"Assistant: {turn['answer'][:ANSWER_PREVIEW_CHARS]}")
    return "\n".join(lines)


def rewrite_query(query: str, history: list[dict]) -> str:
    """Returns a standalone version of `query`. If there is no history, or the
    LLM call fails, returns the original query unchanged (so the pipeline
    never breaks because of the rewrite step)."""
    if not history:
        return query
    try:
        response = (REWRITE_PROMPT | _llm).invoke({
            "history": _format_history(history),
            "question": query,
        })
        rewritten = (response.content or "").strip().strip('"')
        return rewritten or query
    except Exception:
        return query