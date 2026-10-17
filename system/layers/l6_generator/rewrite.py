# rewrite.py
import os
from langchain_core.prompts import ChatPromptTemplate

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        _llm = ChatOpenAI(model=model_name, temperature=0)
    return _llm


REWRITE_PROMPT = ChatPromptTemplate.from_template("""You rewrite follow-up questions so they can be understood as standalone queries.

Rules:
- If the question refers to a specific item from a previous turn using pronouns (e.g. 'මේක', 'මෙය', 'this', 'it'), replace the pronoun with the core entity name (e.g. replace 'මේක ගාල්ලේ කොහෙද තියෙන්නේ?' with 'විකිරණ යන්ත්‍රය ගාල්ලේ කොහෙද තියෙන්නේ?'). Do NOT add the answer or location details into the rewritten question.
- If multiple articles were read in the conversation and the question can apply generally (e.g. 'මෙයින් ඇතිවිය හැකි අනතුර කුමක්ද?'), do NOT arbitrarily restrict it to just one article. Keep the question comprehensive (e.g. 'කියවන ලද ලිපි වලින් ඇතිවිය හැකි අනතුර කුමක්ද?').
- If the question is in Sinhala, keep the rewritten question in natural Sinhala.
- If the question is in English, return it in English.
- If the question is already fully standalone, return it unchanged.
- Return ONLY the rewritten question text, nothing else.

Conversation so far:
{history}

Latest question: {question}
""")

ANSWER_PREVIEW_CHARS = 300  # answers are long Sinhala text; a preview is enough context


def _format_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        q = (turn.get('standalone_question') or turn.get('question') or
             turn.get('user_query') or turn.get('query') or '')
        ans = (turn.get('answer') or turn.get('assistant_answer') or turn.get('answer_si') or '')[:ANSWER_PREVIEW_CHARS]
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {ans}")
    return "\n".join(lines)


def rewrite_query(query: str, history: list[dict]) -> str:
    """Returns a standalone version of `query`. If there is no history, or the
    LLM call fails, returns the original query unchanged (so the pipeline
    never breaks because of the rewrite step)."""
    if not history:
        return query
    try:
        llm = _get_llm()
        response = (REWRITE_PROMPT | llm).invoke({
            "history": _format_history(history),
            "question": query,
        })
        rewritten = (response.content or "").strip().strip('"')
        return rewritten or query
    except Exception as e:
        print(f"[rewrite] rewriting failed ({e}) - using original query")
        return query
