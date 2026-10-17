# services/rag/rewrite.py
from core import llm

ANSWER_PREVIEW_CHARS = 300


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
    """Returns a standalone version of `query` using conversation history."""
    if not history or not query:
        return query

    # Only rewrite if there is history and query looks like a follow-up
    prompt = [
        {
            "role": "system",
            "content": (
                "You rewrite follow-up questions so they can be understood as standalone queries.\n\n"
                "Rules:\n"
                "- If the question refers to a specific item from a previous turn using pronouns (e.g. 'මේක', 'මෙය', 'this', 'it'), "
                "replace the pronoun with the core entity name (e.g. replace 'මේක ගාල්ලේ කොහෙද තියෙන්නේ?' with 'විකිරණ යන්ත්‍රය ගාල්ලේ කොහෙද තියෙන්නේ?'). "
                "Do NOT add the answer or location details into the rewritten question.\n"
                "- If multiple articles were read in the conversation and the question can apply generally (e.g. 'මෙයින් ඇතිවිය හැකි අනතුර කුමක්ද?'), "
                "do NOT arbitrarily restrict it to just one article. Keep the question comprehensive (e.g. 'කියවන ලද ලිපි වලින් ඇතිවිය හැකි අනතුර කුමක්ද?').\n"
                "- If the question is in Sinhala, keep the rewritten question in natural Sinhala.\n"
                "- If the question is in English, return it in English.\n"
                "- If the question is already fully standalone, return it unchanged.\n"
                "- Return ONLY the rewritten question text, nothing else."
            )
        },
        {
            "role": "user",
            "content": f"Conversation so far:\n{_format_history(history)}\n\nLatest question: {query}"
        }
    ]
    try:
        rewritten, reason = llm.chat(prompt, temperature=0.0)
        if rewritten and rewritten.strip():
            return rewritten.strip().strip('"')
        return query
    except Exception as e:
        print(f"[rewrite] rewriting failed: {e!r} - using original query")
        return query
