# llm_generate.py
import math
import os

try:
    from .prompt import PROMPT
    from .tts_clean import make_speakable
except (ImportError, ValueError):
    from prompt import PROMPT
    from tts_clean import make_speakable

SINHALA_RANGE = (0x0D80, 0x0DFF)  # Unicode block for Sinhala script
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        _llm = ChatOpenAI(model=model_name, temperature=0.2, logprobs=True)
    return _llm


def _sinhala_purity(text: str) -> float:
    """Fraction of alphabetic characters in `text` that are Sinhala script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    sinhala = sum(1 for c in letters if SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1])
    return sinhala / len(letters)


def _perplexity_from_response(response) -> float | None:
    """Self-perplexity of the generated text, computed from the chat-completions logprobs."""
    logprob_content = (response.response_metadata or {}).get("logprobs", {}).get("content")
    if not logprob_content:
        return None
    token_logprobs = [tok["logprob"] for tok in logprob_content if tok.get("logprob") is not None]
    if not token_logprobs:
        return None
    avg_neg_logprob = -sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_neg_logprob)


def generate_answer(query_text, intent, style_class, prompt_modifier, age_instruction,
                     max_words, evidence_docs, chat_history=""):
    evidence_text = "\n---\n".join(d.page_content for d in evidence_docs)
    llm = _get_llm()
    chain = PROMPT | llm

    inputs = {
        "intent": intent,
        "style_class": style_class,
        "prompt_modifier": prompt_modifier,
        "age_instruction": age_instruction,
        "max_words": max_words,
        "evidence": evidence_text,
        "query_text": query_text,
        "chat_history": chat_history or "(නැත)",
    }
    response = chain.invoke(inputs)
    # The answer is spoken: strip markdown/symbols the TTS would read aloud.
    answer = make_speakable(response.content)
    perplexity = _perplexity_from_response(response)

    # Guard: prompting alone can't guarantee zero code-switching.
    if _sinhala_purity(answer) < 0.85:
        strict_inputs = dict(inputs)
        strict_inputs["prompt_modifier"] = (
            (prompt_modifier or "")
            + " වැදගත්: පෙර පිළිතුරේ සිංහල නොවන වචන තිබුණි. මෙවර පිළිතුර "
              "සම්පූර්ණයෙන්ම සිංහල අක්ෂර වලින් පමණක් ලියන්න — ඉංග්‍රීසි වචනයක් හෝ "
              "අකුරක් හෝ නොමැතිව."
        )
        retry_response = chain.invoke(strict_inputs)
        retry_answer = make_speakable(retry_response.content)
        retry_perplexity = _perplexity_from_response(retry_response)
        if _sinhala_purity(retry_answer) >= _sinhala_purity(answer):
            answer = retry_answer
            perplexity = retry_perplexity

    return answer, perplexity
