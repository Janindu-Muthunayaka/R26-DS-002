# generate.py
import math

from langchain_openai import ChatOpenAI
from prompt import PROMPT

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.2, logprobs=True)

SINHALA_RANGE = (0x0D80, 0x0DFF)  # Unicode block for Sinhala script


def _sinhala_purity(text: str) -> float:
    """Fraction of alphabetic characters in `text` that are Sinhala script.
    Ignores digits, punctuation, and whitespace so word counts / evidence
    numbers don't skew the ratio."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    sinhala = sum(1 for c in letters if SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1])
    return sinhala / len(letters)


def _perplexity_from_response(response) -> float | None:
    """Self-perplexity of the generated text, computed from the chat-completions
    logprobs the API returns. Returns None if logprobs weren't included
    (e.g. unsupported model, or the field name differs by langchain-openai version —
    print response.response_metadata once to confirm the actual shape)."""
    logprob_content = (response.response_metadata or {}).get("logprobs", {}).get("content")
    if not logprob_content:
        return None
    token_logprobs = [tok["logprob"] for tok in logprob_content if tok.get("logprob") is not None]
    if not token_logprobs:
        return None
    avg_neg_logprob = -sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_neg_logprob)


def generate_answer(query_text, intent, style_class, prompt_modifier, age_instruction,
                     max_words, evidence_docs, chat_history=""):     # NEW parameter
    evidence_text = "\n---\n".join(d.page_content for d in evidence_docs)
    chain = PROMPT | llm

    inputs = {
        "intent": intent,
        "style_class": style_class,
        "prompt_modifier": prompt_modifier,
        "age_instruction": age_instruction,
        "max_words": max_words,
        "evidence": evidence_text,
        "query_text": query_text,
        "chat_history": chat_history or "(නැත)",                    # NEW
    }
    response = chain.invoke(inputs)
    answer = response.content
    perplexity = _perplexity_from_response(response)

    # Guard: prompting alone can't guarantee zero code-switching. If more than
    # ~15% of the answer's letters are non-Sinhala, retry once with an even
    # stricter, isolated instruction appended — don't silently ship English text.
    if _sinhala_purity(answer) < 0.85:
        strict_inputs = dict(inputs)
        strict_inputs["prompt_modifier"] = (
            prompt_modifier
            + " වැදගත්: පෙර පිළිතුරේ සිංහල නොවන වචන තිබුණි. මෙවර පිළිතුර "
              "සම්පූර්ණයෙන්ම සිංහල අක්ෂර වලින් පමණක් ලියන්න — ඉංග්‍රීසි වචනයක් හෝ "
              "අකුරක් හෝ නොමැතිව."
        )
        retry_response = chain.invoke(strict_inputs)
        retry_answer = retry_response.content
        retry_perplexity = _perplexity_from_response(retry_response)
        if _sinhala_purity(retry_answer) >= _sinhala_purity(answer):
            answer = retry_answer
            perplexity = retry_perplexity

    return answer, perplexity