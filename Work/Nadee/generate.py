from langchain_openai import ChatOpenAI
from prompt import PROMPT

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.2)

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


def generate_answer(query_text, intent, style_class, prompt_modifier, max_words, evidence_docs):
    evidence_text = "\n---\n".join(d.page_content for d in evidence_docs)
    chain = PROMPT | llm

    inputs = {
        "intent": intent,
        "style_class": style_class,
        "prompt_modifier": prompt_modifier,
        "max_words": max_words,
        "evidence": evidence_text,
        "query_text": query_text,
    }
    response = chain.invoke(inputs)
    answer = response.content

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
        if _sinhala_purity(retry_answer) >= _sinhala_purity(answer):
            answer = retry_answer

    return answer