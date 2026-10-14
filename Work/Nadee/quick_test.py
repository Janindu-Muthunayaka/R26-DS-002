# quick_test.py
from dotenv import load_dotenv
load_dotenv()

from generate import generate_answer
from langchain_core.documents import Document

fake_docs = [Document(page_content="පාස්කු ප්‍රහාරය සම්බන්ධයෙන් විමර්ශන පවතී.")]

answer, perplexity = generate_answer(
    query_text="Can you explain this a little more?",
    intent="EXPLAIN",
    style_class="Detailed",
    prompt_modifier="Provide a thorough, comprehensive explanation with full context and detail.",
    age_instruction="පිළිතුර වැඩිහිටි පාඨකයෙකුට සුදුසු සම්පූර්ණ, විස්තරාත්මක මට්ටමින් ලියන්න.",
    max_words=400,
    evidence_docs=fake_docs,
)

print("ANSWER:", answer)
print("PERPLEXITY:", perplexity)