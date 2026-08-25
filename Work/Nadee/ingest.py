import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_documents():
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    for row in load_jsonl("corpus/articles.jsonl"):
        text = row.get("clean_body") or row.get("raw_body", "")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "article",
                    "chunk_id": f"chunk_{row['article_id']}_{i}",
                    "article_id": row["article_id"],
                    "headline": row.get("headline", ""),
                    "section_category": row.get("section_category", ""),
                    "publication_date": row.get("publication_date", ""),
                    "source_url": row.get("source_url", ""),
                }
            ))

    return docs

def list_chunk_ids(limit=10):
    """Utility for Stage 9/10 testing — prints real chunk_ids you can drop into
    contracts.py's retrieved_chunk_id, instead of the placeholder 'chunk_article_1'."""
    docs = build_documents()
    for d in docs[:limit]:
        print(d.metadata["chunk_id"], "->", d.page_content[:40].replace("\n", " "))