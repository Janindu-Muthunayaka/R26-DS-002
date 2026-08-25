from langchain_core.documents import Document

def _docs_from_get(result):
    docs = []
    for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def retrieve_context(vectorstore, query_text, intent, retrieved_chunk_id=None, top_k=4):
    query_text = f"query: {query_text}"  # E5 prefix convention
    collected = []
    seen_chunk_ids = set()

    def add(docs):
        for d in docs:
            cid = d.metadata.get("chunk_id")
            if cid and cid in seen_chunk_ids:
                continue
            if cid:
                seen_chunk_ids.add(cid)
            collected.append(d)

    # 1. Always pull in the current OCR page being read
    ocr_docs = vectorstore.similarity_search(query_text, k=1, filter={"source_type": "ocr_current"})
    add(ocr_docs)

    # 2. Direct anchor: the specific chunk Voice module already picked
    if retrieved_chunk_id:
        anchor_result = vectorstore.get(where={"chunk_id": retrieved_chunk_id})
        add(_docs_from_get(anchor_result))

    # 3. Fill remaining slots with semantic search over the article corpus
    remaining = top_k - len(collected)
    if remaining > 0:
        article_docs = vectorstore.similarity_search(
            query_text, k=remaining, filter={"source_type": "article"}
        )
        add(article_docs)

    return collected[:top_k]