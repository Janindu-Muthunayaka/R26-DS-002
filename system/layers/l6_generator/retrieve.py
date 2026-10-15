# retrieve.py
from langchain_core.documents import Document

try:
    from .hybrid import hybrid_search
except (ImportError, ValueError):
    from hybrid import hybrid_search

PASSAGE_PREFIX = "passage: "


def _strip_prefix(doc: Document) -> Document:
    text = doc.page_content
    if text.startswith(PASSAGE_PREFIX):
        text = text[len(PASSAGE_PREFIX):]
    return Document(page_content=text, metadata=doc.metadata)


def _docs_from_get(result):
    docs = []
    for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        docs.append(_strip_prefix(Document(page_content=text, metadata=meta)))
    return docs


def retrieve_context(vectorstore, query_text, intent, retrieved_chunk_id=None,
                     top_k=4, use_ocr=True, sparse_query=None,
                     retrieval_mode="hybrid"):
    """
    query_text     : standalone English query (used for the dense search)
    sparse_query   : Sinhala query (used for BM25)
    retrieval_mode : "hybrid" (dense + BM25 + RRF) | "dense" | "bm25"
    """
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

    # 1. Pull in the current OCR page being read (skipped during evaluation)
    if use_ocr and vectorstore is not None:
        try:
            ocr_docs = vectorstore.similarity_search(
                f"query: {query_text}", k=1, filter={"source_type": "ocr_current"}
            )
            add([_strip_prefix(d) for d in ocr_docs])
        except Exception as e:
            print(f"[retrieve] ocr similarity search failed ({e})")

    # 2. Direct anchor: the specific chunk Voice module already picked
    if retrieved_chunk_id and vectorstore is not None:
        try:
            anchor_result = vectorstore.get(where={"chunk_id": retrieved_chunk_id})
            add(_docs_from_get(anchor_result))
        except Exception as e:
            print(f"[retrieve] anchor retrieval failed ({e})")

    # 3. Fill the remaining slots with dense + BM25 + RRF over the article corpus
    if len(collected) < top_k:
        article_docs = hybrid_search(
            vectorstore,
            dense_query=query_text,
            sparse_query=sparse_query,
            top_k=top_k,
            mode=retrieval_mode,
        )
        for d in article_docs:
            if len(collected) >= top_k:
                break
            add([d])

    return collected[:top_k]
