# vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ingest import build_documents

PERSIST_DIR = "./chroma_db"
COLLECTION = "reading_assistant"
BATCH = 500   # Chroma rejects very large single inserts

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")


def _open():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def _first_article_meta(vs):
    try:
        res = vs.get(where={"source_type": "article"}, limit=1, include=["metadatas"])
    except Exception:
        return None
    metas = res.get("metadatas") or []
    return metas[0] if metas else None


def build_vectorstore():
    """Idempotent - embeds the article corpus once, connects on later runs.
    If the stored index was built with the OLD 400-character chunks (no
    'para_index' in the metadata) it is dropped and rebuilt with the new
    paragraph-level chunks."""
    vs = _open()

    meta = _first_article_meta(vs)
    if meta is not None and "para_index" not in meta:
        print("[vectorstore] old chunk format found - rebuilding the index ...")
        vs.delete_collection()
        vs = _open()
        meta = None

    if meta is not None:
        print(f"Vector store already has article chunks - skipping re-index.")
        return vs

    docs = build_documents()
    prefixed = [
        Document(page_content=f"passage: {d.page_content}", metadata=d.metadata)
        for d in docs
    ]
    ids = [d.metadata["chunk_id"] for d in docs]
    for i in range(0, len(prefixed), BATCH):
        vs.add_documents(prefixed[i:i + BATCH], ids=ids[i:i + BATCH])
        print(f"[vectorstore] indexed {min(i + BATCH, len(prefixed))}/{len(prefixed)}")
    return vs


def add_current_ocr_chunk(vectorstore, ocr_payload: dict):
    """No session_id anymore, so we replace the single 'current' OCR chunk on
    every call instead of scoping by session. Fine for a one-user prototype."""
    vectorstore.delete(where={"source_type": "ocr_current"})

    doc = Document(
        page_content=f"passage: {ocr_payload['corrected_text']}",
        metadata={
            "source_type": "ocr_current",
            "chunk_id": "chunk_ocr_current",
        },
    )
    vectorstore.add_documents([doc])

# Assumption flagged: each new OCR page deletes and replaces the previous ocr_current entry. If you ever add multi-user support, this is where session_id filtering would need to come back.
