# vectorstore.py
import os

from langchain_core.documents import Document

try:
    from .ingest import build_documents
except (ImportError, ValueError):
    from ingest import build_documents

_HERE = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(_HERE, "chroma_db")
if not os.path.exists(PERSIST_DIR):
    _ALT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "Work", "Nadee", "chroma_db"))
    if os.path.exists(_ALT):
        PERSIST_DIR = _ALT

COLLECTION = "reading_assistant"
BATCH = 500

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        except Exception as e:
            print(f"[vectorstore] HuggingFaceEmbeddings load failed ({e})")
            _embeddings = None
    return _embeddings


def _open():
    try:
        from langchain_chroma import Chroma
        emb = get_embeddings()
        if emb is None:
            return None
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=emb,
            persist_directory=PERSIST_DIR,
        )
    except Exception as e:
        print(f"[vectorstore] Chroma open failed ({e})")
        return None


def _first_article_meta(vs):
    try:
        res = vs.get(where={"source_type": "article"}, limit=1, include=["metadatas"])
    except Exception:
        return None
    metas = res.get("metadatas") or []
    return metas[0] if metas else None


def build_vectorstore():
    """Idempotent - connects to stored Chroma vector store, or embeds corpus once."""
    vs = _open()
    if vs is None:
        return None

    meta = _first_article_meta(vs)
    if meta is not None and "para_index" not in meta:
        print("[vectorstore] old chunk format found - rebuilding the index ...")
        vs.delete_collection()
        vs = _open()
        meta = None

    if meta is not None:
        print("Vector store already has article chunks - skipping re-index.")
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
    """Replaces the single 'current' OCR chunk."""
    if vectorstore is None:
        return
    try:
        vectorstore.delete(where={"source_type": "ocr_current"})
    except Exception:
        pass

    doc = Document(
        page_content=f"passage: {ocr_payload['corrected_text']}",
        metadata={
            "source_type": "ocr_current",
            "chunk_id": "chunk_ocr_current",
        },
    )
    try:
        vectorstore.add_documents([doc])
    except Exception as e:
        print(f"[vectorstore] add_current_ocr_chunk failed: {e}")
