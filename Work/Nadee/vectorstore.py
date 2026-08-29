from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ingest import build_documents

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

def build_vectorstore():
    """Idempotent — only embeds the article corpus once, connects on later runs."""
    vs = Chroma(
        collection_name="reading_assistant",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    existing = vs._collection.count()
    if existing > 0:
        print(f"Vector store already has {existing} chunks — skipping re-index.")
        return vs

    docs = build_documents()
    for d in docs:
        d.page_content = f"passage: {d.page_content}"
    vs.add_documents(docs)
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

#Assumption flagged: each new OCR page deletes and replaces the previous ocr_current entry. If you ever add multi-user support, this is where session_id filtering would need to come back.