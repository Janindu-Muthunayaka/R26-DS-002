# hybrid.py
# Dense (Chroma + e5) + BM25 retrieval merged with Reciprocal Rank Fusion:
#     RRF(d) = sum over rankers r of 1 / (k + rank_r(d))      (k = 60)
from langchain_core.documents import Document

try:
    from .bm25_index import get_bm25
except (ImportError, ValueError):
    from bm25_index import get_bm25

RRF_K = 60
TOP_N = 20                      # candidates taken from EACH retriever
VALID_MODES = ("hybrid", "dense", "bm25")


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """rankings = list of ranked chunk_id lists (best first).
    Returns [(chunk_id, rrf_score), ...] best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(vectorstore, dense_query, sparse_query, top_k: int = 4,
                  mode: str = "hybrid", n_each: int = TOP_N,
                  use_ngrams: bool = True) -> list[Document]:
    """
    dense_query  : text for the embedding search (English is fine, e5 is multilingual)
    sparse_query : SINHALA text for BM25 (BM25 cannot match across scripts)
    mode         : "hybrid" | "dense" | "bm25"
    If one retriever has no usable query / no hits, the other one is used alone.
    Returned Documents carry clean text (no "passage: " prefix) plus
    rrf_score, dense_rank, bm25_rank in their metadata.
    """
    if mode not in VALID_MODES:
        mode = "hybrid"
    index = get_bm25(use_ngrams)

    dense_ids: list[str] = []
    bm25_ids: list[str] = []

    if mode in ("hybrid", "dense") and dense_query and dense_query.strip():
        try:
            hits = vectorstore.similarity_search(
                f"query: {dense_query}", k=n_each, filter={"source_type": "article"}
            )
            dense_ids = [h.metadata.get("chunk_id") for h in hits
                         if h.metadata.get("chunk_id") in index.by_id]
        except Exception as e:
            print(f"[hybrid] dense search failed ({e})")

    if mode in ("hybrid", "bm25") and sparse_query and sparse_query.strip():
        try:
            bm25_ids = [d.metadata["chunk_id"] for d, _ in index.search(sparse_query, n_each)]
        except Exception as e:
            print(f"[hybrid] bm25 search failed ({e})")

    rankings = [r for r in (dense_ids, bm25_ids) if r]
    if not rankings:
        return []

    dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
    bm25_rank = {cid: i for i, cid in enumerate(bm25_ids, start=1)}

    results = []
    for cid, score in rrf_fuse(rankings)[:top_k]:
        base = index.by_id.get(cid)
        if not base:
            continue
        meta = dict(base.metadata)
        meta.update({
            "rrf_score": round(score, 5),
            "dense_rank": dense_rank.get(cid),
            "bm25_rank": bm25_rank.get(cid),
            "retrieval_mode": mode,
        })
        results.append(Document(page_content=base.page_content, metadata=meta))
    return results
