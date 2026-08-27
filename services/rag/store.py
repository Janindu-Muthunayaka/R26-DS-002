"""
The vector store. JSON for the text, one numpy array for the vectors.

WHY NOT CHROMA + LANGCHAIN + SENTENCE-TRANSFORMERS
--------------------------------------------------
That is what `Work/Nadee/vectorstore.py` uses and it is a reasonable choice in
isolation. Here it is the wrong trade, for three reasons:

  * It is the largest install in the project — chromadb, langchain, torch and
    a sentence-transformers checkpoint — for a store that will hold a few
    thousand chunks. Cosine similarity over 5,000 x 1536 floats is one numpy
    dot product and takes milliseconds.
  * It is a fourth incompatible dependency set to keep alive until October.
  * Chroma's on-disk format is not something you can open and read when a
    retrieval looks wrong at 11pm. `manifest.json` is.

What is KEPT from Component 3 is everything that carries its author's thinking:
the prompt, the Sinhala purity guard, the style/detail word limits and the
chunk metadata shape. See `answer.py`. `Work/Nadee/` is untouched.

Vectors are L2-normalised on the way in, so cosine similarity is a dot product
and ranking is a single matrix multiply.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


def text_hash(text: str) -> str:
    return hashlib.sha1((text or '').encode('utf-8')).hexdigest()[:16]


def record_hash(record: dict) -> str:
    """The dedupe key: source_type AND text.

    Text alone is wrong, and the way it is wrong is subtle. The page being
    read is indexed twice on purpose — once as `ocr_current`, which is
    replaced on every request, and once as `read`, which is kept. Those two
    have identical text, so a text-only key made the second one look like a
    duplicate of the first and the "remember what it reads" corpus silently
    never grew. Caught by a test, not by looking at it.
    """
    kind = (record.get('metadata') or {}).get('source_type', '')
    return text_hash(f'{kind}\x00{record.get("text", "")}')


class VectorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = self.path / 'manifest.json'
        self.vectors_file = self.path / 'vectors.npy'
        self.chunks: list = []           # [{chunk_id, text, hash, metadata}]
        self.vectors: Optional[np.ndarray] = None
        self.load()

    # ---- persistence ----------------------------------------------------
    def load(self) -> None:
        try:
            if self.manifest.exists():
                self.chunks = json.loads(
                    self.manifest.read_text(encoding='utf-8'))
            if self.vectors_file.exists():
                self.vectors = np.load(self.vectors_file)
        except Exception:
            # A corrupt store must not stop the service from starting. It
            # starts empty and says so in /stats, which is recoverable; a
            # service that will not boot at a viva is not.
            self.chunks, self.vectors = [], None

        n = len(self.chunks)
        if self.vectors is None or len(self.vectors) != n:
            # The two files disagree. Trust neither — the alternative is
            # answering questions with text attached to the wrong vectors,
            # which looks like working retrieval and is not.
            self.chunks, self.vectors = [], None

    def save(self) -> None:
        tmp = self.manifest.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.chunks, ensure_ascii=False),
                       encoding='utf-8')
        tmp.replace(self.manifest)
        if self.vectors is not None:
            np.save(self.vectors_file, self.vectors)

    # ---- writing ---------------------------------------------------------
    def known_hashes(self) -> set:
        return {c['hash'] for c in self.chunks}

    def add(self, records: list, vectors: list) -> int:
        """records: [{chunk_id, text, metadata}] with a vector each."""
        if not records:
            return 0
        assert len(records) == len(vectors), 'records and vectors must pair up'
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms

        for r in records:
            r.setdefault('metadata', {})
            r['hash'] = record_hash(r)
        self.chunks.extend(records)
        self.vectors = arr if self.vectors is None else \
            np.vstack([self.vectors, arr])
        return len(records)

    def delete_where(self, key: str, value) -> int:
        keep = [i for i, c in enumerate(self.chunks)
                if c.get('metadata', {}).get(key) != value]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = (self.vectors[keep]
                            if self.vectors is not None and keep else None)
        return removed

    # ---- reading ---------------------------------------------------------
    def search(self, vector, k: int = 4, where: dict = None) -> list:
        """Top-k by cosine similarity. `where` filters on metadata first."""
        if self.vectors is None or not self.chunks:
            return []
        idx = list(range(len(self.chunks)))
        if where:
            idx = [i for i in idx
                   if all(self.chunks[i].get('metadata', {}).get(kk) == vv
                          for kk, vv in where.items())]
        if not idx:
            return []

        q = np.asarray(vector, dtype=np.float32)
        n = float(np.linalg.norm(q)) or 1.0
        scores = self.vectors[idx] @ (q / n)
        order = np.argsort(-scores)[:max(1, k)]
        out = []
        for j in order:
            c = dict(self.chunks[idx[int(j)]])
            c['score'] = round(float(scores[int(j)]), 4)
            out.append(c)
        return out

    def get_where(self, key: str, value) -> list:
        return [dict(c) for c in self.chunks
                if c.get('metadata', {}).get(key) == value]

    def stats(self) -> dict:
        kinds = {}
        for c in self.chunks:
            k = c.get('metadata', {}).get('source_type', 'unknown')
            kinds[k] = kinds.get(k, 0) + 1
        return {'chunks': len(self.chunks),
                'dims': int(self.vectors.shape[1]) if self.vectors is not None
                        and len(self.vectors) else 0,
                'by_source_type': kinds,
                'path': str(self.path)}
