from __future__ import annotations

import re

from poc.ingestion.chunker import Chunk
from poc.retrieval.qdrant_index import ScoredChunk
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index over a list of chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._by_id = {c.id: c for c in self._chunks}
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, *, top_k: int = 50) -> list[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=self._chunks[idx], score=float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        """Look up an indexed chunk by id (None if unknown)."""
        return self._by_id.get(chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)
