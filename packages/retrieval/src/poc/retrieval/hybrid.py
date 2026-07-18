from __future__ import annotations

import logging
from typing import Any

from poc.retrieval.bm25 import BM25Index
from poc.retrieval.qdrant_index import QdrantIndex, ScoredChunk
from poc.retrieval.reranker import Reranker

_log = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    bm25_results: list[ScoredChunk],
    dense_results: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """Merge two ranked lists via Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    for rank, sc in enumerate(bm25_results):
        cid = sc.chunk.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = sc

    for rank, sc in enumerate(dense_results):
        cid = sc.chunk.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = sc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [ScoredChunk(chunk=chunk_map[cid].chunk, score=score) for cid, score in ranked]


class HybridRetriever:
    """BM25 + dense vector search fused via RRF, then reranked.

    Typical flow:
        1. BM25 → top-50
        2. Dense (Qdrant) → top-50
        3. RRF fusion
        4. Reranker → top-k
    """

    def __init__(
        self,
        *,
        qdrant_index: QdrantIndex,
        bm25_index: BM25Index,
        reranker: Reranker,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
    ) -> None:
        self._qdrant = qdrant_index
        self._bm25 = bm25_index
        self._reranker = reranker
        self._bm25_k = bm25_candidates
        self._dense_k = dense_candidates

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        bm25_hits = self._bm25.search(query, top_k=self._bm25_k)
        dense_hits = self._qdrant.search(query, top_k=self._dense_k, filters=filters)

        merged = _reciprocal_rank_fusion(bm25_hits, dense_hits)
        reranked = self._reranker.rerank(query, merged, top_k=top_k)

        _log.debug(
            "HybridRetriever: bm25=%d dense=%d merged=%d reranked=%d",
            len(bm25_hits),
            len(dense_hits),
            len(merged),
            len(reranked),
        )
        return reranked
