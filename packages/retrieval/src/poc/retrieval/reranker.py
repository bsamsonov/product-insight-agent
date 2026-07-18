from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from poc.retrieval.qdrant_index import ScoredChunk

_log = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]: ...


# Fast default (~22M params, English-focused, ~10-15x faster than the heavy alternative).
# Switch to RERANKER_MODEL_HEAVY for multilingual or higher-accuracy needs.
RERANKER_MODEL_FAST = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_HEAVY = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Cross-encoder reranker using a local sentence-transformer cross-encoder model."""

    def __init__(self, model_name: str = RERANKER_MODEL_FAST) -> None:
        from sentence_transformers import CrossEncoder

        _log.info("Loading cross-encoder reranker: %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        """Rerank the provided candidates using the cross-encoder model."""
        if not candidates:
            return []
        pairs = [(query, sc.chunk.text) for sc in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()
        ranked = sorted(zip(candidates, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=sc.chunk, score=float(score)) for sc, score in ranked[:top_k]]


class NoOpReranker:
    """Reranker that just truncates — used for testing without a model."""

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        return sorted(candidates, key=lambda sc: sc.score, reverse=True)[:top_k]
