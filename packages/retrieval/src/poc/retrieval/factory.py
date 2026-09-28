"""Build the hybrid retriever (BM25 + Qdrant dense + reranker) with graceful degradation.

One place that the API and ``scripts/run_eval.py`` share, so both run the same retrieval
stack. Each component falls back independently:

- **BM25** is built in memory from the corpus JSONL (always, whole corpus by default).
- **Dense** search is used only when Qdrant is reachable *and* the tenant collection
  already holds points — the heavy BGE-M3 embedder is not even loaded otherwise, and an
  empty collection is never created as a side effect.
- **Reranker** is the local cross-encoder when it loads, otherwise a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poc.retrieval.bm25 import BM25Index
from poc.retrieval.hybrid import HybridRetriever
from poc.retrieval.reranker import NoOpReranker

_log = logging.getLogger(__name__)


class _NoOpDense:
    def search(self, *args: Any, **kwargs: Any) -> list:
        return []


@dataclass(frozen=True)
class RetrieverInfo:
    """What the factory actually built — for logs, reports and health checks."""

    bm25_chunks: int
    dense_points: int
    reranker: str

    @property
    def mode(self) -> str:
        return "hybrid" if self.dense_points else "bm25-only"


def build_bm25(corpus_path: Path, *, limit: int | None = None) -> BM25Index:
    """Chunk the corpus with the project chunker and build an in-memory BM25 index."""
    from poc.ingestion.chunker import RecursiveTokenChunker
    from poc.ingestion.normalizer import normalize
    from poc.ingestion.sources.jsonl import JsonlSource

    bm25 = BM25Index()
    if not corpus_path.exists():
        _log.warning("Corpus %s not found — BM25 index is empty", corpus_path)
        return bm25  # unbuilt index: search() returns [] and len() is 0
    chunker = RecursiveTokenChunker()
    chunks: list[Any] = []
    for i, raw_doc in enumerate(JsonlSource(corpus_path).iter()):
        if limit is not None and i >= limit:
            break
        chunks.extend(chunker.chunk(normalize(raw_doc)))
    bm25.build(chunks)
    return bm25


def _dense_index(qdrant_url: str, tenant: str) -> tuple[Any, int]:
    """Return (dense index, point count), or a no-op and 0 when unavailable."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=5)
        collection = f"reviews__{tenant}"
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            _log.warning("Qdrant collection %s not found — dense search off", collection)
            return _NoOpDense(), 0
        points = client.count(collection).count
        if points == 0:
            _log.warning("Qdrant collection %s is empty — dense search off", collection)
            return _NoOpDense(), 0

        from poc.retrieval.bge_embedder import BgeM3Embedder
        from poc.retrieval.qdrant_index import QdrantIndex

        index = QdrantIndex(embedder=BgeM3Embedder(), client=client, tenant=tenant)
        return index, points
    except Exception as exc:
        _log.warning("Qdrant unavailable at %s (%s) — dense search off", qdrant_url, exc)
        return _NoOpDense(), 0


def _reranker(enabled: bool) -> tuple[Any, str]:
    if not enabled:
        return NoOpReranker(), "none"
    try:
        from poc.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        return reranker, type(reranker).__name__
    except Exception as exc:
        _log.warning("Cross-encoder reranker unavailable (%s) — no reranking", exc)
        return NoOpReranker(), "none"


def build_hybrid_retriever(
    corpus_path: Path,
    *,
    qdrant_url: str = "http://localhost:6333",
    tenant: str = "default",
    use_dense: bool = True,
    use_reranker: bool = True,
    bm25_limit: int | None = None,
) -> tuple[HybridRetriever, RetrieverInfo]:
    """Build the retriever used by the agent. Never raises for missing optional infra."""
    bm25 = build_bm25(corpus_path, limit=bm25_limit)
    dense, points = _dense_index(qdrant_url, tenant) if use_dense else (_NoOpDense(), 0)
    reranker, reranker_name = _reranker(use_reranker)
    info = RetrieverInfo(bm25_chunks=len(bm25), dense_points=points, reranker=reranker_name)
    _log.info(
        "Retriever ready: %s (bm25=%d chunks, dense=%d points, reranker=%s)",
        info.mode,
        info.bm25_chunks,
        info.dense_points,
        info.reranker,
    )
    retriever = HybridRetriever(qdrant_index=dense, bm25_index=bm25, reranker=reranker)
    return retriever, info
