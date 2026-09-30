"""Build the hybrid retriever (BM25 + Qdrant dense + reranker) with graceful degradation.

One place that the API and ``scripts/run_eval.py`` share, so both run the same retrieval
stack. Each component falls back independently:

- **BM25** is built in memory from the corpus JSONL (always, whole corpus by default).
- **Dense** search is used only when Qdrant is reachable *and* the tenant collection
  already holds points — the heavy BGE-M3 embedder is not even loaded otherwise, and an
  empty collection is never created as a side effect.
- **Reranker** is the local cross-encoder when it loads, otherwise a no-op.

Corpus consistency: ``scripts/index.py`` records the corpus file in the Qdrant collection
metadata (``corpus_source``). :func:`select_corpus` uses it so that BM25 is built from the
same file as the dense index, and :func:`build_hybrid_retriever` warns when they differ.
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
    dense_corpus: str | None = None  # corpus file recorded in the Qdrant collection

    @property
    def mode(self) -> str:
        return "hybrid" if self.dense_points else "bm25-only"


def corpus_label(path: Path, root: Path) -> str:
    """Portable corpus id: repo-relative POSIX path when under *root*, else absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def corpus_matches(path: Path, label: str) -> bool:
    """True when *path* is the file identified by *label* (see :func:`corpus_label`)."""
    resolved = path.resolve().as_posix()
    if Path(label).is_absolute():
        return resolved == Path(label).resolve().as_posix()
    return resolved == label or resolved.endswith("/" + label)


def indexed_corpus(qdrant_url: str, tenant: str) -> str | None:
    """Corpus file recorded in the tenant collection, or None (no Qdrant / no record)."""
    try:
        from poc.retrieval.qdrant_index import read_corpus_source
        from qdrant_client import QdrantClient

        return read_corpus_source(QdrantClient(url=qdrant_url, timeout=5), f"reviews__{tenant}")
    except Exception as exc:
        _log.debug("Could not read indexed corpus from %s: %s", qdrant_url, exc)
        return None


def select_corpus(
    *,
    root: Path,
    override: str | None,
    indexed: str | None,
    fallbacks: list[Path],
) -> Path | None:
    """Pick the BM25 corpus: explicit override > file the dense index was built from >
    first existing fallback. Logs a warning whenever BM25 and dense may diverge."""

    def _abs(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else root / p

    if override:
        chosen = _abs(override)
        if indexed and not corpus_matches(chosen, indexed):
            _log.warning(
                "POC_CORPUS_PATH=%s but the Qdrant collection was indexed from %s — "
                "BM25 and dense retrieval use different corpora",
                override,
                indexed,
            )
        return chosen
    if indexed:
        chosen = _abs(indexed)
        if chosen.exists():
            return chosen
        _log.warning(
            "Qdrant collection was indexed from %s, which is not on disk — "
            "falling back to the first available corpus file",
            indexed,
        )
    return next((p for p in fallbacks if _abs(p).exists()), None)


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


def _dense_index(qdrant_url: str, tenant: str) -> tuple[Any, int, str | None]:
    """Return (dense index, point count, recorded corpus), or a no-op, 0, None."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=5)
        collection = f"reviews__{tenant}"
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            _log.warning("Qdrant collection %s not found — dense search off", collection)
            return _NoOpDense(), 0, None
        points = client.count(collection).count
        if points == 0:
            _log.warning("Qdrant collection %s is empty — dense search off", collection)
            return _NoOpDense(), 0, None

        from poc.retrieval.bge_embedder import BgeM3Embedder
        from poc.retrieval.qdrant_index import QdrantIndex, read_corpus_source

        source = read_corpus_source(client, collection)
        index = QdrantIndex(embedder=BgeM3Embedder(), client=client, tenant=tenant)
        return index, points, source
    except Exception as exc:
        _log.warning("Qdrant unavailable at %s (%s) — dense search off", qdrant_url, exc)
        return _NoOpDense(), 0, None


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
    dense, points, dense_corpus = (
        _dense_index(qdrant_url, tenant) if use_dense else (_NoOpDense(), 0, None)
    )
    if points:
        if dense_corpus is None:
            _log.warning(
                "Qdrant collection reviews__%s does not record its corpus — re-run "
                "scripts/index.py so BM25 (%s) and dense search are known to match",
                tenant,
                corpus_path,
            )
        elif not corpus_matches(corpus_path, dense_corpus):
            _log.warning(
                "Corpus mismatch: BM25 uses %s but Qdrant was indexed from %s — "
                "hybrid results mix two corpora",
                corpus_path,
                dense_corpus,
            )
    reranker, reranker_name = _reranker(use_reranker)
    info = RetrieverInfo(
        bm25_chunks=len(bm25),
        dense_points=points,
        reranker=reranker_name,
        dense_corpus=dense_corpus,
    )
    _log.info(
        "Retriever ready: %s (bm25=%d chunks, dense=%d points, reranker=%s)",
        info.mode,
        info.bm25_chunks,
        info.dense_points,
        info.reranker,
    )
    retriever = HybridRetriever(qdrant_index=dense, bm25_index=bm25, reranker=reranker)
    return retriever, info
