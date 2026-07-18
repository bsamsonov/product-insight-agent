from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from poc.ingestion.chunker import Chunk
from poc.retrieval.embedder import Embedder
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class QdrantIndex:
    """Manages a Qdrant collection for a tenant and provides upsert/search."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        qdrant_url: str = "http://localhost:6333",
        tenant: str = "default",
        recreate: bool = False,
        client: QdrantClient | None = None,
    ) -> None:
        self._embedder = embedder
        self._tenant = tenant
        self._collection = f"reviews__{tenant}"
        # ``client`` lets callers inject a shared/in-memory QdrantClient so that
        # multiple tenants can target one Qdrant instance (different collections)
        # — required to test per-tenant isolation without a live server.
        self._client = client if client is not None else QdrantClient(url=qdrant_url)
        self._ensure_collection(recreate=recreate)

    # ── internal ──────────────────────────────────────────────────────────────

    def _ensure_collection(self, *, recreate: bool = False) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if recreate and self._collection in existing:
            # Needed when the source schema changes (e.g. new payload fields):
            # old points would otherwise linger with stale/missing payload.
            _log.info("Dropping existing collection '%s'", self._collection)
            self._client.delete_collection(self._collection)
            existing.discard(self._collection)
        if self._collection not in existing:
            _log.info("Creating Qdrant collection '%s'", self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    # ── public ────────────────────────────────────────────────────────────────

    def upsert(
        self, chunks: list[Chunk], *, batch_size: int = 64, skip_existing: bool = False
    ) -> None:
        """Embed and upsert chunks in batches.

        With ``skip_existing=True`` each batch is checked against the collection first
        and chunks whose point already exists are dropped *before* embedding — this makes
        a re-run resumable: the expensive BgeM3 embedding is skipped for already-indexed
        chunks (point ids are deterministic via :meth:`_stable_id`).
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if skip_existing:
                batch = self._filter_existing(batch)
                if not batch:
                    continue
            texts = [c.text for c in batch]
            vectors = self._embedder.embed_batch(texts)
            points = [
                PointStruct(
                    id=self._stable_id(c.id),
                    vector=vec,
                    payload={
                        "doc_id": c.doc_id,
                        "chunk_id": c.id,
                        "text": c.text,
                        "position": c.position,
                        **{k: v for k, v in c.metadata.items() if _is_scalar(v)},
                    },
                )
                for c, vec in zip(batch, vectors, strict=False)
            ]
            self._client.upsert(collection_name=self._collection, points=points)
            _log.debug("Upserted %d chunks to '%s'", len(points), self._collection)

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Dense vector search with optional payload filters."""
        query_vec = self._embedder.embed_batch([query_text])[0]

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
                if v is not None
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[ScoredChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id", str(hit.id)),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                position=payload.get("position", 0),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in ("chunk_id", "doc_id", "text", "position")
                },
            )
            results.append(ScoredChunk(chunk=chunk, score=hit.score))

        return results

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def _filter_existing(self, batch: list[Chunk]) -> list[Chunk]:
        """Drop chunks whose point already exists in the collection (resume support)."""
        ids = [self._stable_id(c.id) for c in batch]
        found = self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        existing = {point.id for point in found}
        return [c for c, pid in zip(batch, ids, strict=False) if pid not in existing]

    @staticmethod
    def _stable_id(chunk_id: str) -> int:
        # sha256, not builtin hash(): hash() of str is salted per-process
        # (PYTHONHASHSEED), so re-indexing the same chunk would create a new
        # point instead of overwriting it. Mod 2**53 keeps it JSON-safe.
        return int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**53)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None
