"""S5.T1 integration — per-tenant Qdrant index isolation.

AC (S5.T1): two tenants, one index (collection) per tenant; queries for
tenantA must not return tenantB's data.

This drives a real ``QdrantIndex`` for two tenants over a single *shared*
in-memory Qdrant client (``location=":memory:"``). A shared client is what
makes the test meaningful: both tenants live in the same Qdrant instance and
isolation comes purely from the ``reviews__{tenant}`` collection split, not
from running against separate databases. A tiny deterministic fake embedder
keeps the test offline (no BgeM3 download, no network).
"""

from __future__ import annotations

import pytest
from poc.ingestion.chunker import Chunk

qdrant_client = pytest.importorskip("qdrant_client")
QdrantClient = qdrant_client.QdrantClient

from poc.retrieval.qdrant_index import QdrantIndex  # noqa: E402


class _FakeEmbedder:
    """Deterministic toy embedder: hashes tokens into a small bag-of-words vector."""

    dimension = 16

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            for tok in text.lower().split():
                vec[hash(tok) % self.dimension] += 1.0
            # Avoid all-zero vectors (cosine distance is undefined for them).
            if not any(vec):
                vec[0] = 1.0
            vectors.append(vec)
        return vectors


def _chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=doc, text=text, position=0, metadata={})


@pytest.fixture()
def shared_client():
    try:
        client = QdrantClient(location=":memory:")
    except Exception as exc:  # pragma: no cover - environment without local mode
        pytest.skip(f"Qdrant local mode unavailable: {exc}")
    yield client
    client.close()


def test_tenant_query_does_not_leak_across_tenants(shared_client) -> None:
    embedder = _FakeEmbedder()

    index_a = QdrantIndex(embedder=embedder, tenant="acme", client=shared_client)
    index_b = QdrantIndex(embedder=embedder, tenant="globex", client=shared_client)

    # Distinct corpora per tenant.
    index_a.upsert([_chunk("a1", "docA", "acme rocket booster telemetry report")])
    index_b.upsert([_chunk("b1", "docB", "globex coffee machine descaling guide")])

    # Two separate collections exist, named per tenant.
    names = {c.name for c in shared_client.get_collections().collections}
    assert "reviews__acme" in names
    assert "reviews__globex" in names

    # tenantA query returns only tenantA data.
    hits_a = index_a.search("rocket booster", top_k=5)
    doc_ids_a = {h.chunk.doc_id for h in hits_a}
    assert doc_ids_a == {"docA"}
    assert "docB" not in doc_ids_a

    # tenantB query returns only tenantB data — even querying tenantA's terms.
    hits_b = index_b.search("rocket booster", top_k=5)
    doc_ids_b = {h.chunk.doc_id for h in hits_b}
    assert "docA" not in doc_ids_b
    assert doc_ids_b <= {"docB"}
