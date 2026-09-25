"""Plan filters reach Qdrant as valid conditions, and dense failures keep BM25 hits."""

from __future__ import annotations

from poc.ingestion.chunker import Chunk
from poc.retrieval.bm25 import BM25Index
from poc.retrieval.hybrid import HybridRetriever
from poc.retrieval.qdrant_index import QdrantIndex
from poc.retrieval.reranker import NoOpReranker
from qdrant_client import QdrantClient


class _FakeEmbedder:
    dimension = 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


def _chunk(cid: str, rating: float, title: str) -> Chunk:
    return Chunk(
        id=cid,
        doc_id=cid,
        text=f"review {cid} line tangles and breaks",
        position=0,
        metadata={"rating": rating, "product_title": title},
    )


def _index() -> QdrantIndex:
    index = QdrantIndex(embedder=_FakeEmbedder(), client=QdrantClient(":memory:"))
    index.upsert([_chunk("a__0__c0", 2.0, "Line"), _chunk("a__1__c0", 5.0, "Line")])
    return index


def test_float_rating_filter_matches_instead_of_raising():
    hits = _index().search("tangles", top_k=5, filters={"rating": 2.0})
    assert [h.chunk.id for h in hits] == ["a__0__c0"]


def test_int_rating_matches_float_payload():
    hits = _index().search("tangles", top_k=5, filters={"rating": 5})
    assert [h.chunk.id for h in hits] == ["a__1__c0"]


def test_string_filter_still_exact_match():
    assert _index().search("x", top_k=5, filters={"product_title": "Other"}) == []


class _BrokenDense:
    def search(self, *args, **kwargs):
        raise RuntimeError("bad filter")


def test_dense_failure_keeps_bm25_hits():
    bm25 = BM25Index()
    other = Chunk(id="b__0__c0", doc_id="b", text="smooth reel", position=0, metadata={})
    bm25.build([_chunk("a__0__c0", 2.0, "Line"), other])
    retriever = HybridRetriever(
        qdrant_index=_BrokenDense(), bm25_index=bm25, reranker=NoOpReranker()
    )
    hits = retriever.retrieve("tangles", top_k=5, filters={"rating": 2.0})
    assert [h.chunk.id for h in hits] == ["a__0__c0"]
