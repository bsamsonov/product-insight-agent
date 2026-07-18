from __future__ import annotations

from unittest.mock import MagicMock

from poc.ingestion.chunker import Chunk
from poc.retrieval.bm25 import BM25Index
from poc.retrieval.hybrid import HybridRetriever, _reciprocal_rank_fusion
from poc.retrieval.qdrant_index import ScoredChunk
from poc.retrieval.reranker import NoOpReranker


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            text=f"This is sample text about topic {i} with relevant content.",
            position=i,
        )
        for i in range(n)
    ]


def _make_scored(chunks: list[Chunk], scores: list[float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=s) for c, s in zip(chunks, scores, strict=False)]


class TestBM25Index:
    def test_build_and_search(self) -> None:
        chunks = _make_chunks(5)
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("sample text topic", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_empty_index_returns_empty(self) -> None:
        idx = BM25Index()
        results = idx.search("anything", top_k=5)
        assert results == []

    def test_len(self) -> None:
        chunks = _make_chunks(7)
        idx = BM25Index()
        idx.build(chunks)
        assert len(idx) == 7

    def test_relevant_term_scores_higher(self) -> None:
        chunks = [
            Chunk(id="a", doc_id="d1", text="running shoes comfort fit", position=0),
            Chunk(id="b", doc_id="d2", text="basketball jerseys team uniform", position=1),
            Chunk(id="c", doc_id="d3", text="running marathon trail shoes", position=2),
        ]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("running shoes", top_k=3)
        result_ids = [r.chunk.id for r in results]
        # 'a' and 'c' contain running and shoes, 'b' should rank lower
        assert "b" not in result_ids[:2] or len(result_ids) < 2


class TestRRF:
    def test_rrf_merges_two_lists(self) -> None:
        chunks = _make_chunks(6)
        bm25 = _make_scored(chunks[:3], [1.0, 0.8, 0.6])
        dense = _make_scored(chunks[3:], [0.9, 0.7, 0.5])
        merged = _reciprocal_rank_fusion(bm25, dense)
        assert len(merged) == 6

    def test_rrf_overlap_boosts_score(self) -> None:
        chunk = _make_chunks(1)[0]
        # Same chunk in both lists → should get higher combined score
        bm25 = [ScoredChunk(chunk=chunk, score=1.0)]
        dense = [ScoredChunk(chunk=chunk, score=1.0)]
        # Two separate chunks each in only one list
        other_chunks = _make_chunks(2)
        bm25_only = [ScoredChunk(chunk=other_chunks[0], score=0.9)]
        dense_only = [ScoredChunk(chunk=other_chunks[1], score=0.9)]

        merged = _reciprocal_rank_fusion(bm25 + bm25_only, dense + dense_only)
        top_chunk = merged[0]
        # The chunk in both lists should rank first
        assert top_chunk.chunk.id == chunk.id


class TestHybridRetriever:
    def test_retrieve_returns_scored_chunks(self) -> None:
        chunks = _make_chunks(10)

        # Mock QdrantIndex
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = _make_scored(chunks[:5], [0.9, 0.8, 0.7, 0.6, 0.5])

        bm25 = BM25Index()
        bm25.build(chunks)

        retriever = HybridRetriever(
            qdrant_index=mock_qdrant,
            bm25_index=bm25,
            reranker=NoOpReranker(),
        )

        results = retriever.retrieve("sample text", top_k=5)
        assert len(results) <= 5
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_retrieve_respects_top_k(self) -> None:
        chunks = _make_chunks(20)
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = _make_scored(
            chunks[:10], [0.9 - i * 0.05 for i in range(10)]
        )

        bm25 = BM25Index()
        bm25.build(chunks)

        retriever = HybridRetriever(
            qdrant_index=mock_qdrant,
            bm25_index=bm25,
            reranker=NoOpReranker(),
        )

        results = retriever.retrieve("topic", top_k=3)
        assert len(results) <= 3
