"""build_hybrid_retriever degrades per component instead of failing."""

from __future__ import annotations

from pathlib import Path

from poc.retrieval import factory

_SAMPLE = Path(__file__).resolve().parents[3] / "data" / "raw" / "sample_reviews.jsonl"


def test_bm25_only_when_qdrant_unreachable():
    retriever, info = factory.build_hybrid_retriever(
        _SAMPLE, qdrant_url="http://127.0.0.1:1", use_reranker=False
    )
    assert info.bm25_chunks > 0
    assert info.dense_points == 0
    assert info.mode == "bm25-only"
    assert info.reranker == "none"
    hits = retriever.retrieve("grip on wet rocks", top_k=3)
    assert hits and hits[0].chunk.id.startswith("B0TFRG8842")


def test_missing_corpus_gives_empty_index(tmp_path):
    retriever, info = factory.build_hybrid_retriever(
        tmp_path / "nope.jsonl", use_dense=False, use_reranker=False
    )
    assert info.bm25_chunks == 0
    assert retriever.retrieve("anything", top_k=3) == []


def test_bm25_limit_caps_documents():
    full = factory.build_bm25(_SAMPLE)
    capped = factory.build_bm25(_SAMPLE, limit=5)
    assert 0 < len(capped) < len(full)


def test_chunk_texts_resolves_ids_in_order():
    retriever, _ = factory.build_hybrid_retriever(_SAMPLE, use_dense=False, use_reranker=False)
    hits = retriever.retrieve("grip on wet rocks", top_k=2)
    ids = [h.chunk.id for h in hits]
    texts = retriever.chunk_texts([*ids, "unknown__0__c0"])
    assert texts == [h.chunk.text for h in hits]
