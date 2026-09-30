"""BM25 and dense retrieval must use the same corpus (index records it, API reads it)."""

from __future__ import annotations

import logging
from pathlib import Path

from poc.retrieval.factory import corpus_label, corpus_matches, select_corpus
from poc.retrieval.qdrant_index import QdrantIndex, read_corpus_source
from qdrant_client import QdrantClient


class _FakeEmbedder:
    dimension = 4

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _files(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    full, sample = raw / "reviews.jsonl", raw / "sample_reviews.jsonl"
    full.write_text("{}\n")
    sample.write_text("{}\n")
    return full, sample


def test_corpus_label_is_repo_relative(tmp_path):
    full, _ = _files(tmp_path)
    assert corpus_label(full, tmp_path) == "data/raw/reviews.jsonl"
    outside = tmp_path.parent / "elsewhere.jsonl"
    assert corpus_label(outside, tmp_path) == outside.resolve().as_posix()


def test_corpus_matches(tmp_path):
    full, sample = _files(tmp_path)
    assert corpus_matches(full, "data/raw/reviews.jsonl")
    assert not corpus_matches(sample, "data/raw/reviews.jsonl")
    assert corpus_matches(full, full.resolve().as_posix())


def test_indexed_sample_wins_over_downloaded_full_dataset(tmp_path):
    """The bug this guards: sample indexed in Qdrant, full dataset on disk → BM25 on full."""
    full, sample = _files(tmp_path)
    chosen = select_corpus(
        root=tmp_path,
        override=None,
        indexed="data/raw/sample_reviews.jsonl",
        fallbacks=[full, sample],
    )
    assert chosen == tmp_path / "data/raw/sample_reviews.jsonl"


def test_override_wins_but_warns_on_mismatch(tmp_path, caplog):
    full, sample = _files(tmp_path)
    with caplog.at_level(logging.WARNING):
        chosen = select_corpus(
            root=tmp_path,
            override="data/raw/reviews.jsonl",
            indexed="data/raw/sample_reviews.jsonl",
            fallbacks=[full, sample],
        )
    assert chosen == tmp_path / "data/raw/reviews.jsonl"
    assert "different corpora" in caplog.text


def test_missing_indexed_file_falls_back_with_warning(tmp_path, caplog):
    full, sample = _files(tmp_path)
    with caplog.at_level(logging.WARNING):
        chosen = select_corpus(
            root=tmp_path, override=None, indexed="data/raw/gone.jsonl", fallbacks=[full, sample]
        )
    assert chosen == full
    assert "not on disk" in caplog.text


def test_no_record_uses_first_existing_fallback(tmp_path):
    _, sample = _files(tmp_path)
    missing = tmp_path / "data" / "raw" / "nope.jsonl"
    assert (
        select_corpus(root=tmp_path, override=None, indexed=None, fallbacks=[missing, sample])
        == sample
    )


def test_collection_metadata_roundtrip():
    client = QdrantClient(":memory:")
    index = QdrantIndex(embedder=_FakeEmbedder(), client=client, tenant="t1")
    assert index.corpus_source() is None
    index.set_corpus_source("data/raw/sample_reviews.jsonl")
    assert index.corpus_source() == "data/raw/sample_reviews.jsonl"
    assert read_corpus_source(client, "reviews__t1") == "data/raw/sample_reviews.jsonl"


def test_read_corpus_source_of_missing_collection_is_none():
    assert read_corpus_source(QdrantClient(":memory:"), "reviews__nope") is None
