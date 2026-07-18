from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from poc.core.models import Document
from poc.ingestion.normalizer import normalize
from poc.ingestion.sources.jsonl import JsonlSource


def _write_jsonl(records: list[dict]) -> Path:
    fd, name = tempfile.mkstemp(suffix=".jsonl")
    with open(fd, mode="w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return Path(name)


_BASE_DOC: dict = {
    "id": "r1",
    "source": "test",
    "lang": "en",
    "product_id": "p1",
    "region": "us",
    "created_at": None,
    "raw_text": "Great product!",
    "metadata": {"stars": 5},
}


class TestJsonlSource:
    def test_reads_all_documents(self) -> None:
        records = [{**_BASE_DOC, "id": f"r{i}"} for i in range(5)]
        path = _write_jsonl(records)
        try:
            docs = list(JsonlSource(path).iter())
            assert len(docs) == 5
            assert all(isinstance(d, Document) for d in docs)
        finally:
            path.unlink()

    def test_skips_blank_lines(self) -> None:
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        try:
            with path.open("w") as fh:
                fh.write("\n")
                fh.write(json.dumps(_BASE_DOC) + "\n")
                fh.write("\n")
            docs = list(JsonlSource(path).iter())
            assert len(docs) == 1
        finally:
            path.unlink()

    def test_document_fields_preserved(self) -> None:
        path = _write_jsonl([_BASE_DOC])
        try:
            doc = next(JsonlSource(path).iter())
            assert doc.id == "r1"
            assert doc.lang == "en"
            assert doc.raw_text == "Great product!"
            assert doc.metadata["stars"] == 5
        finally:
            path.unlink()

    def test_raises_on_missing_file(self) -> None:
        from poc.ingestion.sources.jsonl import IngestionError

        with pytest.raises(IngestionError, match="Cannot open"):
            list(JsonlSource("/nonexistent/path.jsonl").iter())

    def test_raises_on_invalid_json(self) -> None:
        from poc.ingestion.sources.jsonl import IngestionError

        path = Path(tempfile.mktemp(suffix=".jsonl"))
        try:
            path.write_text("not-json\n")
            with pytest.raises(IngestionError, match="Invalid JSON"):
                list(JsonlSource(path).iter())
        finally:
            path.unlink()


class TestNormalize:
    def test_trims_whitespace(self) -> None:
        doc = Document(**{**_BASE_DOC, "raw_text": "  hello world  "})
        result = normalize(doc)
        assert result.raw_text == "hello world"

    def test_nfc_unicode(self) -> None:
        # café with decomposed é (U+0065 + U+0301)
        decomposed = "cafe\u0301"
        doc = Document(**{**_BASE_DOC, "raw_text": decomposed})
        result = normalize(doc)
        assert result.raw_text == "caf\u00e9"

    def test_preserves_existing_lang(self) -> None:
        doc = Document(**{**_BASE_DOC, "lang": "fr", "raw_text": "Bonjour"})
        result = normalize(doc)
        assert result.lang == "fr"

    def test_detects_language_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "poc.ingestion.normalizer.detect",
            lambda text, **kwargs: {"lang": "en", "score": 0.99},
        )
        doc = Document(**{**_BASE_DOC, "lang": "", "raw_text": "This is an English sentence."})
        result = normalize(doc)
        assert result.lang == "en"

    def test_returns_new_document(self) -> None:
        doc = Document(**{**_BASE_DOC, "raw_text": "  test  "})
        result = normalize(doc)
        assert result is not doc
        assert doc.raw_text == "  test  "  # original unchanged
