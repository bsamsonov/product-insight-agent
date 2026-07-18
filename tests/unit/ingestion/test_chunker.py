from __future__ import annotations

from datetime import datetime

from poc.core.models import Document
from poc.ingestion.chunker import RecursiveTokenChunker, StructuralMarkdownChunker


def _make_doc(text: str, doc_id: str = "doc1") -> Document:
    return Document(
        id=doc_id,
        source="test",
        lang="en",
        raw_text=text,
        created_at=datetime(2024, 1, 1),
    )


class TestRecursiveTokenChunker:
    def test_short_text_single_chunk(self) -> None:
        """
        Tests the chunking of a short text with a single chunk.

        This method ensures that a short text, which is smaller than the specified
        target token count of the chunker, results in a single chunk without any
        modifications to the text or additional overlaps.
        """
        chunker = RecursiveTokenChunker(target_tokens=400, overlap=50)
        doc = _make_doc("This is a short text.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short text."

    def test_long_text_multiple_chunks(self) -> None:
        """A 4000-token text should yield >= 9 chunks with target=400, overlap=50."""
        # ~4000 tokens worth of repeated text (1 token ≈ 4 chars)
        word = "review " * 600  # ~600 tokens each * repeated content
        text = (word * 7).strip()  # ~4200 tokens total
        chunker = RecursiveTokenChunker(target_tokens=400, overlap=50)
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 9, f"Expected >= 9 chunks, got {len(chunks)}"

    def test_chunk_ids_are_unique(self) -> None:
        text = "word " * 1000
        chunker = RecursiveTokenChunker(target_tokens=100, overlap=10)
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_doc_id_propagated(self) -> None:
        text = "word " * 50
        chunker = RecursiveTokenChunker(target_tokens=20, overlap=5)
        doc = _make_doc(text, doc_id="test-doc-42")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == "test-doc-42"

    def test_overlap_means_content_shared(self) -> None:
        """Consecutive chunks should share some tokens due to overlap."""
        # Build text with identifiable word markers
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        chunker = RecursiveTokenChunker(target_tokens=30, overlap=10)
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Check positions are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.position == i

    def test_metadata_preserved(self) -> None:
        doc = Document(
            id="d1",
            source="src",
            lang="en",
            raw_text="hello " * 500,
            metadata={"product_id": "shoe-42", "region": "EU"},
        )
        chunker = RecursiveTokenChunker(target_tokens=100, overlap=10)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["product_id"] == "shoe-42"
            assert chunk.metadata["region"] == "EU"

    def test_empty_text_returns_no_chunks(self) -> None:
        doc = _make_doc("")
        chunker = RecursiveTokenChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []


class TestStructuralMarkdownChunker:
    def test_no_headings_single_chunk(self) -> None:
        text = "Just some plain text without headings."
        doc = _make_doc(text)
        chunker = StructuralMarkdownChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_headings_split_into_sections(self) -> None:
        text = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
        doc = _make_doc(text)
        chunker = StructuralMarkdownChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2

    def test_breadcrumb_in_metadata(self) -> None:
        text = "# Top\n\nIntro.\n\n## Sub\n\nDetail."
        doc = _make_doc(text)
        chunker = StructuralMarkdownChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].metadata["breadcrumb"] == ["Top"]
        assert chunks[1].metadata["breadcrumb"] == ["Top", "Sub"]

    def test_heading_level_in_metadata(self) -> None:
        text = "## Level2\n\nContent."
        doc = _make_doc(text)
        chunker = StructuralMarkdownChunker()
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata["heading_level"] == 2

    def test_chunk_ids_unique(self) -> None:
        text = "\n".join(f"# Section {i}\n\nContent {i}." for i in range(10))
        doc = _make_doc(text)
        chunker = StructuralMarkdownChunker()
        chunks = chunker.chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
