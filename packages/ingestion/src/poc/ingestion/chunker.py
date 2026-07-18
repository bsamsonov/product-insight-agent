from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from poc.core.models import Document


@dataclass(frozen=True)
class Chunk:
    """A chunk of text derived from a Document."""

    id: str
    doc_id: str
    text: str
    position: int  # index within the document's chunk list
    metadata: dict[str, Any] = field(default_factory=dict)


def _token_len(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


class RecursiveTokenChunker:
    """Splits text into token-bounded chunks with overlap.

    Tries to split on paragraph → sentence → word boundaries to avoid
    cutting mid-sentence, then falls back to raw token slices.
    """

    # Split hierarchy: double-newline → newline → sentence-end → word
    _SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # noqa: RUF012

    def __init__(
        self,
        target_tokens: int = 400,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._target = target_tokens
        self._overlap = overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Splits a document's text into smaller parts and converts them into a list of chunk
        objects. Each chunk inherits the metadata of the original document, along with
        additional information such as position and source.
        """
        parts = self._split_text(doc.raw_text)
        chunks: list[Chunk] = []

        for idx, text in enumerate(parts):
            chunk_meta = {**doc.metadata, "source": doc.source}
            chunks.append(
                Chunk(
                    id=f"{doc.id}__c{idx}",
                    doc_id=doc.id,
                    text=text,
                    position=idx,
                    metadata=chunk_meta,
                )
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Splits a given text into smaller chunks based on token limits while ensuring chunks
        do not exceed a specified target length. The method encodes the text into tokens,
        processes them in segments, and ensures chunk boundaries do not split the meaningful
        content where possible.
        """
        tokens = self._enc.encode(text)
        if len(tokens) <= self._target:
            return [text] if text.strip() else []

        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + self._target, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)

            # Try to find a clean boundary in the decoded chunk
            if end < len(tokens):
                chunk_text = self._trim_to_boundary(chunk_text)

            if chunk_text.strip():
                chunks.append(chunk_text)

            # Advance by target minus overlap
            advance = max(1, self._target - self._overlap)
            start += advance

        return chunks

    def _trim_to_boundary(self, text: str) -> str:
        """Try to end at a sentence/word boundary."""
        for sep in [". ", "! ", "? ", "\n", " "]:
            idx = text.rfind(sep)
            if idx > len(text) // 2:
                return text[: idx + len(sep)].rstrip()
        return text


class StructuralMarkdownChunker:
    """Splits Markdown by headings, preserving heading breadcrumb in metadata."""

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.raw_text
        matches = list(self._HEADING_RE.finditer(text))

        if not matches:
            # No headings — treat as single chunk
            return [
                Chunk(
                    id=f"{doc.id}__c0",
                    doc_id=doc.id,
                    text=text,
                    position=0,
                    metadata={**doc.metadata, "breadcrumb": []},
                )
            ]

        chunks: list[Chunk] = []
        breadcrumb: list[str] = []  # tracks current heading hierarchy

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()

            # Update breadcrumb: pop deeper or same-level headings
            breadcrumb = breadcrumb[: level - 1]
            breadcrumb.append(heading)

            # Content between this heading and the next (or end of text)
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()

            section_text = (
                f"{'#' * level} {heading}\n\n{content}" if content else f"{'#' * level} {heading}"
            )

            chunks.append(
                Chunk(
                    id=f"{doc.id}__c{i}",
                    doc_id=doc.id,
                    text=section_text,
                    position=i,
                    metadata={
                        **doc.metadata,
                        "breadcrumb": list(breadcrumb),
                        "heading": heading,
                        "heading_level": level,
                    },
                )
            )

        return chunks
