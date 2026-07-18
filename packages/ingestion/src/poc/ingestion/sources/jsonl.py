from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from poc.core.errors import DomainError
from poc.core.models import Document


class IngestionError(DomainError):
    """Raised when document ingestion fails."""


class JsonlSource:
    """Iterates over Documents from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def iter(self) -> Iterator[Document]:
        try:
            with self._path.open(encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise IngestionError(f"Invalid JSON at line {lineno}: {exc}") from exc
                    yield Document(**data)
        except OSError as exc:
            raise IngestionError(f"Cannot open {self._path}: {exc}") from exc
