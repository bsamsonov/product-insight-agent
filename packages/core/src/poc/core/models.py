from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    lang: str
    product_id: str | None = None
    region: str | None = None
    created_at: datetime | None = None
    raw_text: str
    metadata: dict[str, Any] = {}


class Citation(BaseModel):
    chunk_id: str
    text_excerpt: str
    score: float = 0.0


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation] = []
    used_chunks: list[str] = []  # chunk IDs used in context
    model: str = ""
    cost_usd: float | None = None
    latency_ms: int = 0


class Question(BaseModel):
    text: str
    filters: dict[str, Any] = {}
    tenant: str = "default"
