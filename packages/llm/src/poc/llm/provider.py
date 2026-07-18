from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal, Protocol

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: int
    finish_reason: str


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_format: type[BaseModel] | None = None,
        cache_control: bool = False,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]: ...
