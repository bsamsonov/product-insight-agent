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
    # Provider-side prompt-cache hits (S3.T3): number of input tokens served from the
    # provider's prompt cache. Populated from `usage.prompt_tokens_details.cached_tokens`
    # (OpenAI-compatible / Gemini implicit cache) or `usage.prompt_cache_hit_tokens`
    # (DeepSeek automatic context cache). 0 when the provider reports no cache activity.
    cached_input_tokens: int = 0


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
