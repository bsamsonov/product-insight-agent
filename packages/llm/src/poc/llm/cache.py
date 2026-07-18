"""Redis-backed response cache for LLM calls (S3.T3)."""

from __future__ import annotations

import functools
import hashlib
import json
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis
from poc.llm.provider import LLMMessage, LLMResponse

if TYPE_CHECKING:
    from collections.abc import Callable


class RedisResponseCache:
    """SHA256-keyed Redis cache for LLM responses.

    The cache key is derived deterministically from provider name, model,
    messages, temperature, and max_tokens — same inputs always produce the
    same key.  TTL defaults to 1 h; set ttl_s=0 to disable expiry.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_s: int = 3600,
    ) -> None:
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl_s = ttl_s

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    def _make_key(
        self,
        provider: str,
        model: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return a deterministic SHA-256 hex key for the given call parameters."""
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return "llm:cache:" + hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def get(self, key: str) -> LLMResponse | None:
        """Return the cached LLMResponse for *key*, or None on a miss."""
        raw = await self._client.get(key)
        if raw is None:
            return None
        return LLMResponse.model_validate_json(raw)

    async def set(self, key: str, response: LLMResponse) -> None:
        """Store *response* under *key* with the configured TTL."""
        serialized = response.model_dump_json()
        if self._ttl_s > 0:
            await self._client.setex(key, self._ttl_s, serialized)
        else:
            await self._client.set(key, serialized)


# ---------------------------------------------------------------------------
# Decorator factory
# ---------------------------------------------------------------------------


def cached(cache: RedisResponseCache | FakeRedisCache) -> Callable[..., Any]:
    """Decorator factory: wraps an async LLM complete() call with Redis caching.

    Usage::

        @cached(my_cache)
        async def complete(self, messages, *, model, max_tokens, temperature=0.0, ...):
            ...

    The decorator inspects the function arguments by name to build the cache
    key.  It requires that the decorated function belongs to an object with a
    ``name`` attribute that identifies the provider (e.g. "openai-compatible").
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> LLMResponse:
            provider_name = getattr(self_obj, "name", type(self_obj).__name__)

            # The first positional argument after self is `messages`.
            if args:
                messages: list[LLMMessage] = args[0]
            else:
                messages = kwargs.get("messages", [])

            model: str = kwargs.get("model", "")
            temperature: float = float(kwargs.get("temperature", 0.0))
            max_tokens: int = int(kwargs.get("max_tokens", 0))

            key = cache._make_key(provider_name, model, messages, temperature, max_tokens)

            cached_response = await cache.get(key)
            if cached_response is not None:
                return cached_response

            response: LLMResponse = await fn(self_obj, *args, **kwargs)
            await cache.set(key, response)
            return response

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# In-memory test double
# ---------------------------------------------------------------------------


class FakeRedisCache:
    """In-memory drop-in for testing without a running Redis instance.

    Implements the same interface as RedisResponseCache.  _make_key is
    identical so that test assertions can verify key construction.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    # Delegate to the real implementation for deterministic key generation.
    _make_key = RedisResponseCache._make_key  # type: ignore[assignment]

    async def get(self, key: str) -> LLMResponse | None:
        raw = self._store.get(key)
        if raw is None:
            return None
        return LLMResponse.model_validate_json(raw)

    async def set(self, key: str, response: LLMResponse) -> None:
        self._store[key] = response.model_dump_json()
