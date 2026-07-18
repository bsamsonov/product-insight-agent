"""Unit tests for the LLM response cache (S3.T3).

Uses FakeRedisCache — no running Redis instance required.
"""

from __future__ import annotations

import pytest
from poc.llm.cache import FakeRedisCache, cached
from poc.llm.provider import LLMMessage, LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MSGS = [LLMMessage(role="user", content="Hello")]


def _response(content: str = "Hi") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100,
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# FakeRedisCache direct tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_miss_returns_none() -> None:
    cache = FakeRedisCache()
    key = cache._make_key("prov", "m1", _MSGS, 0.0, 100)
    assert await cache.get(key) is None


@pytest.mark.asyncio
async def test_cache_hit_after_set() -> None:
    cache = FakeRedisCache()
    resp = _response("cached")
    key = cache._make_key("prov", "m1", _MSGS, 0.0, 100)
    await cache.set(key, resp)
    result = await cache.get(key)
    assert result is not None
    assert result.content == "cached"


@pytest.mark.asyncio
async def test_different_models_produce_different_keys() -> None:
    cache = FakeRedisCache()
    key1 = cache._make_key("prov", "model-a", _MSGS, 0.0, 100)
    key2 = cache._make_key("prov", "model-b", _MSGS, 0.0, 100)
    assert key1 != key2


@pytest.mark.asyncio
async def test_different_temperatures_produce_different_keys() -> None:
    cache = FakeRedisCache()
    key1 = cache._make_key("prov", "m1", _MSGS, 0.0, 100)
    key2 = cache._make_key("prov", "m1", _MSGS, 0.7, 100)
    assert key1 != key2


@pytest.mark.asyncio
async def test_different_messages_produce_different_keys() -> None:
    cache = FakeRedisCache()
    msgs_a = [LLMMessage(role="user", content="Hello")]
    msgs_b = [LLMMessage(role="user", content="Goodbye")]
    key1 = cache._make_key("prov", "m1", msgs_a, 0.0, 100)
    key2 = cache._make_key("prov", "m1", msgs_b, 0.0, 100)
    assert key1 != key2


# ---------------------------------------------------------------------------
# @cached decorator tests
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal provider stub for decorator tests."""

    name = "fake-provider"

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> LLMResponse:
        self.call_count += 1
        return _response(f"call#{self.call_count}")


@pytest.mark.asyncio
async def test_cached_decorator_miss_then_hit() -> None:
    """First call reaches the provider; second call returns cached response."""
    cache = FakeRedisCache()
    provider = _FakeProvider()

    # Patch the method with the decorator at runtime.
    provider.complete = cached(cache)(provider.complete)  # type: ignore[method-assign]

    resp1 = await provider.complete(_MSGS, model="m", max_tokens=100)
    resp2 = await provider.complete(_MSGS, model="m", max_tokens=100)

    assert provider.call_count == 1, "Provider should only be called once"
    assert resp1.content == resp2.content == "call#1"


@pytest.mark.asyncio
async def test_cached_decorator_different_model_no_hit() -> None:
    """Different model → cache miss → provider is called again."""
    cache = FakeRedisCache()
    provider = _FakeProvider()
    provider.complete = cached(cache)(provider.complete)  # type: ignore[method-assign]

    await provider.complete(_MSGS, model="model-a", max_tokens=100)
    await provider.complete(_MSGS, model="model-b", max_tokens=100)

    assert provider.call_count == 2
