"""Unit tests for provider-side prompt-cache token extraction (S3.T3)."""

from __future__ import annotations

from types import SimpleNamespace

from poc.llm.openai_compatible import OpenAICompatibleProvider

_extract = OpenAICompatibleProvider._extract_cached_tokens


def test_none_usage_returns_zero() -> None:
    assert _extract(None) == 0


def test_openai_style_prompt_tokens_details() -> None:
    """OpenAI-compatible (incl. Gemini implicit cache): prompt_tokens_details.cached_tokens."""
    usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=1024))
    assert _extract(usage) == 1024


def test_deepseek_style_prompt_cache_hit_tokens() -> None:
    """DeepSeek automatic context cache: usage.prompt_cache_hit_tokens."""
    usage = SimpleNamespace(prompt_tokens_details=None, prompt_cache_hit_tokens=512)
    assert _extract(usage) == 512


def test_details_take_precedence_over_deepseek_field() -> None:
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=100),
        prompt_cache_hit_tokens=999,
    )
    assert _extract(usage) == 100


def test_no_cache_fields_returns_zero() -> None:
    """Providers without cache reporting (e.g. Ollama) yield 0, not an error."""
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    assert _extract(usage) == 0


def test_none_cached_tokens_value_returns_zero() -> None:
    usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=None))
    assert _extract(usage) == 0
