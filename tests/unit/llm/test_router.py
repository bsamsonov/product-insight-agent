"""Unit tests for RoutedLLM — role-based routing with automatic fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from poc.llm.errors import LLMProviderError, LLMRateLimitError
from poc.llm.provider import LLMMessage, LLMResponse
from poc.llm.router import RoutedLLM

# Path to the real router.yaml shipped with the package
_ROUTER_YAML = (
    Path(__file__).parent.parent.parent.parent / "packages" / "llm" / "data" / "router.yaml"
)


def _make_response(model: str = "gemini-2.5-flash") -> LLMResponse:
    return LLMResponse(
        content="ok",
        model=model,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
        latency_ms=42,
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def routed_llm() -> RoutedLLM:
    """Return a RoutedLLM loaded from the real router.yaml (no network calls)."""
    return RoutedLLM(config_path=_ROUTER_YAML)


@pytest.fixture()
def messages() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="Hello")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoutedLLMConfig:
    def test_loads_all_roles(self, routed_llm: RoutedLLM) -> None:
        """All five roles must be parsed from router.yaml."""
        expected = {"classifier", "planner", "summarizer", "judge", "escalate"}
        assert set(routed_llm._routes) >= expected

    def test_classifier_primary_is_gemini_flash(self, routed_llm: RoutedLLM) -> None:
        cfg = routed_llm._routes["classifier"]
        assert cfg.primary.provider == "gemini"
        assert cfg.primary.model == "gemini-2.5-flash"
        assert cfg.primary.max_tokens == 256

    def test_classifier_fallback_is_groq(self, routed_llm: RoutedLLM) -> None:
        cfg = routed_llm._routes["classifier"]
        assert cfg.fallback.provider == "groq"

    def test_escalate_primary_is_deepseek_reasoner(self, routed_llm: RoutedLLM) -> None:
        cfg = routed_llm._routes["escalate"]
        assert cfg.primary.provider == "deepseek"
        assert cfg.primary.model == "deepseek-reasoner"

    async def test_unknown_role_raises_key_error(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        with pytest.raises(KeyError, match="unknown_role"):
            await routed_llm.route("unknown_role", messages)


class TestRoutedLLMPrimarySuccess:
    async def test_classifier_calls_gemini(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """Successful primary call returns response without touching fallback."""
        expected = _make_response("gemini-2.5-flash")

        gemini_mock = MagicMock()
        gemini_mock.complete = AsyncMock(return_value=expected)

        with patch.object(routed_llm, "_get_provider", return_value=gemini_mock) as mock_get:
            result = await routed_llm.route("classifier", messages)

        assert result.model == "gemini-2.5-flash"
        # _get_provider called exactly once (for primary); fallback never touched
        mock_get.assert_called_once_with("gemini")

    async def test_judge_uses_gemini_flash(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        expected = _make_response("gemini-2.5-flash")
        provider_mock = MagicMock()
        provider_mock.complete = AsyncMock(return_value=expected)

        with patch.object(routed_llm, "_get_provider", return_value=provider_mock):
            result = await routed_llm.route("judge", messages)

        assert result.model == "gemini-2.5-flash"


class TestRoutedLLMFallback:
    async def test_rate_limit_triggers_fallback(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """LLMRateLimitError from primary must switch to fallback provider."""
        fallback_response = _make_response("llama-3.3-70b-versatile")

        gemini_mock = MagicMock()
        gemini_mock.complete = AsyncMock(
            side_effect=LLMRateLimitError("rate limited", status_code=429)
        )

        groq_mock = MagicMock()
        groq_mock.complete = AsyncMock(return_value=fallback_response)

        def _get_provider(name: str) -> MagicMock:
            return gemini_mock if name == "gemini" else groq_mock

        with patch.object(routed_llm, "_get_provider", side_effect=_get_provider):
            result = await routed_llm.route("classifier", messages)

        assert result.model == "llama-3.3-70b-versatile"
        gemini_mock.complete.assert_awaited_once()
        groq_mock.complete.assert_awaited_once()

    async def test_provider_error_503_triggers_fallback(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """LLMProviderError with status_code=503 must also trigger fallback."""
        fallback_response = _make_response("llama-3.3-70b-versatile")

        gemini_mock = MagicMock()
        gemini_mock.complete = AsyncMock(
            side_effect=LLMProviderError("service unavailable", status_code=503)
        )

        groq_mock = MagicMock()
        groq_mock.complete = AsyncMock(return_value=fallback_response)

        def _get_provider(name: str) -> MagicMock:
            return gemini_mock if name == "gemini" else groq_mock

        with patch.object(routed_llm, "_get_provider", side_effect=_get_provider):
            result = await routed_llm.route("classifier", messages)

        assert result.model == "llama-3.3-70b-versatile"

    async def test_non_retriable_error_propagates(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """A 500 error (not 429/503) must NOT trigger fallback."""
        provider_mock = MagicMock()
        provider_mock.complete = AsyncMock(
            side_effect=LLMProviderError("internal error", status_code=500)
        )

        with (
            patch.object(routed_llm, "_get_provider", return_value=provider_mock),
            pytest.raises(LLMProviderError, match="internal error"),
        ):
            await routed_llm.route("classifier", messages)

        # Primary called once; fallback never called because error is non-retriable
        provider_mock.complete.assert_awaited_once()

    async def test_both_fail_reraises_fallback_error(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """If both primary and fallback fail, the fallback exception is re-raised."""
        gemini_mock = MagicMock()
        gemini_mock.complete = AsyncMock(
            side_effect=LLMRateLimitError("rate limited", status_code=429)
        )

        groq_mock = MagicMock()
        groq_mock.complete = AsyncMock(side_effect=LLMProviderError("groq down", status_code=500))

        def _get_provider(name: str) -> MagicMock:
            return gemini_mock if name == "gemini" else groq_mock

        with (
            patch.object(routed_llm, "_get_provider", side_effect=_get_provider),
            pytest.raises(LLMProviderError, match="groq down"),
        ):
            await routed_llm.route("classifier", messages)


class TestRoutedLLMEscalate:
    async def test_escalate_primary_is_deepseek(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """Escalate role should call deepseek as primary."""
        deepseek_response = _make_response("deepseek-reasoner")

        deepseek_mock = MagicMock()
        deepseek_mock.complete = AsyncMock(return_value=deepseek_response)

        with patch.object(routed_llm, "_get_provider", return_value=deepseek_mock) as mock_get:
            result = await routed_llm.route("escalate", messages)

        assert result.model == "deepseek-reasoner"
        mock_get.assert_called_once_with("deepseek")

    async def test_escalate_fallback_is_gemini_pro(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """Escalate fallback should be gemini-2.5-pro."""
        gemini_response = _make_response("gemini-2.5-pro")

        deepseek_mock = MagicMock()
        deepseek_mock.complete = AsyncMock(
            side_effect=LLMRateLimitError("deepseek quota", status_code=429)
        )

        gemini_mock = MagicMock()
        gemini_mock.complete = AsyncMock(return_value=gemini_response)

        def _get_provider(name: str) -> MagicMock:
            return deepseek_mock if name == "deepseek" else gemini_mock

        with patch.object(routed_llm, "_get_provider", side_effect=_get_provider):
            result = await routed_llm.route("escalate", messages)

        assert result.model == "gemini-2.5-pro"


class TestProviderCaching:
    async def test_same_provider_reused_across_calls(
        self, routed_llm: RoutedLLM, messages: list[LLMMessage]
    ) -> None:
        """Provider instances should be cached — from_env called at most once per name."""
        response = _make_response()
        provider_mock = MagicMock()
        provider_mock.complete = AsyncMock(return_value=response)

        with patch(
            "poc.llm.openai_compatible.OpenAICompatibleProvider.from_env",
            return_value=provider_mock,
        ) as from_env_mock:
            await routed_llm.route("classifier", messages)
            await routed_llm.route("classifier", messages)

        # from_env should be called only once for "gemini" across two route() calls
        assert from_env_mock.call_count == 1
