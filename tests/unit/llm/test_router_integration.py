"""Unit tests for RoutedLLM integration seams (S3.T3/S3.T4).

Covers the composition added in the cost/routing sprint: response cache in front
of the providers, budget charging after each real call, and the request-scoped
cost context (contextvars) that ties tenant + running total to every call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from poc.llm.budget import BudgetExceededError, FakeBudgetGuard
from poc.llm.cache import FakeRedisCache
from poc.llm.context import current_budget_context, request_budget
from poc.llm.provider import LLMMessage, LLMResponse
from poc.llm.router import RoutedLLM

_ROUTER_YAML = (
    Path(__file__).parent.parent.parent.parent / "packages" / "llm" / "data" / "router.yaml"
)


def _make_response(cost_usd: float = 0.01, model: str = "gemini-2.5-flash") -> LLMResponse:
    return LLMResponse(
        content="ok",
        model=model,
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost_usd,
        latency_ms=42,
        finish_reason="stop",
    )


@pytest.fixture()
def messages() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="Hello")]


def _routed(cache=None, guard=None) -> RoutedLLM:
    return RoutedLLM(config_path=_ROUTER_YAML, cache=cache, budget_guard=guard)


def _mock_provider(response: LLMResponse) -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=response)
    return provider


# ---------------------------------------------------------------------------
# Response cache (S3.T3)
# ---------------------------------------------------------------------------


class TestRouterCache:
    async def test_second_identical_call_served_from_cache(self, messages) -> None:
        cache = FakeRedisCache()
        router = _routed(cache=cache)
        provider = _mock_provider(_make_response())

        with patch.object(router, "_get_provider", return_value=provider):
            first = await router.route("classifier", messages)
            second = await router.route("classifier", messages)

        assert first.content == second.content == "ok"
        provider.complete.assert_awaited_once()  # second call never dialed out

    async def test_cache_hit_skips_budget_charge(self, messages) -> None:
        cache = FakeRedisCache()
        guard = FakeBudgetGuard(per_request_usd=1.0, per_tenant_daily_usd=10.0)
        router = _routed(cache=cache, guard=guard)
        provider = _mock_provider(_make_response(cost_usd=0.01))

        with patch.object(router, "_get_provider", return_value=provider):
            await router.route("classifier", messages)
            spend_after_first = await guard.get_tenant_spend("default")
            await router.route("classifier", messages)
            spend_after_second = await guard.get_tenant_spend("default")

        assert spend_after_first == pytest.approx(0.01)
        assert spend_after_second == pytest.approx(0.01)  # cache hit charged nothing

    async def test_different_roles_do_not_share_cache_entries(self, messages) -> None:
        cache = FakeRedisCache()
        router = _routed(cache=cache)
        provider = _mock_provider(_make_response())

        with patch.object(router, "_get_provider", return_value=provider):
            await router.route("classifier", messages)
            await router.route("planner", messages)

        assert provider.complete.await_count == 2  # different primary model → miss


# ---------------------------------------------------------------------------
# Budget caps (S3.T4)
# ---------------------------------------------------------------------------


class TestRouterBudget:
    async def test_budget_charged_after_call(self, messages) -> None:
        guard = FakeBudgetGuard(per_request_usd=1.0, per_tenant_daily_usd=10.0)
        router = _routed(guard=guard)
        provider = _mock_provider(_make_response(cost_usd=0.02))

        with patch.object(router, "_get_provider", return_value=provider):
            await router.route("classifier", messages)

        assert await guard.get_tenant_spend("default") == pytest.approx(0.02)

    async def test_tiny_per_request_limit_raises_after_first_call(self, messages) -> None:
        """Plan AC: per_request_usd=0.001 → BudgetExceededError after first LLM call."""
        guard = FakeBudgetGuard(per_request_usd=0.001, per_tenant_daily_usd=10.0)
        router = _routed(guard=guard)
        provider = _mock_provider(_make_response(cost_usd=0.01))

        with (
            patch.object(router, "_get_provider", return_value=provider),
            pytest.raises(BudgetExceededError, match="Per-request"),
        ):
            await router.route("classifier", messages)

        provider.complete.assert_awaited_once()  # the call happened, then the stop

    async def test_tenant_from_request_context_is_charged(self, messages) -> None:
        guard = FakeBudgetGuard(per_request_usd=1.0, per_tenant_daily_usd=10.0)
        router = _routed(guard=guard)
        provider = _mock_provider(_make_response(cost_usd=0.03))

        with (
            patch.object(router, "_get_provider", return_value=provider),
            request_budget(tenant="acme"),
        ):
            await router.route("classifier", messages)

        assert await guard.get_tenant_spend("acme") == pytest.approx(0.03)
        assert await guard.get_tenant_spend("default") == 0.0

    async def test_running_total_accumulates_toward_per_request_cap(self, messages) -> None:
        """Two calls of $0.03 each breach a $0.05 per-request cap on the second call."""
        guard = FakeBudgetGuard(per_request_usd=0.05, per_tenant_daily_usd=10.0)
        router = _routed(guard=guard)
        provider = _mock_provider(_make_response(cost_usd=0.03))

        with (
            patch.object(router, "_get_provider", return_value=provider),
            request_budget(tenant="acme") as ctx,
        ):
            await router.route("classifier", messages)
            assert ctx.total_usd == pytest.approx(0.03)
            with pytest.raises(BudgetExceededError, match="Per-request"):
                await router.route("classifier", messages)


# ---------------------------------------------------------------------------
# Request context (contextvars)
# ---------------------------------------------------------------------------


class TestRequestBudgetContext:
    def test_no_context_outside_scope(self) -> None:
        assert current_budget_context() is None

    def test_context_visible_inside_scope_and_reset_after(self) -> None:
        with request_budget(tenant="t1") as ctx:
            assert current_budget_context() is ctx
            assert ctx.tenant == "t1"
            assert ctx.total_usd == 0.0
        assert current_budget_context() is None

    async def test_cost_accumulates_without_guard(self, messages) -> None:
        """Even with no BudgetGuard, the context still tracks the request total."""
        router = _routed()
        provider = _mock_provider(_make_response(cost_usd=0.02))

        with (
            patch.object(router, "_get_provider", return_value=provider),
            request_budget() as ctx,
        ):
            await router.route("classifier", messages)
            await router.route("judge", messages)

        assert ctx.total_usd == pytest.approx(0.04)
