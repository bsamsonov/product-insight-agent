"""Unit tests for the LLM budget guard (S3.T4).

Uses FakeBudgetGuard — no running Redis instance required.
"""

from __future__ import annotations

import pytest
from poc.llm.budget import BudgetExceededError, FakeBudgetGuard

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_call_passes() -> None:
    guard = FakeBudgetGuard(per_request_usd=0.05, per_tenant_daily_usd=5.0)
    # Should not raise.
    await guard.check_and_increment("tenant-a", cost_usd=0.01)


@pytest.mark.asyncio
async def test_spend_accumulates_across_calls() -> None:
    guard = FakeBudgetGuard(per_request_usd=0.05, per_tenant_daily_usd=5.0)
    await guard.check_and_increment("tenant-a", cost_usd=0.01)
    await guard.check_and_increment("tenant-a", cost_usd=0.01)
    assert await guard.get_tenant_spend("tenant-a") == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_spend_is_per_tenant() -> None:
    guard = FakeBudgetGuard(per_request_usd=0.05, per_tenant_daily_usd=5.0)
    await guard.check_and_increment("tenant-a", cost_usd=0.04)
    await guard.check_and_increment("tenant-b", cost_usd=0.02)
    assert await guard.get_tenant_spend("tenant-a") == pytest.approx(0.04)
    assert await guard.get_tenant_spend("tenant-b") == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_get_tenant_spend_zero_if_no_calls() -> None:
    guard = FakeBudgetGuard()
    assert await guard.get_tenant_spend("unknown-tenant") == 0.0


# ---------------------------------------------------------------------------
# Per-request limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_request_exceeded_raises() -> None:
    guard = FakeBudgetGuard(per_request_usd=0.05, per_tenant_daily_usd=5.0)
    with pytest.raises(BudgetExceededError):
        await guard.check_and_increment(
            "tenant-a",
            cost_usd=0.04,
            request_total_usd=0.03,  # 0.03 + 0.04 = 0.07 > 0.05
        )


@pytest.mark.asyncio
async def test_per_request_tiny_limit_exceeded() -> None:
    """per_request_usd=0.001, cost=0.002 — must raise."""
    guard = FakeBudgetGuard(per_request_usd=0.001, per_tenant_daily_usd=5.0)
    with pytest.raises(BudgetExceededError):
        await guard.check_and_increment("tenant-a", cost_usd=0.002)


@pytest.mark.asyncio
async def test_per_request_exactly_at_limit_passes() -> None:
    """cost_usd == per_request_usd exactly (not strictly greater) — should pass."""
    guard = FakeBudgetGuard(per_request_usd=0.05, per_tenant_daily_usd=5.0)
    # 0.05 is NOT > 0.05
    await guard.check_and_increment("tenant-a", cost_usd=0.05)


# ---------------------------------------------------------------------------
# Per-tenant daily limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_tenant_daily_exceeded_raises() -> None:
    guard = FakeBudgetGuard(per_request_usd=10.0, per_tenant_daily_usd=0.10)
    await guard.check_and_increment("tenant-a", cost_usd=0.06)  # running: 0.06
    with pytest.raises(BudgetExceededError):
        await guard.check_and_increment("tenant-a", cost_usd=0.06)  # running: 0.12 > 0.10


@pytest.mark.asyncio
async def test_per_tenant_daily_exact_limit_passes() -> None:
    """Cumulative spend == daily limit exactly — should pass."""
    guard = FakeBudgetGuard(per_request_usd=10.0, per_tenant_daily_usd=0.10)
    # Two calls of 0.05 each; total == limit, not strictly greater.
    await guard.check_and_increment("tenant-a", cost_usd=0.05)
    await guard.check_and_increment("tenant-a", cost_usd=0.05)


@pytest.mark.asyncio
async def test_budget_exceeded_is_domain_error() -> None:
    """BudgetExceededError must be a subclass of DomainError."""
    from poc.core.errors import DomainError

    assert issubclass(BudgetExceededError, DomainError)
