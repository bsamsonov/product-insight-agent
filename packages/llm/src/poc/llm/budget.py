"""Per-request and per-tenant daily budget enforcement (S3.T4)."""

from __future__ import annotations

from datetime import date

import redis.asyncio as aioredis
from poc.core.errors import DomainError


class BudgetExceededError(DomainError):
    """Raised when the per-request or per-tenant daily budget is exceeded."""


class BudgetGuard:
    """Per-request and per-tenant-day cost enforcement via Redis.

    Cost tracking uses INCRBYFLOAT on a Redis key formatted as
    ``budget:<tenant>:<YYYY-MM-DD>``.  The key expires after 24 h (86 400 s),
    so there is no clean-up job required.

    The two limit types are independent:
    - ``per_request_usd`` - the maximum cost that a *single* LLM call may add
      to the running counter for the current request.  Pass the running total
      of the current request via ``request_total_usd`` on each call.
    - ``per_tenant_daily_usd`` - the maximum total spend for a tenant in a
      calendar day (UTC).

    Both limits are checked *before* the cost is committed.  If either is
    exceeded the atomic INCRBYFLOAT is still executed (so the counter reflects
    reality), but BudgetExceededError is raised immediately after.
    """

    _KEY_TTL_S: int = 86_400  # 24 hours

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        per_request_usd: float = 0.05,
        per_tenant_daily_usd: float = 5.0,
    ) -> None:
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
        self.per_request_usd = per_request_usd
        self.per_tenant_daily_usd = per_tenant_daily_usd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, tenant: str) -> str:
        today = date.today().isoformat()  # UTC assumption is fine for a POC
        return f"budget:{tenant}:{today}"

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def check_and_increment(
        self,
        tenant: str,
        cost_usd: float,
        *,
        request_total_usd: float = 0.0,
    ) -> None:
        """Increment the tenant's daily counter and raise if any budget is exceeded.

        Parameters
        ----------
        tenant:
            Opaque string identifying the tenant (e.g. organisation ID).
        cost_usd:
            The cost of the call that is about to be (or has just been) made.
        request_total_usd:
            The cumulative cost of *previous* calls within the same logical
            request.  Together with *cost_usd* this represents the total cost
            of a single user request.
        """
        # Check per-request limit first (cheap, no Redis round-trip needed).
        if request_total_usd + cost_usd > self.per_request_usd:
            raise BudgetExceededError(
                f"Per-request budget exceeded: "
                f"${request_total_usd + cost_usd:.4f} > ${self.per_request_usd:.4f}"
            )

        key = self._key(tenant)

        # Atomically increment — INCRBYFLOAT is atomic in Redis (like AtomicLong.addAndGet).
        new_total: float = float(await self._client.incrbyfloat(key, cost_usd))

        # Set / refresh TTL so the key expires after one calendar day.
        await self._client.expire(key, self._KEY_TTL_S)

        if new_total > self.per_tenant_daily_usd:
            raise BudgetExceededError(
                f"Daily budget exceeded for tenant '{tenant}': "
                f"${new_total:.4f} > ${self.per_tenant_daily_usd:.4f}"
            )

    async def get_tenant_spend(self, tenant: str) -> float:
        """Return the current day's spend for *tenant* in USD (0.0 if no data)."""
        value = await self._client.get(self._key(tenant))
        return float(value) if value is not None else 0.0


# ---------------------------------------------------------------------------
# In-memory test double
# ---------------------------------------------------------------------------


class FakeBudgetGuard:
    """In-memory drop-in for testing without a running Redis instance.

    Implements the same public interface as BudgetGuard, using a plain dict
    as the counter store.  The key schema is the same as BudgetGuard._key so
    that tests remain realistic.
    """

    def __init__(
        self,
        *,
        per_request_usd: float = 0.05,
        per_tenant_daily_usd: float = 5.0,
    ) -> None:
        self._spend: dict[str, float] = {}
        self.per_request_usd = per_request_usd
        self.per_tenant_daily_usd = per_tenant_daily_usd

    def _key(self, tenant: str) -> str:
        today = date.today().isoformat()
        return f"budget:{tenant}:{today}"

    async def check_and_increment(
        self,
        tenant: str,
        cost_usd: float,
        *,
        request_total_usd: float = 0.0,
    ) -> None:
        if request_total_usd + cost_usd > self.per_request_usd:
            raise BudgetExceededError(
                f"Per-request budget exceeded: "
                f"${request_total_usd + cost_usd:.4f} > ${self.per_request_usd:.4f}"
            )

        key = self._key(tenant)
        self._spend[key] = self._spend.get(key, 0.0) + cost_usd

        if self._spend[key] > self.per_tenant_daily_usd:
            raise BudgetExceededError(
                f"Daily budget exceeded for tenant '{tenant}': "
                f"${self._spend[key]:.4f} > ${self.per_tenant_daily_usd:.4f}"
            )

    async def get_tenant_spend(self, tenant: str) -> float:
        return self._spend.get(self._key(tenant), 0.0)
