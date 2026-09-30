# ADR-0006 — Redis INCRBYFLOAT for per-tenant budget tracking

> Status: Accepted (amended 2026-09-28 — implementation description corrected)
> Date: 2026-05-16
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent calls external LLM APIs on behalf of tenants. Without budget enforcement, a single buggy client or a malicious actor could issue thousands of requests in a short window, resulting in hundreds of dollars in unexpected API charges.

Two levels of budget control are required:

1. **Per-request limit.** A single agent execution must not consume more than a configured token budget (default: 8000 tokens across all LLM calls in the graph). This prevents runaway recursive tool calls or infinite retry loops within one request.

2. **Per-tenant-day limit.** Each tenant has a daily spending cap (default: $5.00/day, `budgets.per_tenant_daily_usd` in `router.yaml`). Once a tenant has exceeded its daily budget, further LLM calls are rejected (the call that crosses the cap still completes — see Decision).

The per-request limit is straightforward: a request-scoped budget context (a `contextvars` value opened by `agent.run()`, see `poc.llm.context`) accumulates the cost of every LLM call made while serving one request. This is in-process and per request — no concurrency issues.

The per-tenant-day limit is harder: multiple concurrent requests from the same tenant arrive at different API workers simultaneously. A naive in-memory counter in each worker would give each worker an independent view of spending — at 4 workers, a tenant could spend 4× their daily cap before any worker detects the breach.

The solution requires an atomic shared counter with automatic expiry. The candidates evaluated were: Postgres counter with cron reset, application-level lock (Redis distributed lock), Redis atomic increment, and no budget cap.

---

## Decision

We will use **Redis `INCRBYFLOAT` with a 24-hour TTL key** for per-tenant-day budget accumulation.

The key schema is `budget:{tenant_id}:{YYYY-MM-DD}` (date from `date.today()`, treated as UTC for the POC). **After** each completed LLM call — once the real cost is known — the `BudgetGuard` issues:

```
INCRBYFLOAT budget:{tenant_id}:{date} {cost_usd}
EXPIRE      budget:{tenant_id}:{date} 86400
```

and raises `BudgetExceededError` if the returned running total exceeds the daily cap. The per-request cap is checked first, in process, against the request-scoped running total. There is no separate pre-call `GET`: the counter always reflects real spend, and the call that crosses the cap is the one that fails.

The guard is **not** a graph node or hook. It is injected into the model router (`RoutedLLM`, see ADR-0008; active only with `POC_USE_ROUTER=1`) at API startup; `RoutedLLM.route()` calls `BudgetGuard.check_and_increment()` after every real (non-cached) provider call. Graph nodes only see the resulting `BudgetExceededError`, which the API maps to HTTP 429. The Redis instance is shared with rate limiting (separate key namespace). When `POC_REDIS_URL` is not set, the API wires the in-memory `FakeBudgetGuard` instead (single-process only).

Cost is estimated from the token counts returned in the LLM API response (`usage.prompt_tokens`, `usage.completion_tokens`) and the per-token prices in `packages/llm/data/pricing.yaml` (`poc.llm.pricing.estimate_cost`).

---

## Consequences

### Positive consequences

- **Atomic increment prevents race conditions.** Redis `INCRBYFLOAT` is a single server-side operation — no read-modify-write window exists. Two concurrent requests incrementing simultaneously cannot double-count or miss each other's spending.
- **TTL-based reset requires no cron job.** The key expires automatically 24 hours after creation. No scheduled task, no migration, no maintenance window. This is simpler to operate than a Postgres-based approach with a cron reset job.
- **Float arithmetic is handled correctly.** `INCRBYFLOAT` stores the value as a double-precision float on the Redis side, so `0.001 + 0.001 + ... × 1000` does not accumulate binary fraction errors in application code.
- **Sub-millisecond latency.** A Redis `GET` + `INCRBYFLOAT` roundtrip is typically <1ms on localhost and <5ms on a co-located network. This overhead is negligible compared to LLM API latency (500ms–5000ms).
- **Per-tenant visibility.** The key namespace allows easy inspection: `redis-cli KEYS "budget:acme:*"` shows all daily caps for tenant `acme`. No SQL query required.

### Negative consequences / risks

- **Redis is a single point of failure for budget enforcement.** If Redis is unavailable, the `BudgetGuard` falls through (graceful degradation by design): LLM calls proceed without budget checking. This is the correct trade-off for a POC — denying all requests when Redis is down is worse than temporarily allowing overspend. In production, Redis Sentinel or Cluster would be required.
- **TTL does not align with calendar days.** The key name rolls over at midnight, but `EXPIRE 86400` is refreshed on every call, so a key lives up to 24 h after its last write. This only wastes a little memory — the date in the key, not the TTL, defines the budget window.
- **Post-charge enforcement can overshoot.** Because cost is committed after the call, N concurrent in-flight calls for a tenant near its cap can all complete before any of them sees the breach; worst-case overshoot ≈ N × max call cost. See the amendment below.
- **No persistence across Redis restarts.** If Redis is restarted without AOF/RDB persistence enabled, daily spending counters are lost. All tenants effectively get a fresh cap on restart. In the POC Docker Compose setup, Redis is configured without persistence (acceptable for development); production would enable AOF.
- **Cost estimation accuracy.** We estimate cost from token counts and a static per-token price. If a provider changes pricing or applies discounts (e.g., prompt caching discount), our estimate diverges from actual charges. The cap is therefore a safety approximation, not a billing guarantee.

### Neutral / noteworthy

- The same Redis instance serves rate limiting (sliding window counters) and budget tracking (daily accumulators). Key namespaces (`ratelimit:` vs `budget:`) prevent collisions.
- A future enhancement could push daily spend summaries to Postgres for audit reporting — Redis is the hot path, Postgres is the cold archive.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **Postgres counter with cron reset** | Persistent; supports complex queries (e.g., monthly rollups); familiar SQL tooling | Requires a cron job or pg_cron for daily reset; higher latency (5–20ms per check vs <1ms Redis); read-modify-write race under concurrency without explicit row-level locking | Cron dependency adds operational complexity; latency overhead on every LLM call; concurrency race without careful locking |
| **Application-level locking (distributed lock via Redis)** | Explicit control over the lock lifecycle | Does not actually solve the problem: the lock serialises requests but each worker still maintains its own counter; requires two Redis round trips (lock acquire + counter update) instead of one | More complex and slower than `INCRBYFLOAT`; does not eliminate the shared counter requirement |
| **No budget cap** | Zero implementation cost | Unbounded spend risk; a single runaway client or bug can generate hundreds of API charges before a human notices; unacceptable for a multi-tenant POC | Risk is too high even for a POC; implementing caps early prevents bad habits |
| **In-memory counter per worker** | Zero external dependencies | Each worker has an independent view; at N workers, effective cap = N × configured cap; not viable for multi-instance deployment | Does not work across multiple API server instances |

---

## References

- `packages/llm/src/poc/llm/budget.py` — `BudgetGuard` / `FakeBudgetGuard`
- `packages/llm/src/poc/llm/context.py` — request-scoped budget context (`contextvars`)
- `packages/llm/src/poc/llm/router.py` — `RoutedLLM._charge_budget()` (enforcement point)
- [Redis INCRBYFLOAT documentation](https://redis.io/commands/incrbyfloat/)
- `docs/cost-model.md` — per-request cost estimates used to set default caps
- ADR-0008 — multi-provider LLM strategy (provider prices used in cost estimation)

---

## Amendment 2026-09-28

**Corrections.** The original text placed `BudgetGuard` in `guardrails/input_checks.py`, invoked from a LangGraph `__before__` hook, with a pre-call `GET` and `EXPIREAT` next midnight. None of that matched the implementation (LangGraph has no `__before__` hook). The Decision section above now describes the actual design: post-call `INCRBYFLOAT` + `EXPIRE 86400` inside `RoutedLLM`.

**Known gap — concurrency overshoot.** "Atomic increment prevents race conditions" is true for counting, not for enforcement: the check happens after the money is spent. Production-grade options, in order of preference:

1. **Reserve-then-settle** — before the call, atomically reserve the *maximum* possible cost (`max_tokens × output price` + input estimate) with a Lua script that rejects if `spent + reserved + estimate > cap`; after the call, settle the difference. This is how LLM gateways enforce hard caps.
2. **Pre-call check** (`GET` + compare) — cheap, closes most of the gap, still racy.
3. **Enforce at the gateway** (e.g. LiteLLM / OmniRoute budgets) — the application no longer owns the counter.

**Revisit when:** more than one API worker serves the same tenant, or caps become contractual (billing) rather than safety limits.

