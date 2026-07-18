# ADR-0006 — Redis INCRBYFLOAT for per-tenant budget tracking

> Status: Accepted
> Date: 2026-05-16
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent calls external LLM APIs on behalf of tenants. Without budget enforcement, a single buggy client or a malicious actor could issue thousands of requests in a short window, resulting in hundreds of dollars in unexpected API charges.

Two levels of budget control are required:

1. **Per-request limit.** A single agent execution must not consume more than a configured token budget (default: 8000 tokens across all LLM calls in the graph). This prevents runaway recursive tool calls or infinite retry loops within one request.

2. **Per-tenant-day limit.** Each tenant has a daily spending cap (default: $1.00/day in the POC). Requests from a tenant that has exceeded its daily budget must be rejected before any LLM call is made.

The per-request limit is straightforward: `AgentState` accumulates token counts as nodes execute, and the `BudgetGuard` node checks the accumulated cost before each LLM call. This is in-process, single-threaded per request — no concurrency issues.

The per-tenant-day limit is harder: multiple concurrent requests from the same tenant arrive at different API workers simultaneously. A naive in-memory counter in each worker would give each worker an independent view of spending — at 4 workers, a tenant could spend 4× their daily cap before any worker detects the breach.

The solution requires an atomic shared counter with automatic expiry. The candidates evaluated were: Postgres counter with cron reset, application-level lock (Redis distributed lock), Redis atomic increment, and no budget cap.

---

## Decision

We will use **Redis `INCRBYFLOAT` with a 24-hour TTL key** for per-tenant-day budget accumulation.

The key schema is `budget:{tenant_id}:{YYYY-MM-DD}` (date in UTC). On each completed LLM call, the `BudgetGuard` issues:

```
INCRBYFLOAT budget:{tenant_id}:{date} {cost_usd}
EXPIREAT budget:{tenant_id}:{date} {next_midnight_unix}
```

Before each LLM call, the guard issues `GET budget:{tenant_id}:{date}` and compares to the configured cap. If the cap is exceeded, the request raises `BudgetExceededError` immediately.

The Redis connection is reused from the rate-limiting pool (same Redis instance, separate key namespace). The `BudgetGuard` is implemented in `packages/guardrails/src/poc/guardrails/input_checks.py` and is invoked by the LangGraph `__before__` hook on every node that calls an LLM.

Cost is estimated from the token counts returned in the LLM API response (`usage.prompt_tokens`, `usage.completion_tokens`) and the provider's per-token price stored in `LLMSettings`.

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
- **24-hour window does not align with calendar days.** The TTL starts from the first request of the day, not from midnight. A tenant who starts using the system at 23:59 gets a fresh 24h window — their usage from 23:59 to midnight and midnight to 23:59 the next day may overlap in a single window. For a POC, this approximation is acceptable.
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

- `packages/guardrails/src/poc/guardrails/input_checks.py` — `BudgetGuard` implementation
- `packages/agent/src/poc/agent/state.py` — `AgentState.accumulated_cost_usd` per-request accumulator
- [Redis INCRBYFLOAT documentation](https://redis.io/commands/incrbyfloat/)
- `docs/cost-model.md` — per-request cost estimates used to set default caps
- ADR-0008 — multi-provider LLM strategy (provider prices used in cost estimation)
