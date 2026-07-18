# LLM — package notes

- `RoutedLLM` loads `packages/llm/data/router.yaml`: roles (classifier, planner, summarizer, judge, escalate) with primary+fallback providers
- Auto-fallback on `LLMRateLimitError` or status 429/503: tries the fallback once, then re-raises
- Provider instances are cached by name in the `_providers` dict (lazy init via `from_env()`)
- `LLMProvider` — Protocol (duck-typed), async `complete()` and `stream()` methods
- `BudgetGuard` stores daily spend in Redis: key `budget:{tenant}:{YYYY-MM-DD}`, TTL 24h
- Two independent limits: per-request (before Redis) and per-tenant-daily (after `INCRBYFLOAT`)
- `check_and_increment()` increments first, then checks — the counter stays accurate even when the limit is exceeded
