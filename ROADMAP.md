# Roadmap

Product Insight Agent is a personal R&D proof of concept, not a production service. This page
lists what is known to be missing or weak, so the README numbers can be read in context.

## Known gaps

**Quality**

- Citation precision is 0.59 against a 0.70 target. Misses are mostly chunks from similar
  products; the planner's `product_title` filter is exact-match only (hypothesis, unverified).
- The golden set has 35 cases and no adversarial ones. Every question names its product, so
  hybrid retrieval scores about the same as BM25 alone.
- The LLM judge is uncalibrated and uses the same model family as the generator. Its score
  for the same answer can vary between runs.
- Clustering is a fixed keyword dictionary, not embedding-based.

**Latency**

- `/ask` makes six sequential LLM calls. The measured p50 is 10.2 s against a p95 < 8 s target.

**Multi-tenancy**

- Tenant identity (`X-Tenant-Id` allowlist) drives audit, rate limits, budgets and tracing,
  but retrieval is still single-tenant: every tenant reads the same collection, and BM25 is
  built in process memory. See [ADR-0005](docs/adr/0005-tenant-isolation-strategy.md).

**Operations**

- The daily budget is charged after each LLM call, so concurrent requests can overshoot it.
- The Redis rate limiter is not atomic and does not send `Retry-After`.
- `/metrics/summary` reads only the JSONL audit log, not the Postgres backend.
- Human review is a flag in the response; there is no durable review queue or resume endpoint.
- No streaming endpoint; the UI calls the synchronous `/ask`.
- PII detection covers only emails and phone numbers in the user question; full prompts are
  exported to traces.

**CI**

- The PR gate is a keyless retrieval smoke test; it does not catch prompt regressions. The
  LLM-judge eval runs manually.

## Next steps

1. Eval hardening: absolute thresholds, 60–100 golden cases including adversarial and
   product-agnostic questions, a judge from another model family, a measured baseline with
   per-PR deltas.
2. Per-tenant retrieval: a tenant-scoped index passed into the graph, server-side sparse
   vectors instead of in-process BM25.
3. Latency: merge or parallelize LLM calls, stream answers.
4. Durable human-in-the-loop: Postgres checkpointer, review queue, resume endpoint.
5. Hardening: reserve-then-settle budgets, atomic rate limiting, fenced retrieved content
   against indirect prompt injection, trace redaction.
