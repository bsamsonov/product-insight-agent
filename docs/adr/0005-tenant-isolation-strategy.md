# ADR-0005 — Per-tenant Qdrant collections over payload filters

> Status: Accepted
> Date: 2026-05-16
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent is designed as a multi-tenant platform: multiple customers (tenants) each upload their own product reviews and receive answers that reference only their data. A hardware retailer must never see a competitor's reviews; a cosmetics brand's data must not appear in a fashion company's retrieval results.

Enforcing tenant isolation in the vector store is a critical security and correctness requirement. The failure mode — a tenant receiving another tenant's data — is a data leakage incident, not a metrics regression.

Two architectural approaches exist for multi-tenant vector databases:

**Option A — Shared collection with payload filter:** All tenants' chunks live in a single Qdrant collection. Each chunk carries a `tenant` field in its payload. At search time, a filter `{must: [{key: "tenant", match: {value: "<id>"}}]}` is applied to restrict results.

**Option B — Separate collection per tenant:** Each tenant gets its own Qdrant collection, named `reviews__{tenant_id}`. The agent selects the collection at request time based on the resolved tenant ID.

The Product Insight Agent resolves tenant identity from an API key header (`X-Tenant-ID`) in the FastAPI middleware layer (`packages/core/src/poc/core/tenant.py`). All downstream calls receive the resolved `tenant_id` as a parameter — it is never inferred from data content.

At POC scale, we expect 3–10 tenants, each with 10k–100k chunks. At production scale, the architecture must support hundreds of tenants without redesign.

---

## Decision

We will use **separate Qdrant collections per tenant**, named `reviews__{tenant_id}`.

Collection creation happens on first ingest for a tenant. The `QdrantIndex` class in `packages/retrieval/src/poc/retrieval/qdrant_index.py` accepts `tenant_id` at construction time and uses it to derive the collection name. No cross-collection queries are ever issued.

The FastAPI middleware resolves tenant identity before the request reaches any retrieval or agent code. If tenant resolution fails (unknown API key), the request is rejected at the middleware layer with a 401 — retrieval code never sees an ambiguous tenant.

---

## Consequences

### Positive consequences

- **Complete data isolation.** There is no filter logic that can be accidentally omitted or bypassed. Even if a bug removes the payload filter from a query, the query targets only that tenant's collection — the physical data of other tenants does not exist in that collection.
- **Independent scaling per tenant.** A single tenant with 10M chunks can have its collection moved to a dedicated Qdrant shard without affecting other tenants. A shared collection would require custom shard-key configuration to achieve the same.
- **Clean deletion.** Deleting a tenant means dropping one collection: `client.delete_collection("reviews__acme")`. With a shared collection, deletion requires a bulk-delete by filter — which is slower, requires a separate scan, and risks partial failure leaving orphaned data.
- **Simple security model.** Auditors can verify isolation by inspecting the collection name in every query — no filter logic to audit. This is the same argument as physical schema separation vs. row-level security in relational databases.
- **No schema change for new tenants.** Adding tenant `newco` requires no DDL, no migration, no schema version bump. The collection is created automatically on first ingest.
- **Per-collection monitoring.** Qdrant exposes collection-level metrics (vector count, memory usage, indexing state). With separate collections, monitoring dashboards can show per-tenant resource consumption without additional grouping logic.

### Negative consequences / risks

- **Collection count grows with tenant count.** At 1000 tenants, there are 1000 collections. Qdrant handles thousands of collections efficiently (each is an independent HNSW index on disk), but operational tooling (backup scripts, monitoring dashboards) must be collection-aware.
- **Higher operational overhead for cross-tenant analytics.** If the platform ever needs to run cross-tenant aggregate analytics (e.g., "what are the most common complaint themes across all tenants?"), it would require querying multiple collections and merging results in application code. With a shared collection, this is a single query.
- **Cold start per tenant.** The first search on a freshly created collection may be slower as the HNSW index is built. For small collections (<1000 chunks) this is imperceptible.

### Neutral / noteworthy

- Qdrant's documentation explicitly supports the per-collection pattern for multi-tenancy and provides collection alias support for zero-downtime collection swaps (useful for re-indexing).
- The naming convention `reviews__{tenant_id}` uses double underscore as a namespace separator to avoid collisions with tenant IDs that might contain underscores.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **Single collection + payload filter** | Simpler operations (one collection to back up, monitor, scale); slightly lower memory overhead (shared HNSW graph for common vectors) | Filter bypass risk: a single missing `must` clause in any query leaks data; bulk deletion is slow and fault-prone; cross-tenant performance coupling (a large tenant's indexing slows searches for small tenants) | Data leakage risk from filter bypass is unacceptable; deletion complexity is a maintenance hazard |
| **Separate collection per tenant + Qdrant collection groups** | Adds namespace management on top of per-collection isolation | Collection groups are not a Qdrant native feature; adds custom orchestration layer | Unnecessary complexity; basic per-collection approach is sufficient |
| **Postgres pgvector with row-level security** | Familiar SQL tooling; native RLS enforced at DB level | pgvector ANN performance is significantly lower than Qdrant at 100k+ vectors; no native sparse vector support for hybrid BM25+dense retrieval | Already rejected in ADR-0002; RLS is compelling but vector search performance wins |

---

## References

- `packages/retrieval/src/poc/retrieval/qdrant_index.py` — `QdrantIndex` with tenant-scoped collection name
- `packages/core/src/poc/core/tenant.py` — tenant resolution middleware
- ADR-0002 — Qdrant chosen as vector store
- [Qdrant multi-tenancy guide](https://qdrant.tech/documentation/guides/multiple-partitions/)
- [Qdrant collection management](https://qdrant.tech/documentation/concepts/collections/)
