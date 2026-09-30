# ADR-0002 — Vector Database: Qdrant

> Status: Accepted (amended 2026-09-28)
> Date: 2026-05-08
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent indexes and retrieves text chunks from review documents. The vector database is the centerpiece of the retrieval layer and must support:

- **Dense vector search** (cosine / dot-product similarity) over high-dimensional embeddings (1024-dim for `bge-m3`).
- **Payload filtering** — every chunk carries structured metadata: `tenant`, `product_id`, `region`, `lang`, `created_at`. Retrieval must filter on these fields with boolean expressions _before_ the ANN scan, not as a post-filter over full results. This is the "pre-filter" or "filtered ANN" requirement.
- **Multi-tenancy isolation.** Each tenant maps to a separate Qdrant collection (`reviews__{tenant}`), so that data cannot leak across tenant boundaries at query time. *Implementation status:* the index layer supports this, but the POC API still serves every tenant from `reviews__default` — see the ADR-0005 amendment.
- **Hybrid search (BM25 + dense).** The retrieval layer combines dense ANN with BM25 lexical scores. The database must either support sparse vectors natively or allow the dense index to be the primary store while BM25 runs externally (in-process with `rank_bm25`).
- **Local development.** The full stack must run on a developer laptop via `docker compose up`. No cloud dependency at development time.
- **Scale target for POC.** 10k–100k chunks. Production-like architecture but not production-scale load testing.

The main candidates evaluated were **Qdrant** and **pgvector** (Postgres extension).

---

## Decision

We will use **Qdrant** as the primary vector store, deployed via Docker Compose.

Qdrant is used through the `qdrant-client` Python SDK. Collections follow the naming convention `reviews__{tenant}`. The `QdrantIndex` class in `packages/retrieval/src/poc/retrieval/qdrant_index.py` wraps the client, exposes `upsert(chunks)` and `search(query_vec, *, filters, top_k)`.

Payload schema per point:

```json
{
  "doc_id": "string",
  "chunk_id": "string",
  "lang": "string",
  "product_id": "string",
  "region": "string",
  "position": 0,
  "text": "string"
}
```

Filters are expressed as Qdrant `Filter` objects using `FieldCondition` on payload keys, which Qdrant evaluates _before_ the ANN scan via HNSW with payload index. This gives sub-100ms filtered search even at 100k points on a laptop.

For hybrid search in Sprint 1, BM25 runs in-process via `rank_bm25` and results are fused with Qdrant ANN scores using Reciprocal Rank Fusion. Qdrant's native sparse vector support (using sparse IDF indices) is an upgrade path for Sprint 3+ if query latency becomes a bottleneck.

---

## Consequences

### Positive consequences

- **Pre-filtering ANN is first-class in Qdrant.** Payload indexes are declared once per collection; filter expressions compose naturally (`must`, `should`, `must_not`). This is the most important retrieval quality lever — without pre-filtering, a `product_id` filter on post-results loses most of the ANN budget.
- **Multi-vector support.** Qdrant supports multiple named vectors per point (e.g., dense + sparse + late-interaction ColBERT). This means we can upgrade from `bge-m3` dense to full hybrid (dense + sparse) without re-ingesting data — just add a second named vector.
- **Snapshots and backups** are native — `POST /collections/{name}/snapshots`. In production, this integrates with object storage trivially.
- **OSS with a managed cloud offering.** If the POC matures into a real product, Qdrant Cloud is the managed path with zero code changes.
- **gRPC + REST APIs.** The Python client uses gRPC by default for batch operations — significantly faster than HTTP for bulk upserts (Sprint 1 indexing scripts).

### Negative consequences / risks

- **One more Docker service.** The stack grows: Postgres + Qdrant + Redis + Langfuse. On laptops with <16 GB RAM, this may be tight. Qdrant is the most memory-hungry (HNSW graph in RAM for the collection). Mitigation: set `on_disk_payload: true` for the POC to cap RAM.
- **Not ACID / not relational.** Qdrant is not a general-purpose database. Document metadata (reviews source URL, ingestion timestamp, audit fields) must live in Postgres; Qdrant stores only the payload fields needed for filtering. Two stores = two consistency concerns. The `QdrantIndex` class does not implement transactions across both.
- **Learning curve for Qdrant-specific concepts.** Payload indexes must be declared explicitly before filtering works efficiently. Developers familiar with Elasticsearch/OpenSearch must unlearn inverted-index thinking.

### Neutral / noteworthy

- Qdrant collection-per-tenant is a deliberate choice (data isolation > resource sharing). Alternative is a single collection with a `tenant` payload filter. Collection-per-tenant gives guaranteed isolation at the ANN level and simpler delete-all-tenant-data operations.
- Qdrant does not replace Postgres. The relational store handles audit logs, review metadata, tenant configuration, and anything that needs JOINs or transactions.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **pgvector** (Postgres extension) | Single database = single consistency domain; familiar SQL for filtering; joins between metadata and vectors in one query; no extra Docker service | Pre-filter ANN is weaker: pgvector's HNSW does not support filtered ANN natively — filter is applied post-scan, degrading recall at high selectivity; no multi-vector per row; schema changes require migrations | The payload-filter requirement is the dealbreaker. At `product_id` selectivity of 1-in-20, pgvector post-filter returns <50% of the ANN budget as valid results |
| **Pinecone** | Managed cloud, zero ops; good filtered ANN support; familiar to many ML engineers | Cloud-only — no local Docker image; costs money even at POC scale; vendor lock-in for vector search | Violates "run locally with one command" requirement |
| **Weaviate** | OSS + Docker; GraphQL API; multi-tenancy built-in; hybrid search (BM25 + vector) integrated | Higher memory footprint than Qdrant; GraphQL over REST adds schema ceremony; less commonly cited in recent production case studies for our use case | Heavier operational footprint; the integrated BM25 is a plus but not worth the trade-off |
| **Chroma** | Simple Python API; in-process option (no Docker needed for dev) | Poor filtering performance at scale; no production deployment story beyond self-managed; less active development of enterprise features | Not production-grade enough for the portfolio objective |
| **Elasticsearch / OpenSearch** | Battle-tested; excellent BM25; good kNN; rich filtering | Very heavy (JVM, 2+ GB RAM); overkill for POC; configuration complexity | Too much operational overhead for a single-developer POC |

---

## References

- [Qdrant documentation: Payload indexes and filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [Qdrant documentation: Collections and named vectors](https://qdrant.tech/documentation/concepts/collections/)
- [pgvector: limitations of filtered ANN](https://github.com/pgvector/pgvector?tab=readme-ov-file#indexing)
- `docs/planning/06_implementation_plan.md` §S1.T4 — Embedder + Qdrant client spec
- `docs/planning/05_recommended_poc.md` — storage layer overview

---

## Amendment 2026-09-28

**What changed since 2026-05.** Qdrant's Query API now does hybrid retrieval server-side: `prefetch` a dense query and a sparse query, fuse with `Fusion.RRF` (or DBSF / weighted), optionally re-score — in one round trip. Sparse vectors support `Modifier.IDF`, so BM25-style weighting is computed by Qdrant over the collection. `bge-m3` (our embedder) can itself emit sparse lexical weights, which we currently discard.

**Current state:** BM25 is still computed in-process (`rank_bm25`, rebuilt from the corpus at startup) and fused with Qdrant dense hits by our own RRF in `hybrid.py`. This works at 12k chunks but duplicates the corpus in API memory, is per-process, and ignores tenant isolation for the lexical side. Proposed replacement: Qdrant server-side hybrid (dense + sparse `prefetch`, `Fusion.RRF`) in one collection per tenant.

**Operational note:** `docker-compose.yml` uses `qdrant/qdrant:latest`. Pin a version — hybrid/multitenancy features are version-gated (e.g. tiered multitenancy needs ≥ 1.16).
