# Observability — OpenTelemetry + Langfuse

Every request and every eval case produces one trace: the LangGraph node tree, each LLM
call with model, token usage and cost, and quality scores attached to the trace.

![Langfuse trace of one eval case](images/langfuse-trace.png)

*One `run_eval.py --agent` case: `eval.case → agent.run → intent / plan / retrieve / cluster /
summarize / judge / groundedness`, each LLM generation with tokens and cost, and five scores
on the trace (`judge_score`, `groundedness`, `citation_precision`, `groundedness_heuristic`,
`substring_match`).*

## How it is wired

Application code only speaks OpenTelemetry; Langfuse is plugged in at one point.

| Layer | What it emits | Where |
|---|---|---|
| Graph nodes | `agent.<node>` span per node (`@traced`) | `packages/agent/src/poc/agent/graph.py` |
| LLM calls | `llm.generate.openai` span with `gen_ai.*` usage, model, cost, prompt and completion | `packages/llm/src/poc/llm/openai_compatible.py` → `record_llm_call` |
| Request | `agent.run` root span: trace name, tenant as user, input/output, intent tag | `packages/agent/src/poc/agent/agent.py` |
| Quality | `judge_score`, `groundedness` scores on the request trace | `ProductInsightAgent.run` |
| Evals | `eval.case` trace per case, one session per run, every metric as a score | `scripts/run_eval.py` |
| Export | Langfuse SDK span processor attached to the project `TracerProvider` | `packages/observability/src/poc/observability/langfuse_client.py` |

Design choices:

- **OTel first, vendor second.** Nodes and providers never import Langfuse. Swapping the
  backend (Jaeger, Honeycomb, Phoenix) means changing `tracing.configure`, not the agent.
- **The SDK owns export.** `langfuse_client.attach_to_provider` hands our `TracerProvider` to
  `Langfuse(tracer_provider=...)`, so spans go out through the SDK's batching exporter, and
  scores go through the same client. `should_export_span` is widened to all spans; by default
  the SDK keeps only `gen_ai.*` spans and would drop the `agent.*` node tree.
- **Opt-in and fail-soft.** With no `LANGFUSE_*` env vars, every helper is a no-op: unit tests
  and keyless CI need no server and no mocks. A failed score upload is logged, never raised.
- **Stable score ids.** A score id is `<trace_id>:<name>`, so re-scoring the same trace
  overwrites the value instead of adding a duplicate.

## Run it locally

```bash
cp infra/langfuse/.env.example infra/langfuse/.env
# fill NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY and pick your own
# LANGFUSE_INIT_PROJECT_PUBLIC_KEY / _SECRET_KEY / LANGFUSE_INIT_USER_PASSWORD
docker compose --profile observability up -d   # Langfuse UI on http://localhost:${LANGFUSE_PORT:-3000}
```

The Langfuse stack (ClickHouse, MinIO, Langfuse server + worker) lives in the `observability`
compose profile, so a plain `docker compose up -d` starts only Postgres, Qdrant and Redis. Set
`COMPOSE_PROFILES=observability` to include it by default. MinIO uses the Chainguard image
(`cgr.dev/chainguard/minio`), because `minio/minio` and `minio/mc` are no longer published on
Docker Hub; its two buckets (`langfuse-events`, `langfuse-media`) are created on startup.

Then enable export in the root `.env` with the same project keys:

```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...   # = LANGFUSE_INIT_PROJECT_PUBLIC_KEY
LANGFUSE_SECRET_KEY=sk-lf-...   # = LANGFUSE_INIT_PROJECT_SECRET_KEY
```

Produce traces:

```bash
set -a; . ./.env; set +a
uv run python scripts/run_eval.py --golden-set golden_set_sample \
  --data-path data/raw/sample_reviews.jsonl --agent --limit 3
```

Open **Tracing** (one trace per case) or **Sessions** (one session per eval run) and log in
with `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`. For the API, traces appear
per `/ask` request under the name `agent.run`, with the tenant as the Langfuse user.

If some of the default host ports (5432, 6379, 8123, 9000, 3000) are taken, override them in
the root `.env`: `POSTGRES_PORT`, `REDIS_PORT`, `CLICKHOUSE_HTTP_PORT`, `MINIO_API_PORT`,
`MINIO_CONSOLE_PORT`, `LANGFUSE_PORT`. Keep `NEXTAUTH_URL` in `infra/langfuse/.env` on the same
port as `LANGFUSE_PORT`.

## What the traces already caught

The first traced eval run showed `ERROR` on the intent, plan and judge generations. Gemini's
OpenAI-compatible endpoint rejected the native `response_mime_type` / `response_schema`
fields with HTTP 400, so every structured call quietly fell back to its default and the
judge always reported `0.5`. The fix switches to the standard
`response_format: {type: json_schema}` shape. The metrics alone did not show the problem;
the trace did.
