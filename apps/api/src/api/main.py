from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header
from poc.audit.logger import AuditEvent, AuditLogger
from poc.audit.settings import AuditSettings
from poc.audit.store import PostgresAuditStore
from poc.core.tenant import current_tenant_token, reset_tenant
from poc.guardrails.input_checks import check_input
from poc.llm.registry import REGISTRY, get_provider
from poc.llm.settings import LLMSettings
from poc.observability.tracing import configure as configure_tracing
from poc.observability.tracing import get_tracer
from pydantic import BaseModel

from api.error_handlers import GuardrailViolation, register_error_handlers
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.tenant import TenantMiddleware
from api.routes.metrics import router as metrics_router

_log = logging.getLogger(__name__)

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

_state: dict[str, Any] = {}

# Project root: apps/api/src/api/main.py → up 4 → repo root holding data/raw/reviews.jsonl
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


async def _build_audit() -> AuditLogger | PostgresAuditStore:
    """Pick the audit backend from env.

    POC_AUDIT_BACKEND=postgres  → append-only Postgres audit_log (S5.T3), with the
    DSN from POC_AUDIT_DB_URL (see AuditSettings). Any other value (default) keeps the
    JSONL file logger used in dev/test. If Postgres is selected but unreachable at
    startup we degrade to JSONL rather than failing the whole app.
    """
    backend = os.getenv("POC_AUDIT_BACKEND", "jsonl").lower()
    if backend == "postgres":
        store = PostgresAuditStore(AuditSettings().db_url)
        try:
            await store.connect()
            _log.info("Audit backend: Postgres audit_log (append-only)")
            return store
        except Exception as exc:
            _log.warning("Postgres audit unavailable (%s) — falling back to JSONL", exc)
    _log.info("Audit backend: JSONL file (audit.jsonl)")
    return AuditLogger("audit.jsonl")


def _build_router() -> Any | None:
    """Build a RoutedLLM with response cache and budget caps (S3.T2-T4).

    Opt-in via POC_USE_ROUTER=1 — the default single-provider path stays intact.
    Backing stores: with POC_REDIS_URL set, cache and budget counters live in Redis
    (docker-compose service); without it we degrade to the in-memory fakes so the
    router still works — with per-process (non-durable) budgets — on a bare laptop.
    Budget limits come from `budgets:` in router.yaml, the single source of truth.
    """
    if os.getenv("POC_USE_ROUTER", "0").lower() not in ("1", "true", "yes"):
        return None
    from poc.llm.budget import BudgetGuard, FakeBudgetGuard
    from poc.llm.cache import FakeRedisCache, RedisResponseCache
    from poc.llm.router import RoutedLLM

    limits = RoutedLLM().budgets  # read budgets: from router.yaml
    per_request = float(limits.get("per_request_usd", 0.05))
    per_tenant_daily = float(limits.get("per_tenant_daily_usd", 5.0))

    redis_url = os.getenv("POC_REDIS_URL", "")
    if redis_url:
        cache: Any = RedisResponseCache(redis_url)
        guard: Any = BudgetGuard(
            redis_url=redis_url,
            per_request_usd=per_request,
            per_tenant_daily_usd=per_tenant_daily,
        )
        _log.info("LLM router enabled: Redis cache+budget at %s", redis_url)
    else:
        cache = FakeRedisCache()
        guard = FakeBudgetGuard(per_request_usd=per_request, per_tenant_daily_usd=per_tenant_daily)
        _log.warning("LLM router enabled without POC_REDIS_URL — in-memory cache/budget")
    return RoutedLLM(cache=cache, budget_guard=guard)


def _build_agent(provider: Any, router: Any | None = None) -> Any | None:
    """Build a ProductInsightAgent backed by a BM25-first HybridRetriever.

    Graceful degradation, mirroring scripts/run_eval.py: if Qdrant or the reranker
    models are unavailable we fall back to BM25-only / NoOp so the API still serves
    /ask through the full LangGraph pipeline (intent…judge) without external infra.
    Returns None if even BM25 can't be built — caller then uses the direct-LLM path.
    """
    try:
        from poc.agent.agent import ProductInsightAgent
        from poc.ingestion.chunker import RecursiveTokenChunker
        from poc.ingestion.normalizer import normalize
        from poc.ingestion.sources.jsonl import JsonlSource
        from poc.retrieval.bm25 import BM25Index
        from poc.retrieval.hybrid import HybridRetriever
        from poc.retrieval.reranker import NoOpReranker

        reviews_path = _PROJECT_ROOT / "data" / "raw" / "reviews.jsonl"
        if not reviews_path.exists():
            _log.warning("reviews.jsonl not found at %s — /ask uses direct LLM path", reviews_path)
            return None

        bm25 = BM25Index()
        chunker = RecursiveTokenChunker()
        chunks: list[Any] = []
        for i, raw_doc in enumerate(JsonlSource(reviews_path).iter()):
            if i >= 2000:
                break
            chunks.extend(chunker.chunk(normalize(raw_doc)))
        bm25.build(chunks)

        class _NoOpQdrant:
            def search(self, *args: Any, **kwargs: Any) -> list:
                return []

        retriever = HybridRetriever(
            qdrant_index=_NoOpQdrant(),
            bm25_index=bm25,
            reranker=NoOpReranker(),
        )
        _log.info("Agent pipeline ready: BM25 index with %d chunks", len(bm25))
        return ProductInsightAgent(llm=provider, retriever=retriever, router=router)
    except Exception as exc:
        _log.warning("Agent pipeline unavailable (%s) — /ask uses direct LLM path", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize OpenTelemetry once at startup. When LANGFUSE_HOST is set, spans are
    # exported to Langfuse over OTLP; with OTEL_CONSOLE_EXPORT=true they print to the
    # console; otherwise they are recorded but not exported (see S3.T6).
    configure_tracing(service_name="product-insight-agent")
    settings = LLMSettings()
    provider = get_provider(settings.default_provider)
    _state["provider"] = provider
    _state["provider_name"] = settings.default_provider
    _state["router"] = _build_router()
    _state["agent"] = _build_agent(provider, _state["router"])
    _state["audit"] = await _build_audit()
    yield
    audit = _state.get("audit")
    if isinstance(audit, PostgresAuditStore):
        await audit.close()
    _state.clear()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Product Insight Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Routers & error handlers ───────────────────────────────────────────────────

app.include_router(metrics_router)
register_error_handlers(app)

# ── Middleware ─────────────────────────────────────────────────────────────────
# Starlette processes middleware LIFO (last added = outermost = runs first).
# Adding RateLimitMiddleware BEFORE TenantMiddleware means:
#   request:  TenantMiddleware → RateLimitMiddleware → handler
#   response: handler → RateLimitMiddleware → TenantMiddleware
# Java analogy: Filter chain ordered by @Order annotation — lower value runs first.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(TenantMiddleware)


# ── Request / Response schemas ─────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    tenant: str = "default"
    filters: dict[str, Any] = {}
    model: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[dict] = []
    model: str = ""
    cost_usd: float | None = None
    latency_ms: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    x_tenant: str | None = Header(default=None),
) -> AskResponse:
    start_ms = int(time.time() * 1000)

    # Resolve tenant from header > body > default
    tenant = x_tenant or req.tenant or "default"
    token = current_tenant_token(tenant)
    audit: AuditLogger | PostgresAuditStore = _state["audit"]

    try:
        # Guardrails: input check
        guard = check_input(req.question)
        if not guard.passed:
            await audit.log(
                AuditEvent(
                    event_type="input_rejected",
                    tenant=tenant,
                    data={"violations": guard.violations, "question": req.question[:200]},
                )
            )
            # Spec S4.T5: a blocked request is a *refusal*, not a client error.
            # Raise GuardrailViolation so the handler returns 200 {status: refused,
            # reason: guardrail.<reason>} instead of a 4xx/5xx. `passed` is False
            # only for prompt-injection (PII is redacted, not blocked).
            reason = next(
                (v.split(":", 1)[0] for v in guard.violations if v.startswith("prompt_injection")),
                "input_rejected",
            )
            raise GuardrailViolation(reason)

        sanitized_q = guard.sanitized_text
        agent = _state.get("agent")

        if agent is not None:
            # Preferred path: run the full LangGraph pipeline. ProductInsightAgent.run()
            # opens the `agent.run` root span; each node is @traced (agent.intent…judge)
            # and every LLM call opens an `llm.generate` child span — so Langfuse renders
            # one trace per /ask with the node tree and per-generation tokens/cost.
            answer = await agent.get_answer(sanitized_q, tenant=tenant, filters=req.filters or None)
            latency_ms = int(time.time() * 1000) - start_ms

            await audit.log(
                AuditEvent(
                    event_type="ask",
                    tenant=tenant,
                    data={
                        "question": sanitized_q[:200],
                        "model": answer.model,
                        "cost_usd": answer.cost_usd,
                        "latency_ms": latency_ms,
                    },
                )
            )

            return AskResponse(
                answer=answer.text,
                citations=[c.model_dump() for c in answer.citations],
                model=answer.model,
                cost_usd=answer.cost_usd,
                latency_ms=latency_ms,
            )

        # Fallback path: retrieval stack unavailable → single direct LLM call, still
        # wrapped in an `agent.run` root span so one trace per /ask reaches Langfuse.
        provider = _state["provider"]
        from poc.llm.provider import LLMMessage

        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a product insight analyst. "
                    "Answer questions about product reviews concisely and factually."
                ),
            ),
            LLMMessage(role="user", content=sanitized_q),
        ]

        model = req.model or REGISTRY[_state["provider_name"]].default_model

        tracer = get_tracer()
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("tenant", tenant)
            span.set_attribute("question.length", len(sanitized_q))
            response = await provider.complete(messages, model=model, max_tokens=800)

        latency_ms = int(time.time() * 1000) - start_ms

        await audit.log(
            AuditEvent(
                event_type="ask",
                tenant=tenant,
                data={
                    "question": sanitized_q[:200],
                    "model": model,
                    "cost_usd": response.cost_usd,
                    "latency_ms": latency_ms,
                },
            )
        )

        return AskResponse(
            answer=response.content,
            model=model,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
        )

    finally:
        reset_tenant(token)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
