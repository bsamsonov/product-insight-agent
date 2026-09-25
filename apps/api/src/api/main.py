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
from poc.observability import langfuse_client
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


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() not in {"0", "false", "no", "off"}


def _corpus_path() -> Path | None:
    """Corpus for BM25: POC_CORPUS_PATH, else the full dataset, else the bundled sample."""
    override = os.getenv("POC_CORPUS_PATH")
    candidates = (
        [Path(override)]
        if override
        else [
            _PROJECT_ROOT / "data" / "raw" / "reviews.jsonl",
            _PROJECT_ROOT / "data" / "raw" / "sample_reviews.jsonl",
        ]
    )
    for path in candidates:
        path = path if path.is_absolute() else _PROJECT_ROOT / path
        if path.exists():
            return path
    return None


def _build_agent(provider: Any, router: Any | None = None) -> Any | None:
    """Build a ProductInsightAgent on the shared hybrid retriever.

    Graceful degradation, identical to scripts/run_eval.py (both use
    :func:`poc.retrieval.factory.build_hybrid_retriever`): dense search is used when
    Qdrant holds the tenant collection, otherwise BM25-only. Returns None when no corpus
    is found — the caller then uses the direct-LLM path.
    """
    try:
        from poc.agent.agent import ProductInsightAgent
        from poc.retrieval.factory import build_hybrid_retriever

        corpus = _corpus_path()
        if corpus is None:
            _log.warning("No corpus found under data/raw — /ask uses direct LLM path")
            return None

        retriever, info = build_hybrid_retriever(
            corpus,
            qdrant_url=os.getenv("POC_QDRANT_URL", "http://localhost:6333"),
            tenant="default",
            use_dense=_env_flag("POC_DENSE_RETRIEVAL"),
            use_reranker=_env_flag("POC_RERANKER"),
        )
        _state["retriever_info"] = info
        _log.info("Agent pipeline ready on %s: %s", corpus.name, info)
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
    langfuse_client.flush()


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
    intent: str | None = None
    judge_score: float | None = None
    groundedness: float | None = None
    needs_human_review: bool = False


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
            result = await agent.ask(sanitized_q, tenant=tenant, filters=req.filters or None)
            answer = result.answer
            answer_model = answer.model or REGISTRY[_state["provider_name"]].default_model
            latency_ms = int(time.time() * 1000) - start_ms

            await audit.log(
                AuditEvent(
                    event_type="ask",
                    tenant=tenant,
                    data={
                        "question": sanitized_q[:200],
                        "model": answer_model,
                        "needs_human_review": result.needs_human_review,
                        "cost_usd": answer.cost_usd,
                        "latency_ms": latency_ms,
                    },
                )
            )

            return AskResponse(
                answer=answer.text,
                citations=[c.model_dump() for c in answer.citations],
                model=answer_model,
                cost_usd=answer.cost_usd,
                latency_ms=latency_ms,
                intent=result.intent,
                judge_score=result.judge_score,
                groundedness=result.groundedness,
                needs_human_review=result.needs_human_review,
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
            langfuse_client.set_trace_attributes(
                span,
                name="agent.run",
                user_id=tenant,
                tags=["direct-llm"],
                input={"question": sanitized_q},
            )
            response = await provider.complete(messages, model=model, max_tokens=800)
            langfuse_client.set_trace_attributes(span, output=response.content)

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
