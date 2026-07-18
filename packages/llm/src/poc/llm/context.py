"""Request-scoped budget context (S3.T4).

The budget guard needs two pieces of information at the moment of an LLM call:
*which tenant* is paying and *how much this request has already spent*. The
LangGraph graph, however, is built once at startup, while tenant arrives with
every request. Threading these values through every node signature would be
invasive, so we use ``contextvars`` — the asyncio-safe analogue of Java's
ThreadLocal/MDC: the agent opens a request context in ``run()``, and
``RoutedLLM`` reads and updates it transparently on every call.

Usage::

    with request_budget(tenant="acme") as ctx:
        await agent_graph.ainvoke(...)   # RoutedLLM charges ctx as it goes
    print(ctx.total_usd)
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class RequestBudgetContext:
    """Mutable per-request accumulator: who pays and how much was spent so far."""

    tenant: str = "default"
    total_usd: float = field(default=0.0)

    def add_cost(self, cost_usd: float) -> None:
        self.total_usd += cost_usd


_current: ContextVar[RequestBudgetContext | None] = ContextVar(
    "llm_request_budget_context", default=None
)


def current_budget_context() -> RequestBudgetContext | None:
    """Return the active request context, or None outside a request scope."""
    return _current.get()


@contextmanager
def request_budget(tenant: str = "default") -> Iterator[RequestBudgetContext]:
    """Open a request-scoped budget context (asyncio-safe via contextvars)."""
    ctx = RequestBudgetContext(tenant=tenant)
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
