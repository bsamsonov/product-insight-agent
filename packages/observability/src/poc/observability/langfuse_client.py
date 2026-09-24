"""Langfuse integration on top of the project's OpenTelemetry tracing.

The agent, LLM and API layers only ever talk to OpenTelemetry (``@traced``,
``get_tracer()``, ``record_llm_call``). This module is the single place that knows
about Langfuse:

- :func:`attach_to_provider` — plugs the Langfuse SDK's span processor into *our*
  ``TracerProvider`` so every OTel span (graph nodes, ``llm.generate`` calls with
  ``gen_ai.*`` usage/cost) is exported to Langfuse. Called by
  :func:`poc.observability.tracing.configure`.
- :func:`set_trace_attributes` — names the trace and tags it with session / user /
  input / output using Langfuse's OTel attribute conventions.
- :func:`score_current_trace` / :func:`score_trace` — attach quality scores (judge
  verdict, groundedness, eval metrics) to a trace via the Langfuse SDK.
- :func:`flush` — drain buffered spans and scores before a short-lived process exits.

Everything degrades to a no-op when Langfuse is not configured — i.e. unless
``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are all set.
Tests and keyless CI therefore run without a Langfuse server and without mocks.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

_log = logging.getLogger(__name__)

_client: Any | None = None


def is_configured() -> bool:
    """True when all three Langfuse env vars are present."""
    return bool(
        os.getenv("LANGFUSE_HOST")
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def attach_to_provider(provider: TracerProvider) -> bool:
    """Attach the Langfuse span processor to *provider*. Returns True when enabled.

    ``should_export_span`` is widened to *all* spans: by default the SDK only exports
    its own spans and ``gen_ai.*`` spans, which would drop the ``agent.*`` node spans
    that give the trace its tree shape.
    """
    global _client
    if not is_configured():
        return False

    from langfuse import Langfuse

    _client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_HOST"],
        tracer_provider=provider,
        should_export_span=lambda _span: True,
    )
    _log.info("Langfuse export enabled -> %s", os.environ["LANGFUSE_HOST"])
    return True


def enabled() -> bool:
    """True once :func:`attach_to_provider` has wired a live client."""
    return _client is not None


def set_trace_attributes(
    span: trace.Span,
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    input: Any = None,
    output: Any = None,
) -> None:
    """Set trace-level Langfuse attributes on the root *span*.

    Pure OTel attributes — harmless when Langfuse is off, so call sites need no guard.
    """
    if name is not None:
        span.set_attribute("langfuse.trace.name", name)
    if session_id is not None:
        span.set_attribute("session.id", session_id)
    if user_id is not None:
        span.set_attribute("user.id", user_id)
    if tags:
        span.set_attribute("langfuse.trace.tags", tags)
    if input is not None:
        span.set_attribute("langfuse.trace.input", _to_str(input))
    if output is not None:
        span.set_attribute("langfuse.trace.output", _to_str(output))


def current_trace_id() -> str | None:
    """Hex trace id of the active OTel span — the same id Langfuse shows in its UI."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def score_trace(
    trace_id: str | None,
    *,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Attach a numeric score to *trace_id* (no-op when Langfuse is off or id is None)."""
    if _client is None or trace_id is None:
        return
    try:
        _client.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment,
            score_id=f"{trace_id}:{name}",
        )
    except Exception as exc:  # observability must never break a request
        _log.warning("Langfuse score %s failed: %s", name, exc)


def score_current_trace(name: str, value: float, comment: str | None = None) -> None:
    """Attach a numeric score to the trace of the active span."""
    score_trace(current_trace_id(), name=name, value=value, comment=comment)


def flush() -> None:
    """Flush buffered spans and scores to Langfuse (no-op when off)."""
    if _client is not None:
        _client.flush()


def _reset_for_tests() -> None:
    global _client
    _client = None


def _to_str(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
