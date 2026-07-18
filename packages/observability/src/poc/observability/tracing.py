from __future__ import annotations

import base64
import functools
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_log = logging.getLogger(__name__)

_provider: TracerProvider | None = None


def _make_langfuse_processor() -> SpanProcessor | None:
    """Build a BatchSpanProcessor that ships spans to Langfuse over OTLP/HTTP.

    Langfuse v3 ingests OpenTelemetry natively on ``/api/public/otel`` and maps
    ``gen_ai.*`` spans to *generations* (model, tokens, cost) automatically. Enabled
    only when ``LANGFUSE_HOST`` is set; otherwise returns None and the app runs with
    no external export (and no errors).
    """
    host = os.getenv("LANGFUSE_HOST")
    if not host:
        return None

    # Lazy import: the OTLP exporter is an optional dependency.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {token}"},
    )
    _log.info("OTel: exporting spans to Langfuse at %s", endpoint)
    return BatchSpanProcessor(exporter)


def configure(
    service_name: str = "product-insight-agent",
    export_to_console: bool = False,
    in_memory: bool = False,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing. Call once at startup.

    Wires up to three exporters, depending on configuration:

    - ``in_memory=True`` → ``InMemorySpanExporter`` (tests only; skips the others).
    - ``export_to_console`` / ``OTEL_CONSOLE_EXPORT=true`` → ``ConsoleSpanExporter``.
    - ``LANGFUSE_HOST`` set → OTLP/HTTP exporter to Langfuse (see _make_langfuse_processor).
    """
    global _provider

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if in_memory:
        # Used in tests
        exporter = InMemorySpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        if export_to_console or os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        langfuse_processor = _make_langfuse_processor()
        if langfuse_processor is not None:
            provider.add_span_processor(langfuse_processor)

    trace.set_tracer_provider(provider)
    _provider = provider
    _log.info("OTel tracing initialized for service: %s", service_name)
    return provider


def get_tracer(name: str = "poc") -> trace.Tracer:
    return trace.get_tracer(name)


def traced(
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> Callable:
    """Decorator that wraps an async function in an OTel span.

    Usage::

        @traced("agent.retrieve", attributes={"retriever": "hybrid"})
        async def retrieve(query: str) -> list[str]:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                return await fn(*args, **kwargs)

        return wrapper

    return decorator


def record_llm_call(
    span: trace.Span,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None,
    provider: str,
    input_messages: list[dict[str, str]] | None = None,
    output_text: str | None = None,
) -> None:
    """Set gen_ai.* attributes on *span* per OTel semantic conventions.

    Attribute names follow the OpenTelemetry Semantic Conventions for Generative AI
    (https://opentelemetry.io/docs/specs/semconv/gen-ai/), plus a few Langfuse-native
    ``langfuse.observation.*`` keys for fields OTel semconv does not (yet) cover —
    prompt/completion payloads and cost. OTel attribute values must be primitives or
    homogeneous arrays, so structured values are JSON-serialized to strings.

    Args:
        span: The active OTel span to annotate.
        model: The model identifier, e.g. ``"claude-sonnet-4-5"``.
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.
        cost_usd: Estimated cost in US dollars, or ``None`` if unknown.
        provider: LLM provider name, e.g. ``"anthropic"`` or ``"openai"``.
        input_messages: The prompt messages ([{"role", "content"}, ...]); rendered as
            the generation's input in Langfuse. Omitted when None.
        output_text: The model's completion text; rendered as the generation's output.
            Omitted when None.
    """
    # gen_ai.operation.name marks the span as an LLM generation so Langfuse (and other
    # OTel-aware backends) classify it correctly instead of a plain internal span.
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    # Cost: Langfuse maps langfuse.observation.cost_details (JSON, "total" key), NOT the
    # ad-hoc "cost.usd". Without this Langfuse ignores our estimate and recomputes from
    # its own Model Definitions.
    if cost_usd is not None:
        span.set_attribute("langfuse.observation.cost_details", json.dumps({"total": cost_usd}))
    # Prompt/completion payloads. Langfuse maps these to the generation's input/output.
    if input_messages is not None:
        span.set_attribute(
            "langfuse.observation.input", json.dumps(input_messages, ensure_ascii=False)
        )
    if output_text is not None:
        span.set_attribute("langfuse.observation.output", output_text)
