"""Integration test: build_graph wires @traced onto every node (S3.T1).

Verifies that running the compiled agent graph emits one OTel span per node
(agent.intent, agent.plan, …). The instrumentation lives in graph.py, so this
asserts the wiring — not the decorator itself (covered by test_tracing.py).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once
from poc.agent.graph import build_graph, initial_state
from poc.llm.provider import LLMResponse


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Install a global TracerProvider backed by an in-memory exporter.

    @traced resolves the *global* provider via get_tracer(), so the graph test
    must set it globally. OTel guards set_tracer_provider with a set-once latch;
    we reset it around the test to keep runs deterministic and order-independent.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # SimpleSpanProcessor exports synchronously on span end — get_finished_spans()
    # is populated immediately, no force_flush needed (unlike BatchSpanProcessor).
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None
    trace.set_tracer_provider(provider)

    yield exporter

    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None


def _make_mock_llm() -> MagicMock:
    """LLM stub returning empty JSON — every node falls back gracefully, so the
    graph runs end-to-end and a span is recorded around each node regardless."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(
        return_value=LLMResponse(
            content="{}",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            latency_ms=100,
            finish_reason="stop",
        )
    )
    return mock_llm


async def test_graph_emits_span_per_node(span_exporter: InMemorySpanExporter) -> None:
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[])

    graph = build_graph(
        llm=_make_mock_llm(),
        retriever=retriever,
        intent_model="m",
        plan_model="m",
        summarize_model="m",
        judge_model="m",
    )

    await graph.ainvoke(initial_state("What are common shoe complaints?"))

    span_names = {span.name for span in span_exporter.get_finished_spans()}
    expected = {
        "agent.intent",
        "agent.plan",
        "agent.retrieve",
        "agent.cluster",
        "agent.summarize",
        "agent.judge",
    }
    assert expected <= span_names, f"missing spans: {expected - span_names}"
