"""Tests for poc.observability.tracing — @traced decorator and record_llm_call."""

from __future__ import annotations

import functools
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from poc.observability import langfuse_client
from poc.observability.tracing import record_llm_call, traced

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Return a fresh isolated TracerProvider + InMemorySpanExporter pair."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # SimpleSpanProcessor exports synchronously — no need to flush in tests.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _traced_with_provider(
    span_name: str,
    provider: TracerProvider,
    attributes: dict | None = None,
):
    """Helper: like @traced but injects a specific TracerProvider.

    This avoids touching the global OTel singleton (which can only be set once
    per process), making tests fully isolated.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            tracer = provider.get_tracer("poc")
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                return await fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# @traced decorator — integration via _traced_with_provider
# ---------------------------------------------------------------------------


class TestTracedDecorator:
    async def test_returns_function_result(self):
        """@traced must not alter the return value of the wrapped function."""
        provider, _ = _make_provider()

        @_traced_with_provider("test.add", provider)
        async def add(a: int, b: int) -> int:
            return a + b

        assert await add(2, 3) == 5

    async def test_span_name_is_set(self):
        """The span created by the decorator must carry the given name."""
        provider, exporter = _make_provider()

        @_traced_with_provider("my.operation", provider)
        async def noop() -> str:
            return "ok"

        result = await noop()
        assert result == "ok"

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "my.operation"

    async def test_attributes_are_attached_to_span(self):
        """Static attributes must appear on the finished span."""
        provider, exporter = _make_provider()

        @_traced_with_provider(
            "tagged.op", provider, attributes={"component": "retriever", "version": "2"}
        )
        async def fetch() -> None:
            return None

        await fetch()

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs["component"] == "retriever"
        assert attrs["version"] == "2"

    async def test_no_attributes_does_not_crash(self):
        """@traced without attributes= must work without error."""
        provider, exporter = _make_provider()

        @_traced_with_provider("bare.op", provider)
        async def bare() -> int:
            return 42

        result = await bare()
        assert result == 42

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

    async def test_preserves_function_metadata(self):
        """functools.wraps must keep __name__ and __doc__."""

        @traced("meta.op")
        async def documented_fn() -> None:
            """My docstring."""

        assert documented_fn.__name__ == "documented_fn"
        assert documented_fn.__doc__ == "My docstring."

    async def test_propagates_exception(self):
        """Exceptions raised inside the decorated function must bubble up."""
        provider, _ = _make_provider()

        @_traced_with_provider("error.op", provider)
        async def boom() -> None:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            await boom()

    async def test_span_is_finished_after_exception(self):
        """The span must be closed even when the wrapped function raises."""
        provider, exporter = _make_provider()

        @_traced_with_provider("error.finished", provider)
        async def boom() -> None:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            await boom()

        # Span must still be exported (finished), not dangling
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "error.finished"


# ---------------------------------------------------------------------------
# record_llm_call
# ---------------------------------------------------------------------------


class TestRecordLlmCall:
    async def test_sets_gen_ai_attributes(self):
        """All mandatory gen_ai.* attributes must be set on the span."""
        provider, exporter = _make_provider()

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("llm-call") as span:
            record_llm_call(
                span,
                model="claude-sonnet-4-5",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.002,
                provider="anthropic",
            )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs["gen_ai.system"] == "anthropic"
        assert attrs["gen_ai.request.model"] == "claude-sonnet-4-5"
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50
        # Cost is mapped via Langfuse-native cost_details (JSON), not the ad-hoc cost.usd.
        cost = json.loads(attrs["langfuse.observation.cost_details"])
        assert cost == {"total": pytest.approx(0.002)}

    async def test_records_input_and_output(self):
        """Prompt messages and completion text land on the Langfuse input/output keys."""
        provider, exporter = _make_provider()

        messages = [{"role": "user", "content": "héllo"}]
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("llm-call") as span:
            record_llm_call(
                span,
                model="claude-sonnet-4-5",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
                provider="anthropic",
                input_messages=messages,
                output_text="hi there",
            )

        attrs = exporter.get_finished_spans()[0].attributes
        # input is JSON-serialized (OTel attrs cannot hold list[dict]); non-ASCII preserved.
        assert json.loads(attrs["langfuse.observation.input"]) == messages
        assert "héllo" in attrs["langfuse.observation.input"]
        assert attrs["langfuse.observation.output"] == "hi there"

    async def test_input_output_omitted_when_none(self):
        """Without input_messages/output_text the Langfuse payload keys are not set."""
        provider, exporter = _make_provider()

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("llm-call") as span:
            record_llm_call(
                span,
                model="gpt-4o",
                input_tokens=200,
                output_tokens=80,
                cost_usd=0.002,
                provider="openai",
            )

        attrs = exporter.get_finished_spans()[0].attributes
        assert "langfuse.observation.input" not in attrs
        assert "langfuse.observation.output" not in attrs

    async def test_cost_usd_omitted_when_none(self):
        """When cost_usd=None the cost_details attribute must NOT be set."""
        provider, exporter = _make_provider()

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("llm-call") as span:
            record_llm_call(
                span,
                model="gpt-4o",
                input_tokens=200,
                output_tokens=80,
                cost_usd=None,
                provider="openai",
            )

        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes
        assert "langfuse.observation.cost_details" not in attrs
        assert "cost.usd" not in attrs
        assert attrs["gen_ai.system"] == "openai"

    async def test_zero_tokens_accepted(self):
        """Zero token counts are valid (e.g., cached responses)."""
        provider, exporter = _make_provider()

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("cached-call") as span:
            record_llm_call(
                span,
                model="claude-haiku-4-5",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                provider="anthropic",
            )

        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 0
        assert attrs["gen_ai.usage.output_tokens"] == 0
        cost = json.loads(attrs["langfuse.observation.cost_details"])
        assert cost == {"total": pytest.approx(0.0)}


# ---------------------------------------------------------------------------
# Langfuse client wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_langfuse_client():
    langfuse_client._reset_for_tests()
    yield
    langfuse_client._reset_for_tests()


@pytest.mark.usefixtures("_no_langfuse_client")
class TestLangfuseClient:
    def _set_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    def test_not_attached_without_env(self, monkeypatch):
        """No LANGFUSE_* env → no export, no client, no errors."""
        for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        provider = TracerProvider()
        assert langfuse_client.attach_to_provider(provider) is False
        assert langfuse_client.enabled() is False

    def test_partial_env_is_not_configured(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        assert langfuse_client.is_configured() is False

    def test_attaches_sdk_processor_to_our_provider(self, monkeypatch):
        """Full env → the Langfuse SDK span processor is added to *our* provider."""
        self._set_env(monkeypatch)
        provider = TracerProvider()
        assert langfuse_client.attach_to_provider(provider) is True
        assert langfuse_client.enabled() is True
        processors = provider._active_span_processor._span_processors
        assert any(type(p).__name__ == "LangfuseSpanProcessor" for p in processors)

    def test_scores_are_noop_when_disabled(self):
        langfuse_client.score_trace("0" * 32, name="judge_score", value=0.9)
        langfuse_client.score_current_trace("judge_score", 0.9)
        langfuse_client.flush()

    def test_score_trace_calls_sdk(self, monkeypatch):
        calls = []

        class _FakeClient:
            def create_score(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(langfuse_client, "_client", _FakeClient())
        langfuse_client.score_trace("abc", name="judge_score", value=1, comment="ok")
        assert calls == [
            {
                "trace_id": "abc",
                "name": "judge_score",
                "value": 1.0,
                "data_type": "NUMERIC",
                "comment": "ok",
                "score_id": "abc:judge_score",
            }
        ]

    def test_score_failure_is_swallowed(self, monkeypatch):
        class _Boom:
            def create_score(self, **kwargs):
                raise RuntimeError("network down")

        monkeypatch.setattr(langfuse_client, "_client", _Boom())
        langfuse_client.score_trace("abc", name="x", value=0.1)  # must not raise

    def test_current_trace_id_inside_span(self):
        provider, _ = _make_provider()
        assert langfuse_client.current_trace_id() is None
        with provider.get_tracer("t").start_as_current_span("root") as span:
            expected = format(span.get_span_context().trace_id, "032x")
            assert langfuse_client.current_trace_id() == expected

    def test_in_active_trace(self):
        provider, _ = _make_provider()
        assert langfuse_client.in_active_trace() is False
        with provider.get_tracer("t").start_as_current_span("root"):
            assert langfuse_client.in_active_trace() is True

    def test_set_trace_attributes(self):
        provider, exporter = _make_provider()
        with provider.get_tracer("t").start_as_current_span("root") as span:
            langfuse_client.set_trace_attributes(
                span,
                name="ask",
                session_id="s1",
                user_id="acme",
                tags=["agent"],
                input={"question": "q"},
                output="answer",
            )
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["langfuse.trace.name"] == "ask"
        assert attrs["session.id"] == "s1"
        assert attrs["user.id"] == "acme"
        assert list(attrs["langfuse.trace.tags"]) == ["agent"]
        assert json.loads(attrs["langfuse.trace.input"]) == {"question": "q"}
        assert attrs["langfuse.trace.output"] == "answer"
