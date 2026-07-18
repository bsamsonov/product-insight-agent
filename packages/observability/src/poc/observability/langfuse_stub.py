"""No-op Langfuse tracer stub.

The real Langfuse SDK (``langfuse``) is an optional dependency that is NOT installed
in this POC environment.  This module provides a drop-in replacement that:

- Exposes the same surface that agent code calls (``trace``, ``span``, ``generation``).
- Returns lightweight sentinel objects so attribute access never raises.
- Emits a single DEBUG log on construction so operators know they are running
  without real Langfuse connectivity.

When the real SDK becomes available, replace ``LangfuseTracer`` with a thin wrapper
around ``langfuse.Langfuse`` and the sentinel objects with real ``StatefulSpanClient``
instances — the call sites in agent code need not change.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


class _NoOpContext:
    """Returned by trace()/span()/generation() — silently accepts any call."""

    def __getattr__(self, name: str) -> _NoOpContext:
        # Allows chaining: tracer.trace(...).end() without AttributeError
        return self

    def __call__(self, *args: object, **kwargs: object) -> _NoOpContext:
        return self

    def __enter__(self) -> _NoOpContext:
        return self

    def __exit__(self, *args: object) -> None:
        pass


_NOOP = _NoOpContext()


class LangfuseTracer:
    """No-op implementation of the Langfuse tracing interface.

    Instantiate once at application startup (or in a dependency-injection
    factory) and pass to any component that needs LLM-level observability.

    Example::

        tracer = LangfuseTracer()
        t = tracer.trace(name="pipeline-run", user_id="u-123")
        gen = t.generation(name="llm-call", model="claude-sonnet-4-5")
        gen.end(output="Hello!")
        t.update(metadata={"intent": "feature_request"})
    """

    def __init__(self) -> None:
        _log.debug(
            "LangfuseTracer: running in no-op mode (langfuse SDK not installed). "
            "Install `langfuse` and replace this stub to enable real tracing."
        )

    # ------------------------------------------------------------------
    # Public API — mirror of langfuse.Langfuse
    # ------------------------------------------------------------------

    def trace(self, **kwargs: object) -> _NoOpContext:
        """Start a new Langfuse trace (no-op)."""
        return _NOOP

    def span(self, **kwargs: object) -> _NoOpContext:
        """Start a new Langfuse span (no-op)."""
        return _NOOP

    def generation(self, **kwargs: object) -> _NoOpContext:
        """Start a new Langfuse generation span (no-op)."""
        return _NOOP

    def flush(self) -> None:
        """Flush pending events to Langfuse (no-op)."""

    def shutdown(self) -> None:
        """Gracefully shut down the client (no-op)."""
