"""Client-side request pacing per provider.

Free tiers enforce requests-per-minute limits. One agent answer costs ~6 LLM calls and an
eval case with LLM metrics ~10, so bursts hit HTTP 429 long before the daily quota. Pacing
calls evenly is cheaper than retrying them.

Configure with ``POC_<PROVIDER>_RPM`` (e.g. ``POC_GEMINI_RPM=12``) or ``POC_LLM_RPM`` for
all providers. Unset or 0 means no pacing.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextvars import ContextVar

# Seconds spent waiting for a pacing slot in the current request. Holds a mutable list so
# waits recorded inside child tasks (LangGraph nodes) are visible to the caller.
_pacing_wait: ContextVar[list[float] | None] = ContextVar("poc_llm_pacing_wait", default=None)


def start_pacing_meter() -> list[float]:
    """Start accumulating pacing waits for the current request; returns the meter."""
    meter = [0.0]
    _pacing_wait.set(meter)
    return meter


class RateLimiter:
    """Spaces calls at least ``60 / rpm`` seconds apart (process-wide, per provider)."""

    def __init__(self, rpm: float) -> None:
        self._interval = 60.0 / rpm
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            meter = _pacing_wait.get()
            if meter is not None:
                meter[0] += wait
            await asyncio.sleep(wait)


_limiters: dict[str, RateLimiter | None] = {}


def _configured_rpm(provider: str) -> float:
    raw = os.getenv(f"POC_{provider.upper()}_RPM") or os.getenv("POC_LLM_RPM") or "0"
    try:
        return float(raw)
    except ValueError:
        return 0.0


def limiter_for(provider: str) -> RateLimiter | None:
    """Return the shared limiter for *provider*, or None when pacing is off."""
    if provider not in _limiters:
        rpm = _configured_rpm(provider)
        _limiters[provider] = RateLimiter(rpm) if rpm > 0 else None
    return _limiters[provider]


def reset() -> None:
    """Forget cached limiters (tests, or after changing the env)."""
    _limiters.clear()
