from __future__ import annotations

import logging
import os
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def _get_rps() -> float:
    return float(os.environ.get("POC_RATE_LIMIT_RPS", "5.0"))


def _get_burst() -> float:
    return float(os.environ.get("POC_RATE_LIMIT_BURST", "10"))


def _make_default_redis():
    """Create the default Redis client from env (REDIS_URL or localhost)."""
    import redis  # type: ignore[import-untyped]

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(
        url, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant token-bucket rate limiter backed by Redis.

    Design notes
    ------------
    Analogous to Spring's ``RateLimiter`` (Resilience4j) or Bucket4j, but
    implemented as a Starlette middleware (≈ ``OncePerRequestFilter``) so it
    applies uniformly to all ``/ask*`` endpoints without touching handler code.

    Token-bucket algorithm
    ----------------------
    Each tenant has two Redis keys:

    - ``rate_limit:{tenant}:tokens``      — current token count (float)
    - ``rate_limit:{tenant}:last_refill`` — unix timestamp of last refill (float)

    On every request the middleware:
    1. Reads both keys in a pipeline (single round-trip).
    2. Computes elapsed time since last refill.
    3. Adds ``min(elapsed * rps, burst)`` tokens (capped at burst).
    4. If tokens >= 1: subtracts 1 and saves; request is **allowed**.
    5. If tokens < 1: returns HTTP 429 without touching the bucket.

    Atomicity caveat
    ----------------
    This implementation uses a read-then-write pipeline which is NOT atomic.
    Under heavy concurrency a true Lua script (``redis.call``) would be required.
    For a POC with low concurrency this approximation is acceptable; the comment
    below marks where to plug in a Lua script for production.

    Graceful degradation
    --------------------
    If Redis is unavailable the middleware logs a WARNING and allows the request
    through — fail-open is the correct default for a non-safety-critical rate
    limiter in a POC.  A production system might fail-closed or use a local
    in-process fallback.

    Only POST /ask* paths are rate-limited; all other paths pass through.
    """

    def __init__(self, app: Any, redis_client: Any = None) -> None:
        super().__init__(app)
        self._redis = redis_client  # injected for tests; lazy-init otherwise

    def _get_redis(self) -> Any:
        if self._redis is None:
            self._redis = _make_default_redis()
        return self._redis

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Only rate-limit POST /ask* paths
        if not (request.method == "POST" and request.url.path.startswith("/ask")):
            return await call_next(request)

        try:
            return await self._apply_rate_limit(request, call_next)
        except Exception as exc:
            # Redis unavailable (connection error, timeout, etc.) → fail-open
            logger.warning("rate limit Redis unavailable, allowing request: %s", exc)
            return await call_next(request)

    async def _apply_rate_limit(self, request: Request, call_next):
        from poc.core.tenant import get_tenant

        tenant = get_tenant()
        rps = _get_rps()
        burst = _get_burst()

        redis = self._get_redis()
        key_tokens = f"rate_limit:{tenant}:tokens"
        key_last = f"rate_limit:{tenant}:last_refill"

        now = time.time()

        # Read current state in a single round-trip
        pipe = redis.pipeline()
        pipe.get(key_tokens)
        pipe.get(key_last)
        raw_tokens, raw_last = pipe.execute()

        tokens: float = float(raw_tokens) if raw_tokens is not None else burst
        last_refill: float = float(raw_last) if raw_last is not None else now

        # Refill bucket
        elapsed = max(0.0, now - last_refill)
        tokens = min(tokens + elapsed * rps, burst)

        if tokens < 1.0:
            # Compute time until 1 token will be available.
            # Guard against rps=0 (e.g. in tests) to avoid ZeroDivisionError.
            retry_after = (1.0 - tokens) / rps if rps > 0 else 3600.0
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "type": "rate_limited",
                    "message": f"rate limit exceeded, retry after {retry_after:.1f}s",
                    "retry_after_s": retry_after,
                },
            )

        # Consume one token and persist new state
        # NOTE: for production, replace with a Lua script for true atomicity.
        new_tokens = tokens - 1.0
        pipe2 = redis.pipeline()
        pipe2.set(key_tokens, new_tokens)
        pipe2.set(key_last, now)
        pipe2.execute()

        return await call_next(request)
