from __future__ import annotations

from unittest.mock import patch

from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.tenant import TenantMiddleware
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeRedis:
    """Minimal in-memory Redis substitute for unit tests.

    Implements only the subset of the redis-py API that RateLimitMiddleware
    uses: ``get``, ``set``, and ``pipeline`` (with ``execute``).
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: object) -> None:
        self._store[key] = str(value)

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple]] = []

    def get(self, key: str) -> _FakePipeline:
        self._commands.append(("get", (key,)))
        return self

    def set(self, key: str, value: object) -> _FakePipeline:
        self._commands.append(("set", (key, value)))
        return self

    def execute(self) -> list:
        results = []
        for cmd, args in self._commands:
            if cmd == "get":
                results.append(self._redis.get(args[0]))
            elif cmd == "set":
                self._redis.set(args[0], args[1])
                results.append(True)
        return results


def make_app(fake_redis: FakeRedis | None = None) -> FastAPI:
    """Build an isolated FastAPI app with TenantMiddleware + RateLimitMiddleware."""
    app = FastAPI()

    # RateLimitMiddleware reads tenant from ContextVar set by TenantMiddleware.
    # Add RateLimitMiddleware first so TenantMiddleware runs first (LIFO order).
    app.add_middleware(RateLimitMiddleware, redis_client=fake_redis)
    app.add_middleware(TenantMiddleware)

    @app.post("/ask")
    async def ask():
        return {"answer": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_requests_pass_within_burst():
    """The first 10 POST /ask requests succeed when burst=10."""
    fake_redis = FakeRedis()

    with (
        patch("api.middleware.rate_limit._get_burst", return_value=10.0),
        patch("api.middleware.rate_limit._get_rps", return_value=5.0),
        TestClient(make_app(fake_redis)) as c,
    ):
        for i in range(10):
            resp = c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
            assert resp.status_code == 200, f"Request {i + 1} should pass, got {resp.status_code}"


def test_eleventh_request_returns_429():
    """After exhausting the burst bucket, the 11th request returns HTTP 429."""
    fake_redis = FakeRedis()

    with (
        patch("api.middleware.rate_limit._get_burst", return_value=10.0),
        patch("api.middleware.rate_limit._get_rps", return_value=5.0),
        TestClient(make_app(fake_redis)) as c,
    ):
        for _ in range(10):
            c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})

        resp = c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        assert resp.status_code == 429
        body = resp.json()
        assert body["status"] == "error"
        assert body["type"] == "rate_limited"
        assert "retry_after_s" in body


def test_429_response_contains_retry_after():
    """The 429 body includes a numeric retry_after_s field."""
    fake_redis = FakeRedis()

    with (
        patch("api.middleware.rate_limit._get_burst", return_value=1.0),
        patch("api.middleware.rate_limit._get_rps", return_value=1.0),
        TestClient(make_app(fake_redis)) as c,
    ):
        c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})  # consume the 1 token
        resp = c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        assert resp.status_code == 429
        body = resp.json()
        assert isinstance(body["retry_after_s"], float)
        assert body["retry_after_s"] > 0


def test_non_ask_paths_bypass_rate_limit():
    """GET /health is never rate-limited, even when bucket is empty."""
    fake_redis = FakeRedis()
    # Manually empty the bucket
    fake_redis.set("rate_limit:default:tokens", "0.0")
    fake_redis.set("rate_limit:default:last_refill", "1000000000.0")

    with (
        patch("api.middleware.rate_limit._get_burst", return_value=1.0),
        patch("api.middleware.rate_limit._get_rps", return_value=0.0),  # no refill
        TestClient(make_app(fake_redis)) as c,
    ):
        resp = c.get("/health", headers={"X-Tenant-Id": "default"})
        assert resp.status_code == 200


def test_redis_unavailable_allows_request():
    """When Redis raises an exception, the request is allowed through (fail-open)."""

    class BrokenRedis:
        def pipeline(self):
            raise ConnectionError("Redis not available")

    broken_redis = BrokenRedis()
    app = make_app(broken_redis)  # type: ignore[arg-type]

    with TestClient(app) as c:
        resp = c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
    assert resp.status_code == 200


def test_bucket_refills_after_elapsed_time():
    """AC: after the bucket empties, it returns to normal once time elapses.

    The bucket is emptied, then ``last_refill`` is rewound ~1s into the past so
    the next request sees ``elapsed * rps`` fresh tokens and is allowed again —
    exercising the time-based refill branch (no real sleep needed).
    """
    fake_redis = FakeRedis()

    with (
        patch("api.middleware.rate_limit._get_burst", return_value=5.0),
        patch("api.middleware.rate_limit._get_rps", return_value=5.0),
        TestClient(make_app(fake_redis)) as c,
    ):
        # Drain the burst bucket → next request is throttled.
        for _ in range(5):
            c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        assert c.post("/ask", headers={"X-Tenant-Id": "default"}, json={}).status_code == 429

        # Rewind last_refill ~1s into the past: 1s * 5 rps = 5 tokens refilled.
        past = float(fake_redis.get("rate_limit:default:last_refill")) - 1.0
        fake_redis.set("rate_limit:default:last_refill", past)

        resp = c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        assert resp.status_code == 200


def test_different_tenants_have_independent_buckets():
    """Rate limit is per-tenant; exhausting one tenant does not affect another."""
    fake_redis = FakeRedis()

    with (
        patch("api.middleware.rate_limit._get_burst", return_value=2.0),
        patch("api.middleware.rate_limit._get_rps", return_value=0.0),  # no refill
        TestClient(make_app(fake_redis)) as c,
    ):
        # Exhaust 'default' tenant
        c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        resp_default = c.post("/ask", headers={"X-Tenant-Id": "default"}, json={})
        assert resp_default.status_code == 429

        # 'acme' tenant still has a full bucket
        resp_acme = c.post("/ask", headers={"X-Tenant-Id": "acme"}, json={})
        assert resp_acme.status_code == 200
