"""/ask must use only the tenant validated by TenantMiddleware.

Regression: the handler used to resolve the tenant as ``X-Tenant`` header > body
``tenant`` > "default", bypassing the allowlist. An arbitrary tenant then reached the
audit log, Langfuse and the per-tenant daily budget (a fresh budget per made-up name).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import api.main  # noqa: F401  (side-effect import — populates sys.modules)
import pytest


@pytest.fixture()
def ctx(monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    # Point the rate limiter at an unreachable Redis → fail-open, no real server needed.
    monkeypatch.setenv("POC_REDIS_URL", "redis://127.0.0.1:1/0")

    response = MagicMock(content="ok", cost_usd=0.0)
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value=response)
    mock_audit = MagicMock()
    mock_audit.log = AsyncMock()

    with (
        patch("api.main.get_provider", return_value=mock_provider),
        patch("api.main.LLMSettings", return_value=MagicMock(default_provider="omniroute")),
        patch("api.main.AuditLogger", return_value=mock_audit),
        patch("api.main._build_agent", return_value=None),  # direct-LLM path
    ):
        from api.main import app

        with TestClient(app) as c:
            yield c, mock_audit


def _logged_tenant(mock_audit: MagicMock) -> str:
    event = mock_audit.log.await_args.args[0]
    return event.tenant


def test_body_tenant_is_ignored(ctx):
    client, audit = ctx
    resp = client.post(
        "/ask",
        json={"question": "How is the battery?", "tenant": "evil"},
        headers={"X-Tenant-Id": "acme"},
    )
    assert resp.status_code == 200
    assert _logged_tenant(audit) == "acme"


def test_legacy_x_tenant_header_is_ignored(ctx):
    client, audit = ctx
    resp = client.post(
        "/ask", json={"question": "How is the battery?"}, headers={"X-Tenant": "evil"}
    )
    assert resp.status_code == 200
    assert _logged_tenant(audit) == "default"


def test_unknown_tenant_header_rejected(ctx):
    client, audit = ctx
    resp = client.post(
        "/ask", json={"question": "How is the battery?"}, headers={"X-Tenant-Id": "evil"}
    )
    assert resp.status_code == 401
    audit.log.assert_not_awaited()


def test_rate_limit_redis_uses_poc_redis_url(monkeypatch: pytest.MonkeyPatch):
    from api.middleware import rate_limit

    monkeypatch.setenv("POC_REDIS_URL", "redis://redis-host:6380/2")
    with patch("redis.Redis.from_url") as from_url:
        rate_limit._make_default_redis()
    assert from_url.call_args.args[0] == "redis://redis-host:6380/2"
