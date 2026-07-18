from __future__ import annotations

import pytest
from api.middleware.tenant import TenantMiddleware
from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(TenantMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


def test_valid_tenant_header():
    with TestClient(make_app()) as c:
        resp = c.get("/ping", headers={"X-Tenant-Id": "default"})
    assert resp.status_code == 200


def test_valid_tenant_acme():
    with TestClient(make_app()) as c:
        resp = c.get("/ping", headers={"X-Tenant-Id": "acme"})
    assert resp.status_code == 200


def test_valid_tenant_demo():
    with TestClient(make_app()) as c:
        resp = c.get("/ping", headers={"X-Tenant-Id": "demo"})
    assert resp.status_code == 200


def test_missing_header_defaults_to_default():
    with TestClient(make_app()) as c:
        resp = c.get("/ping")
    assert resp.status_code == 200


def test_unknown_tenant_rejected():
    with TestClient(make_app()) as c:
        resp = c.get("/ping", headers={"X-Tenant-Id": "unknown_xyz"})
    assert resp.status_code == 401
    assert resp.json()["type"] == "unauthorized"


def test_unknown_tenant_error_body():
    with TestClient(make_app()) as c:
        resp = c.get("/ping", headers={"X-Tenant-Id": "evil_corp"})
    body = resp.json()
    assert body["status"] == "error"
    assert "evil_corp" in body["message"]


def test_allowed_tenants_env_override(monkeypatch: pytest.MonkeyPatch):
    """POC_ALLOWED_TENANTS env var changes the allowed set at call time."""
    monkeypatch.setenv("POC_ALLOWED_TENANTS", "custom_tenant")

    with TestClient(make_app()) as c:
        # Previously-valid "default" should now be rejected
        resp = c.get("/ping", headers={"X-Tenant-Id": "default"})
    assert resp.status_code == 401

    with TestClient(make_app()) as c:
        # New custom tenant should be accepted
        resp = c.get("/ping", headers={"X-Tenant-Id": "custom_tenant"})
    assert resp.status_code == 200
