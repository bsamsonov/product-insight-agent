from __future__ import annotations

import logging
import os

from poc.core.tenant import current_tenant_token, reset_tenant
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_DEFAULT_ALLOWED = "default,acme,demo"


def _get_allowed_tenants() -> frozenset[str]:
    raw = os.environ.get("POC_ALLOWED_TENANTS", _DEFAULT_ALLOWED)
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


class TenantMiddleware(BaseHTTPMiddleware):
    """Extract and validate the tenant from the ``X-Tenant-Id`` request header.

    Design notes
    ------------
    Analogous to a Spring ``OncePerRequestFilter`` that reads a ``X-Tenant-Id``
    header and populates a ``ThreadLocal`` (here: a ``ContextVar``) for the
    duration of the request.

    Resolution order
    ----------------
    1. ``X-Tenant-Id`` header — explicit tenant from caller.
    2. Fallback to ``"default"`` if the header is absent.

    Unknown tenants receive HTTP 401 — they are not authorised to use the API
    regardless of any other credentials they may hold.

    ContextVar safety
    -----------------
    ``ContextVar`` is the async-safe equivalent of ``ThreadLocal``.  Each
    asyncio task has its own copy of the context, so setting the tenant here
    does NOT bleed into other concurrent requests.  We still reset via
    ``try/finally`` to restore the previous value in case middleware is nested.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        tenant = request.headers.get("X-Tenant-Id", "default")

        allowed = _get_allowed_tenants()
        if tenant not in allowed:
            logger.warning("rejected unknown tenant=%s", tenant)
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "type": "unauthorized",
                    "message": f"unknown tenant: {tenant}",
                },
            )

        token = current_tenant_token(tenant)
        try:
            response = await call_next(request)
        finally:
            reset_tenant(token)

        return response
