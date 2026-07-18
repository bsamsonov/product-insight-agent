from __future__ import annotations

import json

from poc.guardrails.input_checks import check_input
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class InputGuardrailsMiddleware(BaseHTTPMiddleware):
    """Validate the request body for ``/ask`` endpoints before routing.

    Design notes
    ------------
    This middleware is *optional* — ``main.py`` already calls ``check_input()``
    inside the handler.  The middleware exists to demonstrate the middleware
    pattern (analogous to a Spring ``OncePerRequestFilter``) so that guardrails
    can be applied uniformly without touching individual handler code.

    Body consumption
    ----------------
    Starlette's ``BaseHTTPMiddleware`` buffers the full body before calling the
    next handler, so reading it here does NOT consume it for downstream use.
    We manually restore the body via a custom ``receive`` callable so that
    FastAPI's request-body parsing still works normally after this check.

    OWASP LLM01 / LLM02 coverage
    -----------------------------
    - LLM01 (Prompt Injection): ``check_input`` detects common injection keywords.
    - LLM06 (Sensitive Info Disclosure): PII is redacted before reaching the LLM.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Only intercept POST requests to /ask paths
        if request.method == "POST" and request.url.path.startswith("/ask"):
            body_bytes = await request.body()

            # Attempt to extract the question field for guardrail checks
            question: str | None = None
            try:
                payload = json.loads(body_bytes)
                question = payload.get("question") if isinstance(payload, dict) else None
            except (json.JSONDecodeError, ValueError):
                pass

            if question is not None:
                result = check_input(question)
                if not result.passed:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "detail": "Input rejected by guardrails",
                            "violations": result.violations,
                        },
                    )

            # Restore body so that downstream FastAPI parsing works
            async def _receive():
                return {"type": "http.request", "body": body_bytes}

            request._receive = _receive  # type: ignore[attr-defined]

        return await call_next(request)
