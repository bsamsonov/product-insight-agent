from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from poc.core.errors import DomainError
from poc.llm.budget import BudgetExceededError
from poc.llm.errors import LLMProviderError


class GuardrailViolation(DomainError):  # noqa: N818  (name is part of public API spec)
    """Raised when a guardrail check blocks a request."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"guardrail.{reason}")
        self.reason = reason


def register_error_handlers(app: FastAPI) -> None:
    """Register all domain-level exception handlers on the FastAPI app.

    Handler precedence: FastAPI matches the *most specific* exception class
    registered first.  GuardrailViolation is a DomainError subclass, so it
    must be registered before the generic DomainError handler.
    """

    @app.exception_handler(GuardrailViolation)
    async def guardrail_handler(request: Request, exc: GuardrailViolation) -> JSONResponse:
        # Spec: return HTTP 200 with status=refused, not 4xx.
        # Reason: client logic should distinguish "request worked but was
        # refused" from "something broke".  Monitoring treats 2xx as
        # successful transactions (no false-positive error-rate alerts).
        return JSONResponse(
            status_code=200,
            content={"status": "refused", "reason": f"guardrail.{exc.reason}"},
        )

    @app.exception_handler(BudgetExceededError)
    async def budget_handler(request: Request, exc: BudgetExceededError) -> JSONResponse:
        # S3.T4 hard-stop: budget breach is a throttling condition, not a client
        # error — 429 tells the caller to back off (or top up), mirroring provider
        # rate-limit semantics. Registered before DomainError (it's a subclass).
        return JSONResponse(
            status_code=429,
            content={"status": "error", "type": "budget_exceeded", "message": str(exc)},
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "type": "domain_error", "message": str(exc)},
        )

    @app.exception_handler(LLMProviderError)
    async def llm_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
        status = 429 if exc.status_code == 429 else 502
        return JSONResponse(
            status_code=status,
            content={"status": "error", "type": "llm_error", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log at ERROR level in production; return opaque message to client.
        import logging

        logging.getLogger(__name__).error("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "type": "internal", "message": "internal server error"},
        )
