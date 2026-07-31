from __future__ import annotations

import logging
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import yaml
from poc.llm.context import current_budget_context
from poc.llm.errors import LLMProviderError, LLMRateLimitError
from poc.llm.provider import LLMMessage, LLMProvider, LLMResponse
from pydantic import BaseModel

_log = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent.parent.parent / "data" / "router.yaml"


def _find_config() -> Path:
    """Locate router.yaml relative to this file (works from the installed package)."""
    # src/poc/llm/router.py  →  packages/llm/data/router.yaml
    candidate = Path(__file__).parent.parent.parent.parent.parent / "data" / "router.yaml"
    if candidate.exists():
        return candidate
    # Fallback: look next to the package root
    here = Path(__file__).parent
    for _ in range(6):
        maybe = here / "data" / "router.yaml"
        if maybe.exists():
            return maybe
        here = here.parent
    raise FileNotFoundError("router.yaml not found; pass config_path explicitly")


class _RouteEndpoint(BaseModel):
    provider: str
    model: str
    max_tokens: int


class _RouteConfig(BaseModel):
    primary: _RouteEndpoint
    fallback: _RouteEndpoint


class RoutedLLM:
    """Role-based LLM router with automatic fallback on rate limits.

    Reads ``packages/llm/data/router.yaml`` to determine which provider+model
    to use for each *role* (classifier, planner, summarizer, judge, escalate).

    On ``LLMRateLimitError`` (HTTP 429) or ``LLMProviderError`` with
    ``status_code`` 429/503 from the primary provider the call is automatically
    retried against the fallback provider declared in the YAML.  If the fallback
    also fails the exception is re-raised.

    Usage::

        llm = RoutedLLM()
        response = await llm.route(
            "classifier",
            [LLMMessage(role="user", content="Is this product feedback?")],
            response_format=ClassificationResult,
        )
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        *,
        cache: Any | None = None,  # RedisResponseCache | FakeRedisCache
        budget_guard: Any | None = None,  # BudgetGuard | FakeBudgetGuard
    ) -> None:
        path = Path(config_path) if config_path is not None else _find_config()
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        self._routes: dict[str, _RouteConfig] = {
            role: _RouteConfig.model_validate(cfg) for role, cfg in raw.get("routes", {}).items()
        }
        # Budget limits declared in router.yaml (`budgets:`) — single source of truth
        # for callers constructing a BudgetGuard (see apps/api lifespan).
        self.budgets: dict[str, Any] = raw.get("budgets", {}) or {}
        # Cache provider instances by name to avoid re-creating the async HTTP client
        self._providers: dict[str, Any] = {}  # str → OpenAICompatibleProvider
        # Optional S3.T3/S3.T4 integrations. When set, route() consults the response
        # cache before dialing out and charges the budget guard after each real call.
        self._cache = cache
        self._budget_guard = budget_guard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        role: str,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Dispatch *messages* to the provider+model configured for *role*.

        Falls back to the secondary provider on rate-limit errors.

        Args:
            role: One of the keys defined in ``routes:`` in router.yaml
                  (classifier, planner, summarizer, judge, escalate).
            messages: Chat messages to send.
            response_format: Optional Pydantic model for structured JSON output.
            temperature: Sampling temperature (default 0.0 for determinism).

        Returns:
            :class:`~poc.llm.provider.LLMResponse` from the chosen provider.

        Raises:
            KeyError: Unknown role.
            LLMProviderError: Both primary and fallback failed.
        """
        if role not in self._routes:
            raise KeyError(f"Unknown LLM role '{role}'. Available: {sorted(self._routes)}")
        cfg = self._routes[role]

        # S3.T3: consult the response cache before dialing out. The key is derived
        # from the *primary* endpoint so a fallback response is served for the same
        # logical request. Cache hits cost $0 and are not charged against budgets.
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = self._cache._make_key(
                cfg.primary.provider,
                cfg.primary.model,
                messages,
                temperature,
                cfg.primary.max_tokens,
            )
            hit = await self._cache.get(cache_key)
            if hit is not None:
                _log.debug("RoutedLLM cache hit for role '%s'", role)
                return hit

        try:
            response = await self._call_endpoint(
                cfg.primary, messages, response_format=response_format, temperature=temperature
            )
        except (LLMRateLimitError, LLMProviderError) as exc:
            if not self._is_retriable(exc):
                raise
            _log.warning(
                "Primary provider '%s' failed for role '%s' (%s) — switching to fallback '%s'",
                cfg.primary.provider,
                role,
                exc,
                cfg.fallback.provider,
            )
            response = await self._call_endpoint(
                cfg.fallback, messages, response_format=response_format, temperature=temperature
            )

        # S3.T4: charge the budget *after* the call, once the real cost is known.
        # BudgetGuard commits the spend and raises BudgetExceededError on breach —
        # the error must propagate to the API layer (mapped to HTTP 429).
        await self._charge_budget(response)

        if cache_key is not None:
            await self._cache.set(cache_key, response)
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _charge_budget(self, response: LLMResponse) -> None:
        """Charge the call cost against the request-scoped budget context (S3.T4).

        Reads tenant and running request total from the contextvar set by
        ``agent.run()`` (see :mod:`poc.llm.context`). Outside a request scope the
        tenant defaults to "default" with a zero running total — the per-call and
        daily-tenant limits still apply.
        """
        cost = response.cost_usd or 0.0
        ctx = current_budget_context()
        if self._budget_guard is not None:
            tenant = ctx.tenant if ctx is not None else "default"
            running_total = ctx.total_usd if ctx is not None else 0.0
            await self._budget_guard.check_and_increment(
                tenant, cost, request_total_usd=running_total
            )
        if ctx is not None:
            ctx.add_cost(cost)

    def _get_provider(self, name: str) -> Any:
        """Return a cached ``OpenAICompatibleProvider`` for *name*."""
        if name not in self._providers:
            from poc.llm.openai_compatible import OpenAICompatibleProvider

            self._providers[name] = OpenAICompatibleProvider.from_env(name)
        return self._providers[name]

    async def _call_endpoint(
        self,
        endpoint: _RouteEndpoint,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel] | None,
        temperature: float,
    ) -> LLMResponse:
        provider = self._get_provider(endpoint.provider)
        _log.debug(
            "RoutedLLM → provider=%s model=%s max_tokens=%d",
            endpoint.provider,
            endpoint.model,
            endpoint.max_tokens,
        )
        return await provider.complete(
            messages,
            model=endpoint.model,
            max_tokens=endpoint.max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    @staticmethod
    def _is_retriable(exc: LLMProviderError) -> bool:
        """True if the exception warrants a fallback attempt."""
        if isinstance(exc, LLMRateLimitError):
            return True
        # Generic provider error with 429 or 503
        return exc.status_code in (429, 503)


# ---------------------------------------------------------------------------
# Deprecated shim — kept for backward-compatibility with code that already
# imports ModelRouter.  New code should use RoutedLLM directly.
# ---------------------------------------------------------------------------


class _RoutingRule(BaseModel):
    """Routing rule: if condition matches, use this model."""

    name: str
    model: str
    max_input_tokens: int | None = None
    intents: list[str] | None = None
    is_fallback: bool = False


# Public alias used in legacy imports
RoutingRule = _RoutingRule


class ModelRouter:
    """DEPRECATED — use :class:`RoutedLLM` instead.

    Legacy token/intent-based router.  This class now delegates to
    :class:`RoutedLLM` for structured role-based routing.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        fast_model: str,
        smart_model: str,
        expert_model: str,
        confidence_threshold: float = 0.6,
        token_threshold: int = 2000,
    ) -> None:
        warnings.warn(
            "ModelRouter is deprecated; use RoutedLLM instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._provider = provider
        self._fast = fast_model
        self._smart = smart_model
        self._expert = expert_model
        self._confidence_threshold = confidence_threshold
        self._token_threshold = token_threshold

    def select_model(
        self,
        *,
        intent: str | None = None,
        confidence: float | None = None,
        estimated_input_tokens: int = 0,
        force_model: str | None = None,
    ) -> str:
        if force_model:
            return force_model
        if confidence is not None and confidence < self._confidence_threshold:
            _log.debug("Escalating to expert model (confidence=%.2f)", confidence)
            return self._expert
        simple_intents = {"general", "other"}
        if estimated_input_tokens <= self._token_threshold and (
            intent is None or intent in simple_intents
        ):
            return self._fast
        return self._smart

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int,
        temperature: float = 0.0,
        response_format: type[BaseModel] | None = None,
        cache_control: bool = False,
        intent: str | None = None,
        confidence: float | None = None,
    ) -> LLMResponse:
        if model is None:
            total_chars = sum(len(m.content) for m in messages)
            model = self.select_model(
                intent=intent,
                confidence=confidence,
                estimated_input_tokens=total_chars // 4,
            )
        _log.debug("ModelRouter.complete: selected model=%s", model)
        return await self._provider.complete(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            cache_control=cache_control,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        if model is None:
            total_chars = sum(len(m.content) for m in messages)
            model = self.select_model(estimated_input_tokens=total_chars // 4)
        return self._provider.stream(
            messages, model=model, max_tokens=max_tokens, temperature=temperature
        )
