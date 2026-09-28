from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

from openai import APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletion
from poc.llm.errors import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from poc.llm.pricing import estimate_cost
from poc.llm.provider import LLMMessage, LLMResponse
from poc.llm.rate_limit import limiter_for
from poc.llm.registry import REGISTRY
from poc.observability.tracing import get_tracer, record_llm_call
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

_log = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    def __init__(self, *, provider_name: str, base_url: str, api_key: str) -> None:
        self._provider_name = provider_name
        # Retries are owned by _call_raw (one policy, visible backoff), not the SDK.
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self._retry_attempts = int(os.getenv("POC_LLM_RETRY_ATTEMPTS", "6"))
        self._retry_max_wait_s = float(os.getenv("POC_LLM_RETRY_MAX_WAIT_S", "60"))

    @classmethod
    def from_env(cls, name: str | None = None) -> OpenAICompatibleProvider:
        from poc.llm.settings import LLMSettings

        settings = LLMSettings()
        if name is None:
            name = settings.default_provider
        config = REGISTRY.get(name)

        api_key: str = getattr(settings, f"{name}_api_key", "") or ""
        base_url: str | None = getattr(settings, f"{name}_base_url", None)
        if base_url is None and config is not None:
            base_url = config.base_url
        if base_url is None:
            raise LLMProviderError(f"No base_url configured for provider '{name}'")

        return cls(provider_name=name, base_url=base_url, api_key=api_key)

    async def _call_raw(self, **kwargs: Any) -> ChatCompletion:
        """One chat completion with pacing and retries on 429 / 5xx / timeouts.

        Client errors (400, 401, 404 …) are not retried: repeating them cannot succeed.
        Backoff grows exponentially up to ``POC_LLM_RETRY_MAX_WAIT_S`` (default 60 s), long
        enough for a per-minute quota window to reset.
        """
        limiter = limiter_for(self._provider_name)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(multiplier=2, min=1, max=self._retry_max_wait_s),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                if limiter is not None:
                    await limiter.acquire()
                return await self._client.chat.completions.create(**kwargs)  # type: ignore[return-value]
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_format: type[BaseModel] | None = None,
        cache_control: bool = False,  # reserved for AnthropicProvider
    ) -> LLMResponse:
        openai_messages: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if response_format is not None:
            self._apply_response_format(kwargs, response_format)

        # Wrap the call in an OTel span. When run inside a @traced graph node this span
        # is a child of the node span, so Langfuse shows it as a generation nested in the
        # request trace. record_llm_call() annotates it with gen_ai.* (model, tokens, cost).
        tracer = get_tracer()
        with tracer.start_as_current_span("llm.generate.openai") as span:
            t0 = time.monotonic()
            try:
                resp = await self._call_raw(**kwargs)
            except APITimeoutError as exc:
                raise LLMTimeoutError(f"Timeout calling {self._provider_name}") from exc
            except APIStatusError as exc:
                if exc.status_code == 429:
                    raise LLMRateLimitError(
                        f"Rate limit from {self._provider_name}", status_code=429
                    ) from exc
                raise LLMProviderError(
                    f"API error from {self._provider_name}: {exc.status_code}",
                    status_code=exc.status_code,
                ) from exc
            latency_ms = int((time.monotonic() - t0) * 1000)

            choice = resp.choices[0]
            content = choice.message.content or ""

            if response_format is not None and self._provider_name != "gemini":
                content = await self._validate_or_retry_json(content, response_format, kwargs)

            usage = resp.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            cached_tok = self._extract_cached_tokens(usage)
            cost_usd = estimate_cost(self._provider_name, model, in_tok, out_tok)

            record_llm_call(
                span,
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost_usd,
                provider=self._provider_name,
                input_messages=openai_messages,
                output_text=content,
            )

            return LLMResponse(
                content=content,
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                finish_reason=choice.finish_reason or "stop",
                cached_input_tokens=cached_tok,
            )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]
        async_stream = await self._client.chat.completions.create(
            model=model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async def _gen() -> AsyncGenerator[str, None]:
            async for chunk in async_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return _gen()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _extract_cached_tokens(usage: Any) -> int:
        """Extract provider-side prompt-cache hit tokens from a usage object (S3.T3).

        Providers report cache hits in different fields:
        - OpenAI-compatible (incl. Gemini implicit cache via the OpenAI endpoint):
          ``usage.prompt_tokens_details.cached_tokens``
        - DeepSeek automatic context cache: ``usage.prompt_cache_hit_tokens``

        Uses getattr so absent fields (older providers, Ollama) safely yield 0.
        """
        if usage is None:
            return 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is None:
            cached = getattr(usage, "prompt_cache_hit_tokens", None)
        return int(cached or 0)

    def _apply_response_format(
        self, kwargs: dict[str, Any], response_format: type[BaseModel]
    ) -> None:
        if self._provider_name == "gemini":
            # Gemini's OpenAI-compatible endpoint takes the standard structured-output
            # shape; the native `response_mime_type`/`response_schema` fields are rejected
            # there with HTTP 400.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
            schema_str = json.dumps(response_format.model_json_schema())
            msgs: list[dict[str, str]] = [dict(m) for m in kwargs["messages"]]
            if any(m["role"] == "system" for m in msgs):
                for m in msgs:
                    if m["role"] == "system":
                        m["content"] += f"\n\nRespond with JSON matching this schema:\n{schema_str}"
                        break
            else:
                system_msg = {
                    "role": "system",
                    "content": f"Respond with JSON matching this schema:\n{schema_str}",
                }
                msgs = [system_msg, *msgs]
            kwargs["messages"] = msgs

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        """Strip markdown code fences (```json ... ```) some models wrap JSON in."""
        text = content.strip()
        if text.startswith("```"):
            text = text[3:]
            if text[:4].lower() == "json":
                text = text[4:]
            if text.endswith("```"):
                text = text[:-3]
        return text.strip()

    async def _validate_or_retry_json(
        self,
        content: str,
        response_format: type[BaseModel],
        original_kwargs: dict[str, Any],
    ) -> str:
        content = self._strip_json_fence(content)
        try:
            response_format.model_validate_json(content)
            return content
        except Exception as exc:
            _log.warning(
                "Invalid JSON from %s, retrying with error feedback. Error: %s",
                self._provider_name,
                exc,
            )
            error_feedback = f"JSON validation failed: {exc}. Return valid JSON only."
            retry_messages = [
                *original_kwargs["messages"],
                {"role": "assistant", "content": content},
                {"role": "user", "content": error_feedback},
            ]
            kwargs2 = {**original_kwargs, "messages": retry_messages}
            try:
                resp2 = await self._call_raw(**kwargs2)
            except (APITimeoutError, APIStatusError) as exc2:
                raise LLMProviderError(
                    f"API error on JSON retry from {self._provider_name}"
                ) from exc2
            return self._strip_json_fence(resp2.choices[0].message.content or "")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APITimeoutError):
        return True
    if not isinstance(exc, APIStatusError):
        return False
    if exc.status_code == 429:
        # A per-day quota does not reset within any sensible backoff window.
        return "PerDay" not in str(exc)
    return exc.status_code >= 500
