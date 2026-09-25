from __future__ import annotations

import json

import httpx
import pytest
import respx
from poc.llm.errors import LLMProviderError, LLMRateLimitError
from poc.llm.openai_compatible import OpenAICompatibleProvider
from poc.llm.pricing import estimate_cost
from poc.llm.provider import LLMMessage
from pydantic import BaseModel

# ------------------------------------------------------------------ fixtures


@pytest.fixture
def groq_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
    )


@pytest.fixture
def gemini_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="test-key",
    )


def _chat_response(content: str, model: str, in_tok: int = 10, out_tok: int = 5) -> dict:
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


# ------------------------------------------------------------------ complete()


@respx.mock
async def test_groq_complete_success(groq_provider: OpenAICompatibleProvider) -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_chat_response("Hello!", "llama-3.3-70b-versatile", 10, 5)
        )
    )

    result = await groq_provider.complete(
        [LLMMessage(role="user", content="hi")],
        model="llama-3.3-70b-versatile",
        max_tokens=20,
        temperature=0.0,
    )

    assert result.content == "Hello!"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.finish_reason == "stop"
    assert result.model == "llama-3.3-70b-versatile"
    assert result.latency_ms >= 0


@respx.mock
async def test_gemini_complete_success(gemini_provider: OpenAICompatibleProvider) -> None:
    respx.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("Pong", "gemini-2.5-flash", 8, 3))
    )

    result = await gemini_provider.complete(
        [LLMMessage(role="user", content="ping")],
        model="gemini-2.5-flash",
        max_tokens=10,
        temperature=0.0,
    )

    assert result.content == "Pong"
    assert result.input_tokens == 8
    assert result.output_tokens == 3


@respx.mock
async def test_complete_cost_calculated(groq_provider: OpenAICompatibleProvider) -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_chat_response("ok", "llama-3.3-70b-versatile", 1000, 500)
        )
    )

    result = await groq_provider.complete(
        [LLMMessage(role="user", content="test")],
        model="llama-3.3-70b-versatile",
        max_tokens=100,
        temperature=0.0,
    )

    # 1000 in @ $0.59/1M + 500 out @ $0.79/1M = $0.000590 + $0.000395 = $0.000985
    assert result.cost_usd is not None
    assert abs(result.cost_usd - 0.000985) < 1e-8


# ------------------------------------------------------------------ rate limit / errors


@respx.mock
async def test_rate_limit_raises_llm_rate_limit_error(
    groq_provider: OpenAICompatibleProvider,
) -> None:
    # Return 429 on every attempt so tenacity gives up and re-raises as domain error.
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limit"}})
    )

    with pytest.raises(LLMRateLimitError):
        await groq_provider.complete(
            [LLMMessage(role="user", content="hi")],
            model="llama-3.3-70b-versatile",
            max_tokens=10,
            temperature=0.0,
        )


@respx.mock
async def test_server_error_raises_llm_provider_error(
    groq_provider: OpenAICompatibleProvider,
) -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "internal server error"}})
    )

    with pytest.raises(LLMProviderError):
        await groq_provider.complete(
            [LLMMessage(role="user", content="hi")],
            model="llama-3.3-70b-versatile",
            max_tokens=10,
            temperature=0.0,
        )


# ------------------------------------------------------------------ structured output


class _Answer(BaseModel):
    answer: str
    confidence: float


@respx.mock
async def test_groq_structured_output_valid(groq_provider: OpenAICompatibleProvider) -> None:
    payload = json.dumps({"answer": "Paris", "confidence": 0.95})
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(payload, "llama-3.3-70b-versatile"))
    )

    result = await groq_provider.complete(
        [LLMMessage(role="user", content="Capital of France?")],
        model="llama-3.3-70b-versatile",
        max_tokens=50,
        temperature=0.0,
        response_format=_Answer,
    )

    assert result.content == payload
    parsed = _Answer.model_validate_json(result.content)
    assert parsed.answer == "Paris"


@respx.mock
async def test_groq_structured_output_retry_on_invalid_json(
    groq_provider: OpenAICompatibleProvider,
) -> None:
    """First response is invalid JSON → retry → second response is valid."""
    valid_payload = json.dumps({"answer": "Berlin", "confidence": 0.9})
    route = respx.post("https://api.groq.com/openai/v1/chat/completions")
    route.side_effect = [
        httpx.Response(200, json=_chat_response("not-json-at-all", "llama-3.3-70b-versatile")),
        httpx.Response(200, json=_chat_response(valid_payload, "llama-3.3-70b-versatile")),
    ]

    result = await groq_provider.complete(
        [LLMMessage(role="user", content="Capital of Germany?")],
        model="llama-3.3-70b-versatile",
        max_tokens=50,
        temperature=0.0,
        response_format=_Answer,
    )

    assert result.content == valid_payload


# ------------------------------------------------------------------ gemini structured output


@respx.mock
async def test_gemini_structured_output_uses_json_schema(
    gemini_provider: OpenAICompatibleProvider,
) -> None:
    captured: list[dict] = []

    async def capture(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        payload = json.dumps({"answer": "Tokyo", "confidence": 0.99})
        return httpx.Response(200, json=_chat_response(payload, "gemini-2.5-flash"))

    respx.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions").mock(
        side_effect=capture
    )

    await gemini_provider.complete(
        [LLMMessage(role="user", content="Capital of Japan?")],
        model="gemini-2.5-flash",
        max_tokens=50,
        temperature=0.0,
        response_format=_Answer,
    )

    fmt = captured[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "_Answer"
    assert fmt["json_schema"]["schema"] == _Answer.model_json_schema()
    # Native Gemini fields are rejected by the OpenAI-compatible endpoint (HTTP 400).
    assert "response_mime_type" not in captured[0]


# ------------------------------------------------------------------ pricing unit tests


def test_estimate_cost_known_model() -> None:
    cost = estimate_cost("groq", "llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.59 + 0.79, rel=1e-6)


def test_estimate_cost_ollama_is_free() -> None:
    assert estimate_cost("ollama", "anything", 999_999, 999_999) == 0.0


def test_estimate_cost_unknown_model_returns_none() -> None:
    assert estimate_cost("groq", "nonexistent-model-xyz", 100, 100) is None


def test_estimate_cost_unknown_provider_returns_none() -> None:
    assert estimate_cost("unknown-provider", "model", 100, 100) is None
