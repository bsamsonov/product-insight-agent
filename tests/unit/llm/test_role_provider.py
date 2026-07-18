"""Unit tests for RoleScopedLLM — the LLMProvider adapter over RoutedLLM (S3.T2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from poc.llm.provider import LLMMessage, LLMResponse
from poc.llm.role_provider import RoleScopedLLM
from pydantic import BaseModel


class _Schema(BaseModel):
    answer: str


def _make_response() -> LLMResponse:
    return LLMResponse(
        content='{"answer": "ok"}',
        model="gemini-2.5-flash",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
        latency_ms=42,
        finish_reason="stop",
    )


@pytest.fixture()
def router_mock() -> MagicMock:
    router = MagicMock()
    router.route = AsyncMock(return_value=_make_response())
    return router


async def test_complete_delegates_to_route_with_bound_role(router_mock: MagicMock) -> None:
    llm = RoleScopedLLM(router_mock, "classifier")
    messages = [LLMMessage(role="user", content="hi")]

    result = await llm.complete(
        messages, model="ignored-model", max_tokens=999, temperature=0.7, response_format=_Schema
    )

    assert result.content == '{"answer": "ok"}'
    router_mock.route.assert_awaited_once_with(
        "classifier", messages, response_format=_Schema, temperature=0.7
    )


async def test_model_and_max_tokens_arguments_are_ignored(router_mock: MagicMock) -> None:
    """router.yaml owns model selection — whatever the node passes must not leak through."""
    llm = RoleScopedLLM(router_mock, "judge")
    await llm.complete([LLMMessage(role="user", content="x")], model="gpt-99", max_tokens=1)

    _, kwargs = router_mock.route.await_args
    assert "model" not in kwargs
    assert "max_tokens" not in kwargs


async def test_stream_is_not_supported(router_mock: MagicMock) -> None:
    llm = RoleScopedLLM(router_mock, "summarizer")
    with pytest.raises(NotImplementedError):
        await llm.stream([LLMMessage(role="user", content="x")], model="m", max_tokens=10)


def test_role_property(router_mock: MagicMock) -> None:
    assert RoleScopedLLM(router_mock, "escalate").role == "escalate"
