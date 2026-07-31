"""Unit tests for judge escalation on low confidence (S3.T2).

Plan AC: a low-confidence verdict from the cheap judge triggers one re-run
through the stronger `escalate` route; budget hard-stops always propagate.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from poc.agent.graph import initial_state
from poc.agent.nodes.judge import make_judge_node
from poc.llm.budget import BudgetExceededError
from poc.llm.provider import LLMResponse


def _judge_response(score: float, cost: float = 0.01) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"score": score, "passed": score >= 0.6, "reasoning": "test"}),
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
        latency_ms=1,
        finish_reason="stop",
    )


def _state_with_draft() -> dict:
    state = initial_state("What do users complain about?")
    state["draft"] = "Users complain about battery life [c1]."
    state["retrieved"] = [{"chunk_id": "c1", "text": "battery drains fast", "score": 0.9}]
    return state


def _llm_returning(response: LLMResponse) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


async def test_low_score_triggers_escalation_and_uses_its_verdict() -> None:
    cheap = _llm_returning(_judge_response(0.3))
    strong = _llm_returning(_judge_response(0.9, cost=0.05))

    node = make_judge_node(cheap, "cheap-model", escalate_llm=strong)
    result = await node(_state_with_draft())

    strong.complete.assert_awaited_once()
    assert result["judgement"]["score"] == 0.9
    assert result["needs_human_review"] is False
    assert result["traces"][-1]["escalated"] is True
    # cost of both calls is accounted
    assert result["cost_usd"] == pytest.approx(0.06)


async def test_high_score_does_not_escalate() -> None:
    cheap = _llm_returning(_judge_response(0.8))
    strong = _llm_returning(_judge_response(0.9))

    node = make_judge_node(cheap, "cheap-model", escalate_llm=strong)
    result = await node(_state_with_draft())

    strong.complete.assert_not_awaited()
    assert result["judgement"]["score"] == 0.8
    assert result["traces"][-1]["escalated"] is False


async def test_no_escalate_llm_keeps_low_verdict_and_flags_review() -> None:
    cheap = _llm_returning(_judge_response(0.3))

    node = make_judge_node(cheap, "cheap-model")  # no escalate_llm
    result = await node(_state_with_draft())

    assert result["judgement"]["score"] == 0.3
    assert result["needs_human_review"] is True


async def test_escalation_failure_keeps_first_verdict() -> None:
    cheap = _llm_returning(_judge_response(0.3))
    strong = MagicMock()
    strong.complete = AsyncMock(side_effect=RuntimeError("escalate provider down"))

    node = make_judge_node(cheap, "cheap-model", escalate_llm=strong)
    result = await node(_state_with_draft())

    assert result["judgement"]["score"] == 0.3  # fail-soft: first verdict kept
    assert result["traces"][-1]["escalated"] is False


async def test_budget_exceeded_in_judge_propagates() -> None:
    cheap = MagicMock()
    cheap.complete = AsyncMock(side_effect=BudgetExceededError("Per-request budget exceeded"))

    node = make_judge_node(cheap, "cheap-model")
    with pytest.raises(BudgetExceededError):
        await node(_state_with_draft())


async def test_budget_exceeded_in_escalation_propagates() -> None:
    cheap = _llm_returning(_judge_response(0.3))
    strong = MagicMock()
    strong.complete = AsyncMock(side_effect=BudgetExceededError("Daily budget exceeded"))

    node = make_judge_node(cheap, "cheap-model", escalate_llm=strong)
    with pytest.raises(BudgetExceededError):
        await node(_state_with_draft())
