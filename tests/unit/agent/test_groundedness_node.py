from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from poc.agent.nodes.groundedness import GROUNDEDNESS_THRESHOLD, make_groundedness_node
from poc.agent.state import AgentState
from poc.core.models import Answer
from poc.llm.provider import LLMResponse


def _make_state(**kwargs) -> AgentState:
    base = AgentState(
        question="What do customers say?",
        tenant="default",
        filters={},
        intent="",
        intent_confidence=0.0,
        plan={},
        retrieved=[{"chunk_id": "c1", "text": "shoes are comfortable"}],
        clusters=[],
        draft="Customers like comfort [c1].",
        judgement={"score": 0.85, "passed": True, "reasoning": "ok"},
        final=None,
        cost_usd=0.0,
        traces=[],
        needs_human_review=False,
        human_feedback=None,
    )
    base.update(kwargs)
    return base


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            latency_ms=10,
            finish_reason="stop",
        )
    )
    return llm


_ALL_SUPPORTED = (
    '[{"claim": "comfort", "verdict": "supported", "reasoning": "r"}, '
    '{"claim": "fit", "verdict": "supported", "reasoning": "r"}]'
)
_ALL_UNSUPPORTED = (
    '[{"claim": "x", "verdict": "unsupported", "reasoning": "r"}, '
    '{"claim": "y", "verdict": "unsupported", "reasoning": "r"}]'
)


class TestGroundednessNode:
    async def test_high_score_does_not_escalate(self) -> None:
        node = make_groundedness_node(_mock_llm(_ALL_SUPPORTED), "test-model")
        result = await node(_make_state())

        assert result["judgement"]["groundedness"] == 1.0
        # preserves the judge's prior verdict
        assert result["judgement"]["passed"] is True
        assert "needs_human_review" not in result
        assert any(t["node"] == "groundedness" for t in result["traces"])

    async def test_low_score_escalates_for_review(self) -> None:
        node = make_groundedness_node(_mock_llm(_ALL_UNSUPPORTED), "test-model")
        result = await node(_make_state())

        assert result["judgement"]["groundedness"] == 0.0
        assert result["judgement"]["groundedness"] < GROUNDEDNESS_THRESHOLD
        assert result["needs_human_review"] is True

    async def test_prefers_final_answer_text(self) -> None:
        llm = _mock_llm(_ALL_SUPPORTED)
        node = make_groundedness_node(llm, "test-model")
        final = Answer(
            question="q",
            text="Final answer text [c1].",
            citations=[],
            used_chunks=["c1"],
            cost_usd=0.0,
            latency_ms=0,
        )
        await node(_make_state(final=final))

        # GroundednessChecker.check passes the prompt as the first positional arg
        # (a list of LLMMessage); assert it carries the finalized Answer.text.
        sent_prompt = llm.complete.call_args.args[0][0].content
        assert "Final answer text" in sent_prompt

    async def test_no_draft_passes_through_without_check(self) -> None:
        llm = _mock_llm(_ALL_SUPPORTED)
        node = make_groundedness_node(llm, "test-model")
        result = await node(_make_state(draft="UNKNOWN: no data"))

        assert result["judgement"]["groundedness"] is None
        llm.complete.assert_not_called()
        assert "needs_human_review" not in result

    async def test_llm_failure_is_fail_soft(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        node = make_groundedness_node(llm, "test-model")
        result = await node(_make_state())

        assert result["judgement"]["groundedness"] is None
        assert "needs_human_review" not in result

    async def test_does_not_downgrade_existing_review_flag(self) -> None:
        node = make_groundedness_node(_mock_llm(_ALL_SUPPORTED), "test-model")
        result = await node(_make_state(needs_human_review=True))

        # high groundedness → node doesn't touch the flag; prior True survives in state
        assert "needs_human_review" not in result


async def test_unparseable_checker_output_is_unknown_not_escalated() -> None:
    llm = _mock_llm("not json at all")
    node = make_groundedness_node(llm, "test-model")
    result = await node(_make_state())

    assert result["judgement"]["groundedness"] is None
    assert "needs_human_review" not in result
    assert result["traces"][-1]["unavailable"] is True
