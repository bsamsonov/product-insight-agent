from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from poc.guardrails.groundedness import GroundednessChecker, GroundednessResult
from poc.llm.provider import LLMResponse


def _make_llm(content: str) -> AsyncMock:
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = LLMResponse(
        content=content,
        model="gemini-2.5-flash",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0,
        latency_ms=500,
        finish_reason="stop",
    )
    return mock_llm


async def test_fully_grounded():
    """All claims supported → score=1.0."""
    payload = json.dumps(
        [
            {
                "claim": "Product has good grip",
                "verdict": "supported",
                "reasoning": "chunk mentions grip",
            }
        ]
    )
    checker = GroundednessChecker(_make_llm(payload), model="gemini-2.5-flash")
    result = await checker.check(
        "Product has good grip.",
        ["The shoe provides excellent grip on wet surfaces."],
    )
    assert result.score == 1.0
    assert result.supported_count == 1
    assert result.total_count == 1
    assert len(result.claims) == 1
    assert result.claims[0].verdict == "supported"


async def test_ungrounded_answer():
    """All claims unsupported → score=0.0."""
    payload = json.dumps(
        [
            {
                "claim": "The product is made of titanium",
                "verdict": "unsupported",
                "reasoning": "context says nothing about material",
            }
        ]
    )
    checker = GroundednessChecker(_make_llm(payload), model="gemini-2.5-flash")
    result = await checker.check(
        "The product is made of titanium.",
        ["The shoe has a rubber sole."],
    )
    assert result.score == 0.0
    assert result.supported_count == 0
    assert result.total_count == 1


async def test_mixed_verdicts():
    """partial counts as 0.5; supported=1, partial=1, total=2 → score=0.75."""
    payload = json.dumps(
        [
            {"claim": "Good grip", "verdict": "supported", "reasoning": "explicitly stated"},
            {
                "claim": "Good for hiking",
                "verdict": "partial",
                "reasoning": "implied but not direct",
            },
        ]
    )
    checker = GroundednessChecker(_make_llm(payload), model="gemini-2.5-flash")
    result = await checker.check(
        "Good grip. Good for hiking.",
        ["The shoe provides excellent grip on wet surfaces."],
    )
    assert result.score == pytest.approx(0.75)
    assert result.supported_count == 1
    assert result.total_count == 2


async def test_invalid_json_returns_empty_result():
    """LLM returns non-JSON → GroundednessResult with score=0.0, no exception."""
    checker = GroundednessChecker(_make_llm("I cannot parse this."), model="gemini-2.5-flash")
    result = await checker.check("Some answer.", ["Some context."])
    assert isinstance(result, GroundednessResult)
    assert result.score == 0.0
    assert result.claims == []
    assert result.total_count == 0


async def test_markdown_fenced_json_is_parsed():
    """LLM wraps JSON in markdown code fence → still parsed correctly."""
    inner = json.dumps(
        [{"claim": "Fast delivery", "verdict": "supported", "reasoning": "ships in 1 day"}]
    )
    fenced = f"```json\n{inner}\n```"
    checker = GroundednessChecker(_make_llm(fenced), model="gemini-2.5-flash")
    result = await checker.check("Fast delivery.", ["Ships within 1 business day."])
    assert result.score == 1.0
    assert result.total_count == 1


async def test_max_chunks_respected():
    """Only first max_chunks chunks are sent to LLM (verified via call args)."""
    payload = json.dumps([{"claim": "claim", "verdict": "supported", "reasoning": "ok"}])
    mock_llm = _make_llm(payload)
    checker = GroundednessChecker(mock_llm, model="gemini-2.5-flash")

    chunks = [f"chunk {i}" for i in range(20)]
    await checker.check("Some answer.", chunks, max_chunks=3)

    call_args = mock_llm.complete.call_args
    messages = call_args[0][0]  # positional arg: list[LLMMessage]
    user_content = messages[0].content
    # Only first 3 chunks should appear
    assert "chunk 3" not in user_content
    assert "chunk 0" in user_content


async def test_all_partial_score():
    """All partial → score=0.5."""
    payload = json.dumps(
        [
            {"claim": "claim A", "verdict": "partial", "reasoning": "somewhat"},
            {"claim": "claim B", "verdict": "partial", "reasoning": "somewhat"},
        ]
    )
    checker = GroundednessChecker(_make_llm(payload), model="gemini-2.5-flash")
    result = await checker.check("A. B.", ["Some context."])
    assert result.score == pytest.approx(0.5)
    assert result.supported_count == 0


async def test_empty_claims_list_from_llm():
    """LLM returns [] → score=0.0, total_count=0."""
    checker = GroundednessChecker(_make_llm("[]"), model="gemini-2.5-flash")
    result = await checker.check("Short.", ["Context."])
    assert result.score == 0.0
    assert result.total_count == 0
