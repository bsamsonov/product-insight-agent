"""Unit tests for RAGAS-style metrics."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from poc.evals.metrics import (
    MetricResult,
    answer_relevance,
    citation_precision,
    context_precision,
    context_recall,
    faithfulness,
    groundedness_heuristic,
)
from poc.llm.provider import LLMResponse

# ── Non-LLM metric tests ──────────────────────────────────────────────────────


def test_citation_precision_exact_match():
    result = citation_precision(["doc_001_chunk_002"], ["doc_001_chunk_002"])
    assert result.score == 1.0
    assert result.name == "citation_precision"


def test_citation_precision_partial():
    result = citation_precision(["doc_001_chunk_001", "doc_999_chunk_999"], ["doc_001_chunk_001"])
    assert result.score == 0.5


def test_citation_precision_empty_expected():
    result = citation_precision(["doc_001_chunk_001"], [])
    assert result.score == 1.0  # no expectation = pass


def test_citation_precision_empty_citations():
    result = citation_precision([], ["doc_001_chunk_001"])
    assert result.score == 0.0


def test_citation_precision_no_overlap():
    result = citation_precision(["doc_999"], ["doc_001"])
    assert result.score == 0.0


def test_citation_precision_multiple_hits():
    result = citation_precision(["a", "b", "c"], ["a", "b", "c"])
    assert result.score == 1.0


def test_groundedness_with_citations():
    answer = "The product has good cushioning [chunk_001]. Users love the comfort [chunk_002]."
    result = groundedness_heuristic(answer, [])
    assert result.score == 1.0
    assert result.name == "groundedness_heuristic"


def test_groundedness_no_citations():
    answer = "The product has good cushioning. Users love the comfort overall."
    result = groundedness_heuristic(answer, [])
    assert result.score == 0.0


def test_groundedness_partial_citations():
    # Two sentences, only one has a citation
    answer = "The product has excellent grip [ref_1]. Customers are generally satisfied."
    result = groundedness_heuristic(answer, [])
    assert result.score == pytest.approx(0.5)


def test_groundedness_empty_answer():
    result = groundedness_heuristic("", [])
    assert result.score == 1.0  # no substantive sentences


def test_groundedness_short_sentences_ignored():
    # Sentences shorter than 20 chars are skipped
    answer = "Yes [ref]. No [ref]."
    result = groundedness_heuristic(answer, [])
    # Both sentences < 20 chars → no substantive sentences → score=1.0
    assert result.score == 1.0


# ── LLM-based metric tests (with mocks) ──────────────────────────────────────


def _make_llm_mock(judge_output: dict) -> MagicMock:
    """Create a mock LLMProvider returning judge_output as JSON content."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(judge_output),
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0,
            latency_ms=100,
            finish_reason="stop",
        )
    )
    return mock_llm


async def test_faithfulness_with_mock_llm():
    judge_output = {
        "claims": ["product is comfortable"],
        "supported_claims": ["product is comfortable"],
        "unsupported_claims": [],
        "score": 1.0,
    }
    mock_llm = _make_llm_mock(judge_output)

    result = await faithfulness(
        "The product is comfortable.",
        ["Product reviews show great comfort ratings."],
        mock_llm,
        model="test-model",
    )
    assert result.score >= 0.0
    assert result.name == "faithfulness"
    assert mock_llm.complete.called


async def test_faithfulness_score_computed_from_claims():
    """Score must be supported_count / total_claims, not hardcoded."""
    judge_output = {
        "claims": ["claim A", "claim B", "claim C"],
        "supported_claims": ["claim A"],
        "unsupported_claims": ["claim B", "claim C"],
        "score": 0.5,  # LLM might give wrong score; we recompute
    }
    mock_llm = _make_llm_mock(judge_output)

    result = await faithfulness(
        "Claim A. Claim B. Claim C.",
        ["Only claim A is in context."],
        mock_llm,
        model="test-model",
    )
    # Our code recomputes: 1 supported / 3 total = 0.333
    assert result.score == pytest.approx(1 / 3, abs=0.01)


async def test_faithfulness_empty_claims():
    """When no claims extracted, score should be 1.0 (vacuous truth)."""
    judge_output = {
        "claims": [],
        "supported_claims": [],
        "unsupported_claims": [],
        "score": 1.0,
    }
    mock_llm = _make_llm_mock(judge_output)

    result = await faithfulness(
        "",
        [],
        mock_llm,
        model="test-model",
    )
    assert result.score == 1.0


async def test_faithfulness_llm_failure_returns_zero():
    """If LLM call raises, metric returns 0.0 gracefully."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("network error"))

    result = await faithfulness(
        "Some answer.",
        ["Some context."],
        mock_llm,
        model="test-model",
    )
    assert result.score == 0.0
    assert result.name == "faithfulness"


async def test_answer_relevance_relevant():
    judge_output = {"is_relevant": True, "reasoning": "answer addresses question", "score": 1.0}
    mock_llm = _make_llm_mock(judge_output)

    result = await answer_relevance(
        "What are shoe complaints?",
        "Customers complain about sole durability and fit.",
        mock_llm,
        model="test-model",
    )
    assert result.score == 1.0
    assert result.name == "answer_relevance"


async def test_answer_relevance_not_relevant():
    judge_output = {"is_relevant": False, "reasoning": "off topic", "score": 0.0}
    mock_llm = _make_llm_mock(judge_output)

    result = await answer_relevance(
        "What are shoe complaints?",
        "Paris is the capital of France.",
        mock_llm,
        model="test-model",
    )
    assert result.score == 0.0


async def test_context_precision_with_mock():
    judge_output = {"relevant_contexts": [0, 1], "irrelevant_contexts": [2], "score": 0.667}
    mock_llm = _make_llm_mock(judge_output)

    result = await context_precision(
        "What is shoe cushioning?",
        ["Cushioning info...", "More cushion...", "Irrelevant text..."],
        mock_llm,
        model="test-model",
    )
    assert result.name == "context_precision"
    # Our code recomputes: 2 relevant / 3 total
    assert result.score == pytest.approx(2 / 3, abs=0.01)


async def test_context_precision_empty_contexts():
    mock_llm = _make_llm_mock({})
    result = await context_precision("question", [], mock_llm, model="test-model")
    assert result.score == 1.0
    assert not mock_llm.complete.called


async def test_context_recall_with_mock():
    judge_output = {"is_covered": True, "reasoning": "contexts cover expected answer", "score": 1.0}
    mock_llm = _make_llm_mock(judge_output)

    result = await context_recall(
        "Shoe cushioning is excellent.",
        "cushioning excellent",
        ["Product cushioning review: excellent!"],
        mock_llm,
        model="test-model",
    )
    assert result.score == 1.0
    assert result.name == "context_recall"


# ── MetricResult dataclass tests ──────────────────────────────────────────────


def test_metric_result_defaults():
    r = MetricResult(name="test", score=0.75)
    assert r.name == "test"
    assert r.score == 0.75
    assert r.reasoning == ""
    assert r.details == {}


def test_metric_result_with_details():
    r = MetricResult(name="faithfulness", score=0.5, details={"claims": ["a", "b"]})
    assert r.details["claims"] == ["a", "b"]
