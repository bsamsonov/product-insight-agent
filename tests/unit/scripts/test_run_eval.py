"""Unit tests for the retrieval-only eval mode in scripts/run_eval.py.

scripts/ is not a package, so it is added to sys.path directly (matches the
pattern already used by tests/unit/scripts/test_eval_diff.py).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import typer

_SCRIPTS_DIR = Path(__file__).parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from poc.evals.runner import EvalCase  # noqa: E402
from run_eval import (  # noqa: E402
    _record_llm_metric,
    _run_retrieval_only,
    compute_hit,
    latency_cost_summary,
)


@dataclass
class _FakeChunk:
    id: str


@dataclass
class _FakeScoredChunk:
    chunk: _FakeChunk
    score: float = 1.0


class _FakeRetriever:
    """Returns a preconfigured retrieval result per question, ignoring ranking details."""

    def __init__(self, hits_by_question: dict[str, list[str]]) -> None:
        self._hits_by_question = hits_by_question

    def retrieve(self, query: str, *, top_k: int = 10, filters=None):
        ids = self._hits_by_question.get(query, [])
        return [_FakeScoredChunk(chunk=_FakeChunk(id=cid)) for cid in ids]


def _case(case_id: str, question: str, expected_chunk_ids: list[str]) -> EvalCase:
    return EvalCase(
        id=case_id,
        question=question,
        expected_answer_substrings=[],
        expected_chunk_ids=expected_chunk_ids,
    )


# ── compute_hit ────────────────────────────────────────────────────────────────


def test_compute_hit_true_when_expected_chunk_retrieved():
    assert compute_hit(["a", "b", "c"], ["c", "z"]) is True


def test_compute_hit_false_when_no_overlap():
    assert compute_hit(["a", "b"], ["x", "y"]) is False


def test_compute_hit_false_when_expected_empty():
    assert compute_hit(["a", "b"], []) is False


def test_compute_hit_false_when_nothing_retrieved():
    assert compute_hit([], ["a"]) is False


# ── _run_retrieval_only: hit-rate computation + exit-code gate ────────────────


async def test_run_retrieval_only_passes_when_hit_rate_meets_threshold(tmp_path: Path) -> None:
    cases = [
        _case("c1", "q1", ["chunk_1"]),
        _case("c2", "q2", ["chunk_2"]),
    ]
    retriever = _FakeRetriever({"q1": ["chunk_1"], "q2": ["chunk_2"]})
    output = tmp_path / "report.json"

    # hit_rate is 1.0 >= 0.8 -> must not raise
    await _run_retrieval_only(
        cases=cases,
        retriever=retriever,
        eval_set=Path("golden_set_sample.jsonl"),
        output=output,
        min_hit_rate=0.8,
    )

    report = json.loads(output.read_text())
    assert report["mode"] == "retrieval_only"
    assert report["summary"]["hit_rate"] == 1.0
    assert report["summary"]["total_cases"] == 2
    assert all(c["hit"] for c in report["per_case"])


async def test_run_retrieval_only_exits_nonzero_below_threshold(tmp_path: Path) -> None:
    cases = [
        _case("c1", "q1", ["chunk_1"]),
        _case("c2", "q2", ["chunk_2"]),
        _case("c3", "q3", ["chunk_3"]),
        _case("c4", "q4", ["chunk_4"]),
    ]
    # Only 1/4 cases retrieve their expected chunk -> hit_rate 0.25, below 0.8.
    retriever = _FakeRetriever({"q1": ["chunk_1"], "q2": [], "q3": [], "q4": []})
    output = tmp_path / "report.json"

    with pytest.raises(typer.Exit) as exc_info:
        await _run_retrieval_only(
            cases=cases,
            retriever=retriever,
            eval_set=Path("golden_set_sample.jsonl"),
            output=output,
            min_hit_rate=0.8,
        )

    assert exc_info.value.exit_code == 1
    # Report is still written even on failure, for CI artifact inspection.
    report = json.loads(output.read_text())
    assert report["summary"]["hit_rate"] == 0.25


async def test_run_retrieval_only_exits_zero_when_no_cases_below_threshold_is_moot(
    tmp_path: Path,
) -> None:
    """An empty case list has hit_rate 0.0, which fails any positive threshold."""
    output = tmp_path / "report.json"
    with pytest.raises(typer.Exit) as exc_info:
        await _run_retrieval_only(
            cases=[],
            retriever=_FakeRetriever({}),
            eval_set=Path("golden_set_sample.jsonl"),
            output=output,
            min_hit_rate=0.8,
        )
    assert exc_info.value.exit_code == 1


def _result(latency_ms: int, cost: float | None, error: str | None = None):
    from poc.evals.runner import EvalResult

    return EvalResult(
        case_id="c",
        question="q",
        answer_text="a",
        cited_chunk_ids=[],
        faithfulness_score=0.0,
        citation_precision_score=0.0,
        substring_match=True,
        latency_ms=latency_ms,
        cost_usd=cost,
        error=error,
    )


def test_latency_cost_summary_percentiles_and_cost():
    results = [_result(ms, 0.001) for ms in range(100, 2100, 100)]  # 20 cases
    results.append(_result(99_999, 5.0, error="boom"))  # errors are excluded
    summary = latency_cost_summary(results)
    assert summary["latency_p50_ms"] == 1000
    assert summary["latency_p95_ms"] == 1900
    assert summary["mean_cost_usd"] == 0.001
    assert summary["total_cost_usd"] == 0.02


def test_latency_cost_summary_empty():
    assert latency_cost_summary([]) == {
        "latency_p50_ms": 0,
        "latency_p95_ms": 0,
        "mean_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }


def test_failed_llm_metric_is_counted_not_scored():
    from poc.evals.metrics import MetricResult

    scores: dict[str, float] = {}
    failures: dict[str, int] = {}
    ok = MetricResult(name="faithfulness", score=0.9, reasoning="fine")
    bad = MetricResult(name="faithfulness", score=0.0, reasoning="eval error: Rate limit")
    _record_llm_metric(scores, failures, "faithfulness", ok)
    _record_llm_metric(scores, failures, "answer_relevance", bad)
    assert scores == {"faithfulness": 0.9}
    assert failures == {"answer_relevance": 1}
