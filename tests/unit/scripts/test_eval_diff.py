"""Unit tests for scripts/eval_diff.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# eval_diff.py lives in scripts/, which is not a package — add it to path.
_SCRIPTS_DIR = Path(__file__).parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_diff import compare, format_markdown, load_summary  # noqa: E402

_EVAL_DIFF = _SCRIPTS_DIR / "eval_diff.py"


def _write_summary(path: Path, metrics: dict[str, float]) -> None:
    path.write_text(json.dumps({"metrics_summary": {k: {"mean": v} for k, v in metrics.items()}}))


def _run_diff(current: Path, baseline: Path, threshold: float = 0.05) -> int:
    """Run eval_diff.py as a subprocess and return its exit code (the CI gate)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_EVAL_DIFF),
            "--current",
            str(current),
            "--baseline",
            str(baseline),
            "--threshold",
            str(threshold),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode


def test_no_regression() -> None:
    current = {"citation_precision": 0.75, "groundedness_heuristic": 0.80}
    baseline = {"citation_precision": 0.70, "groundedness_heuristic": 0.80}
    comparisons = compare(current, baseline, threshold=0.05)
    assert all(not c["regressed"] for c in comparisons)


def test_regression_detected() -> None:
    current = {"citation_precision": 0.60}  # dropped 0.15, exceeds 0.05 threshold
    baseline = {"citation_precision": 0.75}
    comparisons = compare(current, baseline, threshold=0.05)
    assert len(comparisons) == 1
    assert comparisons[0]["regressed"]


def test_within_threshold() -> None:
    current = {"citation_precision": 0.71}  # dropped 0.04 < threshold 0.05
    baseline = {"citation_precision": 0.75}
    comparisons = compare(current, baseline, threshold=0.05)
    assert len(comparisons) == 1
    assert not comparisons[0]["regressed"]


def test_exactly_at_threshold_boundary() -> None:
    # drop of exactly threshold → not regressed (uses integers to avoid float imprecision)
    current = {"citation_precision": 0.650}
    baseline = {"citation_precision": 0.700}
    comparisons = compare(current, baseline, threshold=0.05)
    # delta = -0.05 exactly (both values representable in float) — not strictly less than -0.05
    assert not comparisons[0]["regressed"]


def test_metric_not_in_baseline_is_skipped() -> None:
    current = {"citation_precision": 0.50, "new_metric": 0.90}
    baseline = {"citation_precision": 0.70}
    comparisons = compare(current, baseline, threshold=0.05)
    # new_metric has no baseline → not included
    assert len(comparisons) == 1
    assert comparisons[0]["metric"] == "citation_precision"


def test_load_summary(tmp_path: Path) -> None:
    result_file = tmp_path / "results.json"
    result_file.write_text(
        json.dumps(
            {"metrics_summary": {"citation_precision": {"mean": 0.75, "min": 0.5, "max": 1.0}}}
        )
    )
    summary = load_summary(result_file)
    assert summary["citation_precision"] == 0.75


def test_load_summary_skips_entries_without_mean(tmp_path: Path) -> None:
    result_file = tmp_path / "results.json"
    result_file.write_text(
        json.dumps(
            {
                "metrics_summary": {
                    "good": {"mean": 0.8, "min": 0.0, "max": 1.0},
                    "bad": {"min": 0.0, "max": 1.0},  # no mean
                }
            }
        )
    )
    summary = load_summary(result_file)
    assert "good" in summary
    assert "bad" not in summary


def test_format_markdown_contains_fail_for_regression() -> None:
    comparisons = [
        {
            "metric": "citation_precision",
            "baseline": 0.75,
            "current": 0.60,
            "delta": -0.15,
            "regressed": True,
        },
    ]
    md = format_markdown(comparisons, threshold=0.05)
    assert "FAIL" in md
    assert "citation_precision" in md


def test_format_markdown_ok_for_passing() -> None:
    comparisons = [
        {
            "metric": "groundedness_heuristic",
            "baseline": 0.50,
            "current": 0.55,
            "delta": 0.05,
            "regressed": False,
        },
    ]
    md = format_markdown(comparisons, threshold=0.05)
    assert "OK" in md
    assert "FAIL" not in md


def test_format_markdown_empty_comparisons() -> None:
    md = format_markdown([], threshold=0.05)
    assert "No shared metrics" in md


# ---------------------------------------------------------------------------
# CI gate — exit codes (AC S6.T2: a fake regression turns CI red)
# ---------------------------------------------------------------------------


def test_cli_exit_1_on_regression(tmp_path: Path) -> None:
    """AC: a deliberate metric drop (e.g. garbage prompt) → non-zero exit → CI red."""
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    _write_summary(current, {"faithfulness": 0.55})  # dropped 0.20
    _write_summary(baseline, {"faithfulness": 0.75})
    assert _run_diff(current, baseline) == 1


def test_cli_exit_0_when_within_threshold(tmp_path: Path) -> None:
    """No regression → exit 0 → CI green."""
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    _write_summary(current, {"faithfulness": 0.74})  # dropped 0.01 < 0.05
    _write_summary(baseline, {"faithfulness": 0.75})
    assert _run_diff(current, baseline) == 0


def test_cli_exit_2_when_baseline_missing(tmp_path: Path) -> None:
    """Missing baseline is a warning (exit 2), not a hard CI failure."""
    current = tmp_path / "current.json"
    _write_summary(current, {"faithfulness": 0.75})
    assert _run_diff(current, tmp_path / "nope.json") == 2
