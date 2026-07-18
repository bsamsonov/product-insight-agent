#!/usr/bin/env python3
"""Compare eval results against a baseline and detect regressions.

Usage:
    python scripts/eval_diff.py --current current.json --baseline baseline.json --threshold 0.05
    python scripts/eval_diff.py --current current.json --baseline baseline.json --report-md out.md

Exit codes: 0 = OK, 1 = regression detected, 2 = baseline not found (warning, not error)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_summary(path: Path) -> dict[str, float]:
    """Extract {metric_name: mean_score} from eval results JSON."""
    data = json.loads(path.read_text())
    summary = data.get("metrics_summary", {})
    return {metric: info["mean"] for metric, info in summary.items() if "mean" in info}


def compare(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold: float,
) -> list[dict]:
    """Return list of comparisons; regressions are metrics where current < baseline - threshold."""
    regressions = []
    for metric, current_score in current.items():
        if metric not in baseline:
            continue
        baseline_score = baseline[metric]
        delta = current_score - baseline_score
        regressed = delta < -threshold
        regressions.append(
            {
                "metric": metric,
                "baseline": baseline_score,
                "current": current_score,
                "delta": delta,
                "regressed": regressed,
            }
        )
    return regressions


def format_markdown(comparisons: list[dict], threshold: float) -> str:
    """Format comparison results as a Markdown table."""
    lines: list[str] = []
    lines.append(f"Regression threshold: **{threshold * 100:.0f}%**\n")
    lines.append("| Metric | Baseline | Current | Delta | Status |")
    lines.append("|--------|----------|---------|-------|--------|")
    for c in comparisons:
        status = "FAIL" if c["regressed"] else "OK"
        delta_sign = "+" if c["delta"] >= 0 else ""
        lines.append(
            f"| {c['metric']} "
            f"| {c['baseline']:.4f} "
            f"| {c['current']:.4f} "
            f"| {delta_sign}{c['delta']:.4f} "
            f"| {status} |"
        )
    if not comparisons:
        lines.append("| — | — | — | — | No shared metrics found |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eval results against a baseline and detect regressions."
    )
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--report-md", type=Path, default=None)
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"WARNING: baseline not found at {args.baseline} — skipping regression check")
        if args.report_md:
            args.report_md.write_text("No baseline found. This is the first eval run.")
        sys.exit(2)

    current = load_summary(args.current)
    baseline = load_summary(args.baseline)
    comparisons = compare(current, baseline, args.threshold)

    regressions = [c for c in comparisons if c["regressed"]]

    md = format_markdown(comparisons, args.threshold)
    print(md)
    if args.report_md:
        args.report_md.write_text(md)

    if regressions:
        print(
            f"\nFAIL: {len(regressions)} metric(s) regressed by more than"
            f" {args.threshold * 100:.0f}%"
        )
        sys.exit(1)
    else:
        print(f"\nPASS: all metrics within {args.threshold * 100:.0f}% of baseline")
        sys.exit(0)


if __name__ == "__main__":
    main()
