"""Render eval report JSONs (from run_eval.py) as one Markdown comparison table.

Usage:
    uv run python scripts/eval_table.py docs/evals/<run-a>.json docs/evals/<run-b>.json

Each report becomes a column; rows are the per-metric means plus substring match,
latency percentiles and cost per answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_METRIC_ROWS = [
    ("faithfulness", "Faithfulness (LLM judge)"),
    ("citation_precision", "Citation precision"),
    ("groundedness_heuristic", "Sentences with citations"),
    ("answer_relevance", "Answer relevance"),
    ("context_precision", "Context precision"),
    ("context_recall", "Context recall"),
]


def _column_name(report: dict, path: Path) -> str:
    return report.get("label") or path.stem


def render(paths: list[Path]) -> str:
    reports = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    header = "| Metric | " + " | ".join(_column_name(r, p) for p, r in reports) + " |"
    lines = [header, "|---|" + "---|" * len(reports)]

    for key, label in _METRIC_ROWS:
        cells = [r.get("metrics_summary", {}).get(key, {}).get("mean") for _, r in reports]
        if all(c is None for c in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(_fmt(c) for c in cells) + " |")

    def summary_row(label: str, key: str, fmt) -> None:
        cells = [r.get("summary", {}).get(key) for _, r in reports]
        if any(c is not None for c in cells):
            lines.append(f"| {label} | " + " | ".join(fmt(c) for c in cells) + " |")

    summary_row("Expected-substring match", "substring_match_rate", _fmt)
    summary_row("Latency p50", "latency_p50_ms", _ms)
    summary_row("Latency p95", "latency_p95_ms", _ms)
    summary_row("Cost per answer", "mean_cost_usd", _usd)
    summary_row("Cases", "total_cases", str)
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _ms(value: int | None) -> str:
    return "—" if value is None else f"{value / 1000:.1f} s"


def _usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.stdout.write(render([Path(a) for a in sys.argv[1:]]) + "\n")


if __name__ == "__main__":
    main()
