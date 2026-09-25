import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from eval_table import render


def test_render_two_reports(tmp_path):
    a = {
        "label": "BM25",
        "metrics_summary": {"faithfulness": {"mean": 0.7}},
        "summary": {"total_cases": 35, "latency_p95_ms": 12000, "mean_cost_usd": 0.0012},
    }
    b = {
        "label": "Hybrid",
        "metrics_summary": {"faithfulness": {"mean": 0.9}, "citation_precision": {"mean": 0.8}},
        "summary": {"total_cases": 35, "latency_p95_ms": 9000, "mean_cost_usd": 0.001},
    }
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    table = render([pa, pb])
    assert "| Metric | BM25 | Hybrid |" in table
    assert "| Faithfulness (LLM judge) | 0.70 | 0.90 |" in table
    assert "| Citation precision | — | 0.80 |" in table
    assert "| Latency p95 | 12.0 s | 9.0 s |" in table
    assert "| Cost per answer | $0.0012 | $0.0010 |" in table
