from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class EvalCase:
    id: str
    question: str
    expected_answer_substrings: list[str]
    expected_chunk_ids: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    case_id: str
    question: str
    answer_text: str
    cited_chunk_ids: list[str]
    faithfulness_score: float
    citation_precision_score: float
    substring_match: bool
    latency_ms: int
    cost_usd: float | None = None
    error: str | None = None


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                cases.append(EvalCase(**data))
    return cases


def _check_substrings(text: str, substrings: list[str]) -> bool:
    text_lower = text.lower()
    return all(s.lower() in text_lower for s in substrings)
