from __future__ import annotations

import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


@dataclass
class OutputCheckResult:
    passed: bool
    violations: list[str]
    sanitized_text: str


# Patterns that suggest the model is making things up or leaking info
_HALLUCINATION_SIGNALS: list[re.Pattern] = [
    re.compile(r"as\s+of\s+my\s+(?:last\s+)?(?:update|knowledge|training)", re.IGNORECASE),
    re.compile(r"I\s+(?:don'?t\s+have\s+access\s+to|cannot\s+access)", re.IGNORECASE),
    re.compile(r"based\s+on\s+my\s+(?:general\s+)?knowledge", re.IGNORECASE),
]

_MAX_OUTPUT_LENGTH = 8000


class RefusalDetector:
    """Detects LLM refusal responses.

    A refusal is a valid LLM output — it means the model consciously declined.
    It is NOT an error, but callers need to know about it to route correctly
    (e.g. return a structured 200 with ``refused=True`` instead of a 500).
    """

    REFUSAL_PATTERNS: typing.ClassVar[list[re.Pattern]] = [
        re.compile(r"I(?:'m| am) (?:unable|not able) to", re.IGNORECASE),
        re.compile(r"I (?:cannot|can't|won't)", re.IGNORECASE),
        re.compile(r"(?:As|Being) an AI", re.IGNORECASE),
        re.compile(r"I don'?t have (?:access|information)", re.IGNORECASE),
        re.compile(r"I (?:must|need to) (?:decline|refuse)", re.IGNORECASE),
        re.compile(r"(?:Sorry|Apologies),? (?:but )?I(?:'m| am) not", re.IGNORECASE),
        re.compile(r"This (?:request|question) (?:is|seems)", re.IGNORECASE),
    ]

    def is_refusal(self, text: str) -> bool:
        """Return True if *text* looks like an LLM refusal."""
        return any(p.search(text) for p in self.REFUSAL_PATTERNS)

    def get_standard_refusal(self, trace_id: str = "") -> str:
        """Return a structured refusal message suitable for end-users."""
        suffix = f" (trace_id={trace_id})" if trace_id else ""
        return (
            "The assistant was unable to answer this question. "
            "Please rephrase your request or contact support." + suffix
        )


async def retry_with_fix_prompt(
    original_text: str,
    schema: type[BaseModel],
    llm_complete_fn: Callable,  # async callable: (messages) -> str
    *,
    model: str,
    max_tokens: int,
) -> BaseModel | None:
    """Try to fix invalid JSON output via a single follow-up prompt.

    Sends a correction request to the LLM and attempts to parse the result.
    Returns a validated ``schema`` instance, or ``None`` if still invalid.
    No recursion — exactly one retry.
    """
    schema_name = schema.__name__
    fix_prompt = (
        f"The following text was supposed to be valid JSON matching the schema "
        f"'{schema_name}', but it failed validation.\n\n"
        f"Original output:\n```\n{original_text}\n```\n\n"
        f"Please return ONLY valid JSON that matches the schema, with no prose."
    )

    try:
        fixed_text = await llm_complete_fn(
            [{"role": "user", "content": fix_prompt}],
            model=model,
            max_tokens=max_tokens,
        )
        return schema.model_validate_json(fixed_text)
    except (ValidationError, json.JSONDecodeError, Exception):
        return None


def check_output(
    text: str,
    *,
    expected_schema: type[BaseModel] | None = None,
    max_length: int = _MAX_OUTPUT_LENGTH,
) -> OutputCheckResult:
    """Validate and sanitize LLM output.

    Checks:
    - Length limit
    - Schema validation (if expected_schema provided)
    - Hallucination signal detection
    - Refusal detection (informational — does NOT set passed=False)

    Notes on refusal:
        A refusal is a *valid* LLM response. ``passed`` stays ``True`` even
        when a refusal is detected so that callers can distinguish between
        "LLM output is broken" (passed=False) and "LLM consciously declined"
        (passed=True, violation ``refusal_detected`` present).
    """
    violations: list[str] = []
    sanitized = text

    # Length check
    if len(text) > max_length:
        violations.append(f"output_too_long:{len(text)}")
        sanitized = sanitized[:max_length]

    # Schema validation
    if expected_schema is not None:
        try:
            expected_schema.model_validate_json(sanitized)
        except (ValidationError, json.JSONDecodeError) as exc:
            violations.append(f"schema_violation:{type(exc).__name__}")

    # Hallucination signal detection (informational)
    for pattern in _HALLUCINATION_SIGNALS:
        if pattern.search(sanitized):
            violations.append(f"hallucination_signal:{pattern.pattern[:40]}")

    # Refusal detection (informational — valid outcome, not an error)
    detector = RefusalDetector()
    if detector.is_refusal(sanitized):
        violations.append("refusal_detected")

    passed = not any(v.startswith("schema_violation") for v in violations)
    return OutputCheckResult(passed=passed, violations=violations, sanitized_text=sanitized)


def check_groundedness(answer_text: str, context_chunks: list[str]) -> float:
    """Heuristic groundedness check: fraction of answer sentences with a citation.

    A sentence is considered grounded if it contains a [citation] marker.
    Returns 0.0-1.0.
    """
    # Split by sentence endings
    sentences = re.split(r"[.!?]+", answer_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return 1.0  # empty or very short → assume fine

    citation_re = re.compile(r"\[[^\]]+\]")
    grounded = sum(1 for s in sentences if citation_re.search(s))
    return grounded / len(sentences)
