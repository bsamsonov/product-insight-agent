from __future__ import annotations

import json
import logging
from typing import Literal

from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a factual accuracy evaluator. Given an answer and context chunks, you must:
1. Break the answer into atomic factual claims (1-2 sentences each)
2. For each claim, determine if it is:
   - "supported": explicitly supported by the context
   - "partial": partially supported or implied
   - "unsupported": not found in context or contradicts it
3. Return a JSON array of claims with verdicts.

Context:
{context}

Answer to evaluate:
{answer}

Respond with JSON array:
[{{"claim": "...", "verdict": "supported|partial|unsupported", "reasoning": "..."}}]\
"""


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "partial", "unsupported"]
    reasoning: str


class GroundednessResult(BaseModel):
    score: float  # supported_count + 0.5 * partial_count / total_count
    claims: list[ClaimVerdict]
    supported_count: int
    total_count: int


def _compute_score(claims: list[ClaimVerdict]) -> tuple[float, int, int]:
    """Return (score, supported_count, total_count). partial counts as 0.5."""
    total = len(claims)
    if total == 0:
        return 0.0, 0, 0
    supported = sum(1 for c in claims if c.verdict == "supported")
    partial = sum(1 for c in claims if c.verdict == "partial")
    score = (supported + 0.5 * partial) / total
    return score, supported, total


def _parse_claims(raw: str) -> list[ClaimVerdict] | None:
    """Parse LLM JSON output; return None on any error."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # LLM may wrap JSON in markdown fences; strip and retry
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
            try:
                data = json.loads(inner)
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, list):
        return None

    try:
        return [ClaimVerdict.model_validate(item) for item in data]
    except Exception:
        return None


class GroundednessChecker:
    """LLM-as-judge: checks if answer claims are supported by retrieved context."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens

    async def check(
        self,
        answer_text: str,
        context_chunks: list[str],
        *,
        max_chunks: int = 8,
    ) -> GroundednessResult:
        """Atomize answer into claims, verify each against context via LLM.

        Falls back to score=0.0, claims=[] if LLM returns unparseable output,
        rather than raising — callers must not crash because of an eval utility.
        """
        chunks = context_chunks[:max_chunks]
        context_str = "\n\n---\n\n".join(chunks) if chunks else "(no context provided)"

        prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            context=context_str,
            answer=answer_text,
        )

        messages: list[LLMMessage] = [LLMMessage(role="user", content=prompt)]

        response = await self._llm.complete(
            messages,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        claims = _parse_claims(response.content)
        if claims is None:
            logger.warning(
                "groundedness_checker: failed to parse LLM output",
                extra={"raw_output": response.content[:200]},
            )
            return GroundednessResult(score=0.0, claims=[], supported_count=0, total_count=0)

        score, supported_count, total_count = _compute_score(claims)
        return GroundednessResult(
            score=score,
            claims=claims,
            supported_count=supported_count,
            total_count=total_count,
        )
