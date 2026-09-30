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
3. Return a JSON object with the list of claims and their verdicts.

Context:
{context}

Answer to evaluate:
{answer}

Respond with JSON only:
{{"claims": [{{"claim": "...", "verdict": "supported|partial|unsupported", "reasoning": "..."}}]}}\
"""


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "partial", "unsupported"]
    reasoning: str


class ClaimsEnvelope(BaseModel):
    """Structured-output schema sent to the LLM (``response_format``).

    An object, not a bare array: OpenAI-style ``json_object`` mode and most
    ``json_schema`` implementations require a top-level object.
    """

    claims: list[ClaimVerdict]


class GroundednessResult(BaseModel):
    # (supported_count + 0.5 * partial_count) / total_count.
    # ``None`` means the check could not be performed (unparseable LLM output) —
    # callers must treat that as "unknown", not as "ungrounded".
    score: float | None
    claims: list[ClaimVerdict]
    supported_count: int
    total_count: int

    @property
    def available(self) -> bool:
        return self.score is not None


def _compute_score(claims: list[ClaimVerdict]) -> tuple[float, int, int]:
    """Return (score, supported_count, total_count). partial counts as 0.5."""
    total = len(claims)
    if total == 0:
        return 0.0, 0, 0
    supported = sum(1 for c in claims if c.verdict == "supported")
    partial = sum(1 for c in claims if c.verdict == "partial")
    score = (supported + 0.5 * partial) / total
    return score, supported, total


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
    return text.strip()


def _parse_claims(raw: str) -> list[ClaimVerdict] | None:
    """Parse LLM output into claims; return None when it cannot be parsed.

    Accepts the structured-output shape ``{"claims": [...]}`` as well as a bare
    array (older prompts, providers that ignore ``response_format``), optionally
    wrapped in a markdown fence or surrounded by prose.
    """
    text = _strip_fence(raw)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Prose around the JSON: decode from the first "{" or "[" onwards.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch in "{[":
                try:
                    data, _ = decoder.raw_decode(text, i)
                    break
                except json.JSONDecodeError:
                    continue
    if isinstance(data, dict):
        data = data.get("claims")
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

        Asks for structured output (``response_format=ClaimsEnvelope``). If the LLM
        output still cannot be parsed, returns ``score=None`` (check unavailable)
        rather than raising or reporting 0.0 — an unparseable verdict says nothing
        about the answer, so it must not flag it as ungrounded.
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
            response_format=ClaimsEnvelope,
        )

        claims = _parse_claims(response.content)
        if claims is None:
            logger.warning(
                "groundedness_checker: failed to parse LLM output",
                extra={"raw_output": response.content[:200]},
            )
            return GroundednessResult(score=None, claims=[], supported_count=0, total_count=0)

        score, supported_count, total_count = _compute_score(claims)
        return GroundednessResult(
            score=score,
            claims=claims,
            supported_count=supported_count,
            total_count=total_count,
        )
