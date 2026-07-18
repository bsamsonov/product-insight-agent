"""RAGAS-style evaluation metrics suite.

Provides both LLM-based (semantic) and heuristic metrics for evaluating
RAG pipeline quality.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

_log = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    score: float  # 0.0-1.0
    reasoning: str = ""
    details: dict = field(default_factory=dict)


# ── Pydantic models for structured LLM output ────────────────────────────────


class _FaithfulnessJudge(BaseModel):
    claims: list[str]
    supported_claims: list[str]
    unsupported_claims: list[str]
    score: float


class _RelevanceJudge(BaseModel):
    is_relevant: bool
    reasoning: str
    score: float  # 0.0 or 1.0


class _ContextRelevanceJudge(BaseModel):
    relevant_contexts: list[int]  # indices of relevant contexts (0-based)
    irrelevant_contexts: list[int]
    score: float


class _RecallJudge(BaseModel):
    is_covered: bool
    reasoning: str
    score: float


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_json_fallback(content: str, model_class: type[BaseModel]) -> BaseModel:
    """Parse JSON from raw text content when structured output isn't available."""
    # Try direct parse first
    try:
        return model_class.model_validate_json(content)
    except Exception:
        pass

    # Extract JSON block from markdown code fence or raw braces
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        return model_class.model_validate_json(json_match.group(1))

    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        return model_class.model_validate_json(brace_match.group(0))

    raise ValueError(f"Cannot parse JSON from LLM content: {content[:200]}")


async def _llm_judge(
    messages: list[LLMMessage],
    llm: LLMProvider,
    model: str,
    max_tokens: int,
    model_class: type[BaseModel],
) -> BaseModel:
    """Call LLM with structured output, falling back to JSON parsing."""
    try:
        response = await llm.complete(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format=model_class,
        )
    except TypeError:
        # response_format not supported by this provider/mock — call without it
        response = await llm.complete(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    return _parse_json_fallback(response.content, model_class)


# ── LLM-based metrics ─────────────────────────────────────────────────────────


async def faithfulness(
    answer: str,
    contexts: list[str],
    llm: LLMProvider,
    *,
    model: str,
    max_tokens: int = 800,
) -> MetricResult:
    """Fraction of answer claims supported by retrieved contexts.

    Prompt: extract atomic claims from answer, then verify each against context.

    Score = supported_claims / total_claims.
    """
    context_text = "\n\n---\n\n".join(contexts[:10])

    system_msg = LLMMessage(
        role="system",
        content=(
            "You are an impartial evaluator. Your task is to assess faithfulness of an answer "
            "to given context documents.\n\n"
            "Step 1: Extract every atomic factual claim from the answer.\n"
            "Step 2: For each claim, determine whether it is directly supported by the context.\n\n"
            "Respond ONLY with valid JSON matching this schema:\n"
            '{"claims": ["..."], "supported_claims": ["..."], "unsupported_claims": ["..."], '
            '"score": <float 0.0-1.0>}\n\n'
            "score = len(supported_claims) / len(claims). If claims is empty, score = 1.0."
        ),
    )
    user_msg = LLMMessage(
        role="user",
        content=(
            f"Context:\n{context_text}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            "Faithfulness evaluation (JSON):"
        ),
    )

    try:
        raw = await _llm_judge([system_msg, user_msg], llm, model, max_tokens, _FaithfulnessJudge)
        judge = _FaithfulnessJudge.model_validate(raw)
        total = len(judge.claims)
        supported = len(judge.supported_claims)
        score = supported / total if total > 0 else 1.0
        score = max(0.0, min(1.0, score))
        return MetricResult(
            name="faithfulness",
            score=score,
            reasoning=f"{supported}/{total} claims supported",
            details={
                "claims": judge.claims,
                "supported_claims": judge.supported_claims,
                "unsupported_claims": judge.unsupported_claims,
            },
        )
    except Exception as exc:
        _log.warning("faithfulness metric failed: %s", exc)
        return MetricResult(name="faithfulness", score=0.0, reasoning=f"eval error: {exc}")


async def answer_relevance(
    question: str,
    answer: str,
    llm: LLMProvider,
    *,
    model: str,
    max_tokens: int = 400,
) -> MetricResult:
    """Whether the answer addresses the question.

    Asks the LLM to judge if the answer is relevant to and attempts to answer
    the question. Score is 1.0 (relevant) or 0.0 (not relevant).
    """
    system_msg = LLMMessage(
        role="system",
        content=(
            "You are an impartial evaluator. Assess whether the given answer addresses "
            "the question.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"is_relevant": <true|false>, "reasoning": "<brief explanation>", '
            '"score": <1.0 if relevant, 0.0 if not>}'
        ),
    )
    user_msg = LLMMessage(
        role="user",
        content=f"Question: {question}\n\nAnswer: {answer}\n\nRelevance evaluation (JSON):",
    )

    try:
        judge = await _llm_judge([system_msg, user_msg], llm, model, max_tokens, _RelevanceJudge)
        return MetricResult(
            name="answer_relevance",
            score=max(0.0, min(1.0, judge.score)),
            reasoning=judge.reasoning,
        )
    except Exception as exc:
        _log.warning("answer_relevance metric failed: %s", exc)
        return MetricResult(name="answer_relevance", score=0.0, reasoning=f"eval error: {exc}")


async def context_precision(
    question: str,
    contexts: list[str],
    llm: LLMProvider,
    *,
    model: str,
    max_tokens: int = 400,
) -> MetricResult:
    """Fraction of retrieved contexts that are relevant to the question.

    Score = relevant_contexts / total_contexts. 1.0 if no contexts.
    """
    if not contexts:
        return MetricResult(name="context_precision", score=1.0, reasoning="no contexts")

    contexts_text = "\n\n".join(f"[Context {i}]: {ctx[:500]}" for i, ctx in enumerate(contexts))
    system_msg = LLMMessage(
        role="system",
        content=(
            "You are an impartial evaluator. Given a question and a list of retrieved contexts, "
            "determine which contexts are relevant to answering the question.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"relevant_contexts": [<0-based indices>], '
            '"irrelevant_contexts": [<0-based indices>], '
            '"score": <relevant / total, float 0.0-1.0>}'
        ),
    )
    user_msg = LLMMessage(
        role="user",
        content=(
            f"Question: {question}\n\n"
            f"Retrieved contexts:\n{contexts_text}\n\n"
            "Context precision evaluation (JSON):"
        ),
    )

    try:
        judge = await _llm_judge(
            [system_msg, user_msg], llm, model, max_tokens, _ContextRelevanceJudge
        )
        total = len(contexts)
        relevant = len(judge.relevant_contexts)
        score = relevant / total if total > 0 else 1.0
        score = max(0.0, min(1.0, score))
        return MetricResult(
            name="context_precision",
            score=score,
            reasoning=f"{relevant}/{total} contexts relevant",
            details={
                "relevant_indices": judge.relevant_contexts,
                "irrelevant_indices": judge.irrelevant_contexts,
            },
        )
    except Exception as exc:
        _log.warning("context_precision metric failed: %s", exc)
        return MetricResult(name="context_precision", score=0.0, reasoning=f"eval error: {exc}")


async def context_recall(
    answer: str,
    expected: str,
    contexts: list[str],
    llm: LLMProvider,
    *,
    model: str,
    max_tokens: int = 400,
) -> MetricResult:
    """Whether contexts cover the expected answer (are contexts sufficient?).

    Asks whether, given only the retrieved contexts, one could derive
    the expected answer. Score is 1.0 (covered) or 0.0 (not covered).
    """
    context_text = "\n\n---\n\n".join(contexts[:8])
    system_msg = LLMMessage(
        role="system",
        content=(
            "You are an impartial evaluator. Determine whether the retrieved contexts contain "
            "enough information to produce the expected answer.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"is_covered": <true|false>, "reasoning": "<brief explanation>", '
            '"score": <1.0 if covered, 0.0 if not>}'
        ),
    )
    user_msg = LLMMessage(
        role="user",
        content=(
            f"Expected answer: {expected}\n\n"
            f"Retrieved contexts:\n{context_text}\n\n"
            "Context recall evaluation (JSON):"
        ),
    )

    try:
        judge = await _llm_judge([system_msg, user_msg], llm, model, max_tokens, _RecallJudge)
        return MetricResult(
            name="context_recall",
            score=max(0.0, min(1.0, judge.score)),
            reasoning=judge.reasoning,
        )
    except Exception as exc:
        _log.warning("context_recall metric failed: %s", exc)
        return MetricResult(name="context_recall", score=0.0, reasoning=f"eval error: {exc}")


# ── Non-LLM metrics ───────────────────────────────────────────────────────────


def citation_precision(citations: list[str], expected_chunk_ids: list[str]) -> MetricResult:
    """Fraction of cited chunks in expected_chunk_ids set. 1.0 if no citations expected."""
    if not expected_chunk_ids:
        return MetricResult(name="citation_precision", score=1.0, reasoning="no expectation")
    if not citations:
        return MetricResult(name="citation_precision", score=0.0, reasoning="no citations provided")
    expected_set = set(expected_chunk_ids)
    hits = sum(1 for c in citations if c in expected_set)
    score = hits / len(citations)
    return MetricResult(
        name="citation_precision",
        score=score,
        reasoning=f"{hits}/{len(citations)} citations matched",
    )


def groundedness_heuristic(answer: str, context_chunks: list[str]) -> MetricResult:
    """Heuristic: fraction of answer sentences that contain a [citation] marker."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 20]
    if not sentences:
        return MetricResult(
            name="groundedness_heuristic",
            score=1.0,
            reasoning="no substantive sentences found",
        )
    citation_re = re.compile(r"\[[^\]]+\]")
    grounded = sum(1 for s in sentences if citation_re.search(s))
    score = grounded / len(sentences)
    return MetricResult(
        name="groundedness_heuristic",
        score=score,
        reasoning=f"{grounded}/{len(sentences)} sentences have citation markers",
    )


# ── Registry helper ───────────────────────────────────────────────────────────

_NON_LLM_METRICS = {
    "citation_precision": citation_precision,
    "groundedness_heuristic": groundedness_heuristic,
}

_LLM_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevance": answer_relevance,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

ALL_METRICS = list(_NON_LLM_METRICS.keys()) + list(_LLM_METRICS.keys())
