from __future__ import annotations

import logging

from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

_log = logging.getLogger(__name__)


class FaithfulnessResult(BaseModel):
    score: float
    reasoning: str


async def faithfulness(
    answer_text: str,
    context_chunks: list[str],
    *,
    llm: LLMProvider,
    model: str,
    max_tokens: int = 256,
) -> float:
    """LLM-as-judge faithfulness score (0.0-1.0).

    Checks whether all claims in the answer are supported by context.
    """
    context = "\n\n---\n\n".join(context_chunks[:10])  # limit context size
    system_prompt = (
        "You are an impartial evaluator. Assess whether ALL claims in the answer "
        "are supported by the given context. Respond with JSON: "
        '{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}'
    )
    user_prompt = (
        f"Context:\n{context}\n\nAnswer to evaluate:\n{answer_text}\n\nFaithfulness score (JSON):"
    )

    try:
        response = await llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format=FaithfulnessResult,
        )
        result = FaithfulnessResult.model_validate_json(response.content)
        return max(0.0, min(1.0, result.score))
    except Exception as exc:
        _log.warning("Faithfulness eval failed: %s", exc)
        return 0.0


def citation_precision(
    cited_chunk_ids: list[str],
    expected_chunk_ids: list[str],
) -> float:
    """Fraction of cited chunks that are in the expected set."""
    if not cited_chunk_ids:
        return 0.0
    expected = set(expected_chunk_ids)
    correct = sum(1 for cid in cited_chunk_ids if cid in expected)
    return correct / len(cited_chunk_ids)


def citation_recall(
    cited_chunk_ids: list[str],
    expected_chunk_ids: list[str],
) -> float:
    """Fraction of expected chunks that were cited."""
    if not expected_chunk_ids:
        return 1.0
    cited = set(cited_chunk_ids)
    correct = sum(1 for cid in expected_chunk_ids if cid in cited)
    return correct / len(expected_chunk_ids)
