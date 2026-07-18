from __future__ import annotations

import logging
import re

from poc.agent.state import AgentState
from poc.core.models import Answer, Citation
from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

_log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[([^\]]+)\]")

_JUDGE_THRESHOLD = 0.6  # below this → needs human review


class JudgeResult(BaseModel):
    score: float
    passed: bool
    reasoning: str


def _extract_citations(text: str, retrieved: list[dict]) -> list[Citation]:
    chunk_map = {c["chunk_id"]: c for c in retrieved}
    cited_ids = _CITATION_RE.findall(text)
    seen: set[str] = set()
    citations: list[Citation] = []
    for cid in cited_ids:
        if cid in chunk_map and cid not in seen:
            seen.add(cid)
            c = chunk_map[cid]
            citations.append(
                Citation(
                    chunk_id=cid,
                    text_excerpt=c["text"][:200],
                    score=c.get("score", 0.0),
                )
            )
    return citations


def make_judge_node(llm: LLMProvider, model: str):
    """Factory: LLM-as-judge evaluates the draft answer."""

    async def judge_node(state: AgentState) -> dict:
        draft = state.get("draft", "")
        retrieved = state.get("retrieved", [])

        if not draft or draft.startswith("UNKNOWN"):
            judgement = {"score": 0.0, "passed": False, "reasoning": "no draft to judge"}
            return {
                "judgement": judgement,
                "needs_human_review": True,
                "traces": list(state.get("traces", [])),
            }

        context = "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in retrieved[:8])
        system_prompt = (
            "You are a quality evaluator. Assess whether the answer is:\n"
            "1. Factually supported by the context (no hallucinations)\n"
            "2. Complete (answers the question)\n"
            "3. Uses proper citations\n\n"
            'Respond with JSON: {"score": <0.0-1.0>, "passed": <bool>, "reasoning": "<brief>"}\n'
            "passed=true if score >= 0.6"
        )
        user_prompt = (
            f"Question: {state['question']}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer to judge:\n{draft}\n\n"
            "Evaluation (JSON):"
        )

        try:
            response = await llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                model=model,
                max_tokens=256,
                temperature=0.0,
                response_format=JudgeResult,
            )
            result = JudgeResult.model_validate_json(response.content)
            judgement = result.model_dump()
            cost = response.cost_usd or 0.0
        except Exception as exc:
            _log.warning("Judge node failed: %s", exc)
            judgement = {"score": 0.5, "passed": True, "reasoning": "judge unavailable"}
            cost = 0.0

        needs_review = judgement.get("score", 1.0) < _JUDGE_THRESHOLD

        # Build final Answer
        citations = _extract_citations(draft, retrieved)
        final = Answer(
            question=state["question"],
            text=draft,
            citations=citations,
            used_chunks=[c["chunk_id"] for c in retrieved],
            cost_usd=state.get("cost_usd", 0.0) + cost,
            latency_ms=0,
        )

        traces = list(state.get("traces", []))
        traces.append(
            {
                "node": "judge",
                "score": judgement.get("score"),
                "passed": judgement.get("passed"),
            }
        )

        return {
            "judgement": judgement,
            "final": final,
            "needs_human_review": needs_review,
            "cost_usd": state.get("cost_usd", 0.0) + cost,
            "traces": traces,
        }

    return judge_node
