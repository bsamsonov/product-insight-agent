from __future__ import annotations

import logging
import re

from poc.agent.state import AgentState
from poc.core.models import Answer, Citation
from poc.llm.budget import BudgetExceededError
from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

_log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[([^\]]+)\]")

_JUDGE_THRESHOLD = 0.6  # below this → needs human review
_ESCALATE_THRESHOLD = 0.5  # below this → re-judge once with the stronger escalate model


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


def make_judge_node(
    llm: LLMProvider,
    model: str,
    *,
    escalate_llm: LLMProvider | None = None,
):
    """Factory: LLM-as-judge evaluates the draft answer.

    When *escalate_llm* is provided (S3.T2: a RoleScopedLLM bound to the
    ``escalate`` route) and the first judgement scores below
    ``_ESCALATE_THRESHOLD``, the evaluation is re-run once through the stronger
    model and its verdict replaces the first one. Low confidence from a cheap
    judge is a signal to spend more, not to fail the answer outright.
    """

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

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = await llm.complete(
                messages,
                model=model,
                max_tokens=256,
                temperature=0.0,
                response_format=JudgeResult,
            )
            result = JudgeResult.model_validate_json(response.content)
            judgement = result.model_dump()
            cost = response.cost_usd or 0.0
        except BudgetExceededError:
            raise  # hard-stop must reach the API layer (S3.T4), never swallowed
        except Exception as exc:
            _log.warning("Judge node failed: %s", exc)
            judgement = {"score": 0.5, "passed": True, "reasoning": "judge unavailable"}
            cost = 0.0

        # S3.T2: low-confidence verdict from the cheap judge → one re-run through the
        # stronger escalate route. Escalation failure is fail-soft (keep first verdict);
        # budget breaches still propagate.
        escalated = False
        if judgement.get("score", 1.0) < _ESCALATE_THRESHOLD and escalate_llm is not None:
            try:
                response2 = await escalate_llm.complete(
                    messages,
                    model=model,  # RoleScopedLLM resolves the real model from router.yaml
                    max_tokens=256,
                    temperature=0.0,
                    response_format=JudgeResult,
                )
                judgement = JudgeResult.model_validate_json(response2.content).model_dump()
                cost += response2.cost_usd or 0.0
                escalated = True
            except BudgetExceededError:
                raise
            except Exception as exc:
                _log.warning("Judge escalation failed, keeping first verdict: %s", exc)

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
                "escalated": escalated,
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
