from __future__ import annotations

import logging

from langgraph.types import interrupt
from poc.agent.state import AgentState
from poc.core.models import Answer

_log = logging.getLogger(__name__)


def make_hitl_node():
    """Factory: returns an async node that pauses for a human approve/reject decision.

    The node pauses the graph via ``interrupt()`` and hands the draft + judge score to
    the caller. The caller resumes with ``Command(resume={"approved": bool})`` (or a
    bare bool). On reject the final answer is marked as withheld.
    """

    async def hitl_node(state: AgentState) -> dict:
        score = state.get("judgement", {}).get("score", 0.0)
        _log.warning(
            "HITL checkpoint triggered for question: %s (judge_score=%.2f)",
            state.get("question"),
            score,
        )

        decision = interrupt(
            {
                "question": state.get("question"),
                "draft": state.get("draft"),
                "judge_score": score,
                "judge_reasoning": state.get("judgement", {}).get("reasoning"),
            }
        )
        approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)

        traces = [*list(state.get("traces", [])), {"node": "hitl", "approved": approved}]
        update: dict = {"human_feedback": {"approved": approved}, "traces": traces}

        if not approved:
            final = state.get("final")
            text = final.text if final else state.get("draft", "")
            update["final"] = Answer(
                question=state["question"],
                text=f"REJECTED by human reviewer. Draft was:\n{text}",
                citations=final.citations if final else [],
                used_chunks=final.used_chunks if final else [],
                cost_usd=state.get("cost_usd", 0.0),
                latency_ms=final.latency_ms if final else 0,
            )
        return update

    return hitl_node
