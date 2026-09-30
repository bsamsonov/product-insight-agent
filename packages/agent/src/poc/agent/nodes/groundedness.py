from __future__ import annotations

import logging

from poc.agent.state import AgentState
from poc.guardrails.groundedness import GroundednessChecker
from poc.llm.budget import BudgetExceededError
from poc.llm.provider import LLMProvider

_log = logging.getLogger(__name__)

# Below this groundedness score the answer is escalated for human review (S4.T3 spec).
GROUNDEDNESS_THRESHOLD = 0.7


def make_groundedness_node(
    llm: LLMProvider, model: str, *, threshold: float = GROUNDEDNESS_THRESHOLD
):
    """Factory: per-claim groundedness gate, runs after the judge node.

    Decomposes the (already judged) answer into atomic claims and verifies each
    against the retrieved context via :class:`GroundednessChecker`. Writes the
    fraction-supported score into ``judgement["groundedness"]`` and escalates to
    human review when it falls below ``threshold`` — the S4.T3 contract
    (``judgement.groundedness < 0.7 → escalate``).
    """

    checker = GroundednessChecker(llm, model=model)

    async def groundedness_node(state: AgentState) -> dict:
        draft = state.get("draft", "")
        retrieved = state.get("retrieved", [])
        # Carry the judge's existing verdict forward; we only add a field.
        judgement = dict(state.get("judgement", {}))
        traces = list(state.get("traces", []))

        # No real answer to check (judge already flagged review) — pass through.
        if not draft or draft.startswith("UNKNOWN"):
            judgement["groundedness"] = None
            return {"judgement": judgement, "traces": traces}

        # Prefer the finalized answer text; fall back to the raw draft.
        final = state.get("final")
        answer_text = final.text if final is not None else draft
        context_chunks = [c["text"] for c in retrieved]

        try:
            result = await checker.check(answer_text, context_chunks)
        except BudgetExceededError:
            raise  # hard-stop must reach the API layer (S3.T4)
        except Exception as exc:  # checker is fail-soft, but never crash the graph
            _log.warning("Groundedness node failed: %s", exc)
            judgement["groundedness"] = None
            return {"judgement": judgement, "traces": traces}

        if not result.available:
            # Unparseable checker output: unknown, not ungrounded — do not escalate.
            judgement["groundedness"] = None
            traces.append({"node": "groundedness", "score": None, "unavailable": True})
            return {"judgement": judgement, "traces": traces}

        score = result.score
        judgement["groundedness"] = score
        escalate = score < threshold

        traces.append(
            {
                "node": "groundedness",
                "score": score,
                "supported_count": result.supported_count,
                "total_count": result.total_count,
                "escalate": escalate,
            }
        )

        out: dict = {"judgement": judgement, "traces": traces}
        if escalate:
            # Combine with any prior signal from the judge — never downgrade.
            out["needs_human_review"] = True
        return out

    return groundedness_node
