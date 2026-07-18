from __future__ import annotations

import logging
from typing import Any

from poc.agent.state import AgentState
from poc.retrieval.hybrid import HybridRetriever

_log = logging.getLogger(__name__)


def make_retrieve_node(retriever: HybridRetriever):
    """Factory: returns a node that retrieves chunks based on the plan."""

    async def retrieve_node(state: AgentState) -> dict:
        plan = state.get("plan", {})
        filters: dict[str, Any] = {k: v for k, v in plan.get("filters", {}).items() if v}
        top_k = plan.get("top_k", 10)

        try:
            scored_chunks = retriever.retrieve(
                state["question"],
                top_k=top_k,
                filters=filters or None,
            )
            retrieved = [
                {
                    "chunk_id": sc.chunk.id,
                    "doc_id": sc.chunk.doc_id,
                    "text": sc.chunk.text,
                    "score": sc.score,
                    "metadata": sc.chunk.metadata,
                }
                for sc in scored_chunks
            ]
        except Exception as exc:
            _log.warning("Retrieval failed: %s", exc)
            retrieved = []

        traces = list(state.get("traces", []))
        traces.append({"node": "retrieve", "num_chunks": len(retrieved)})

        return {
            "retrieved": retrieved,
            "traces": traces,
        }

    return retrieve_node
