from __future__ import annotations

import logging

from poc.agent.state import AgentState
from poc.llm.budget import BudgetExceededError
from poc.llm.provider import LLMMessage, LLMProvider

_log = logging.getLogger(__name__)


def _format_context(retrieved: list[dict]) -> str:
    parts = [f"[{c['chunk_id']}]\n{c['text']}" for c in retrieved[:10]]
    return "\n\n---\n\n".join(parts)


def _format_clusters(clusters: list[dict]) -> str:
    if not clusters:
        return "No thematic clusters identified."
    lines = []
    for cl in clusters:
        lines.append(f"Theme: {cl['label']} ({cl['size']} chunks)")
        for quote in cl.get("sample_quotes", [])[:1]:
            lines.append(f"  Sample: {quote[:100]}")
    return "\n".join(lines)


def make_summarize_node(llm: LLMProvider, model: str):
    """Factory: generates structured summary with citations."""

    async def summarize_node(state: AgentState) -> dict:
        retrieved = state.get("retrieved", [])
        clusters = state.get("clusters", [])

        if not retrieved:
            draft = "UNKNOWN: no relevant documents found."
            return {
                "draft": draft,
                "traces": list(state.get("traces", [])),
            }

        context = _format_context(retrieved)
        cluster_summary = _format_clusters(clusters)
        intent = state.get("intent", "general")

        system_prompt = (
            "You are a product insight analyst. Synthesize the provided context fragments "
            "into a structured answer. Rules:\n"
            "- Cite every claim with [chunk_id] notation\n"
            "- If context is insufficient, say 'UNKNOWN: insufficient context'\n"
            "- Structure: summary paragraph + key themes + 2-3 action items\n"
            "- Keep the answer concise (max 400 words)"
        )
        user_prompt = (
            f"Question: {state['question']}\n"
            f"Intent: {intent}\n\n"
            f"Thematic clusters:\n{cluster_summary}\n\n"
            f"Context fragments:\n{context}\n\n"
            "Provide a structured answer with citations:"
        )

        try:
            response = await llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                model=model,
                max_tokens=1024,
                temperature=0.0,
            )
            draft = response.content
            cost = response.cost_usd or 0.0
        except BudgetExceededError:
            raise  # hard-stop must reach the API layer (S3.T4)
        except Exception as exc:
            _log.warning("Summarize node failed: %s", exc)
            draft = "UNKNOWN: summarization failed."
            cost = 0.0

        traces = list(state.get("traces", []))
        traces.append({"node": "summarize", "draft_len": len(draft)})

        return {
            "draft": draft,
            "cost_usd": state.get("cost_usd", 0.0) + cost,
            "traces": traces,
        }

    return summarize_node
