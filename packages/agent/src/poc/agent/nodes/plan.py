from __future__ import annotations

import logging

from poc.agent.state import AgentState
from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

_log = logging.getLogger(__name__)


class _Filters(BaseModel):
    main_category: str | None = None
    verified_purchase: bool | None = None
    rating: float | None = None
    product_title: str | None = None
    product_avg_rating: float | None = None
    helpful_vote: int | None = None
    product_rating_count: int | None = None


class RetrievalPlan(BaseModel):
    filters: _Filters
    top_k: int
    strategy: str  # "dense" | "hybrid" | "bm25_only"
    focus_keywords: list[str]
    reasoning: str


def make_plan_node(llm: LLMProvider, model: str):
    """Factory: returns an async node that generates a retrieval plan."""

    async def plan_node(state: AgentState) -> dict:
        system_prompt = (
            "You are a retrieval planner. Given a user question and its intent, "
            "generate a retrieval plan as JSON:\n"
            "{\n"
            '  "filters": {\n'
            '    "main_category": null, "verified_purchase": null,\n'
            '    "rating": null, "product_title": null,\n'
            '    "product_avg_rating": null, "helpful_vote": null,\n'
            '    "product_rating_count": null\n'
            "  },\n"
            '  "top_k": 10,\n'
            '  "strategy": "hybrid",\n'
            '  "focus_keywords": ["key1", "key2"],\n'
            '  "reasoning": "brief explanation"\n'
            "}\n"
            "Available filter fields (set only if clearly implied by the question):\n"
            '- "main_category": one of "Sports & Outdoors", "AMAZON FASHION", "Amazon Home", '
            '"Health & Personal Care", "Automotive", "Pet Supplies", "Industrial & Scientific", '
            '"Cell Phones & Accessories", "Tools & Home Improvement", "Toys & Games", '
            '"Camera & Photo". null if not category-specific.\n'
            '- "verified_purchase": true for verified reviews only, '
            "false for unverified, null for all.\n"
            '- "rating": exact review star rating — 1.0, 2.0, 3.0, 4.0, or 5.0. '
            "Use when user asks about positive (4.0/5.0) or negative (1.0/2.0) reviews.\n"
            '- "product_title": exact product name string. '
            "Use only when user names a specific product.\n"
            '- "product_avg_rating": exact product average rating (float). Rarely needed.\n'
            '- "helpful_vote": exact number of helpful votes on the review. Rarely needed.\n'
            '- "product_rating_count": exact total number of ratings for the product. '
            "Rarely needed.\n"
            "strategy must be one of: dense, hybrid, bm25_only."
        )
        user_prompt = (
            f"Question: {state['question']}\n"
            f"Intent: {state.get('intent', 'general')}\n"
            f"Existing filters: {state.get('filters', {})}"
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
                response_format=RetrievalPlan,
            )
            plan_obj = RetrievalPlan.model_validate_json(response.content)
            plan = plan_obj.model_dump()
            # Merge with existing filters from the question
            plan["filters"].update({k: v for k, v in state.get("filters", {}).items() if v})
            cost = response.cost_usd or 0.0
        except Exception as exc:
            _log.warning("Plan generation failed: %s", exc)
            plan = {
                "filters": state.get("filters", {}),
                "top_k": 10,
                "strategy": "hybrid",
                "focus_keywords": [],
                "reasoning": "fallback plan",
            }
            cost = 0.0

        traces = list(state.get("traces", []))
        traces.append(
            {  # noqa: E501, RUF100
                "node": "plan",
                "strategy": plan.get("strategy"),
                "top_k": plan.get("top_k"),
            }
        )

        return {
            "plan": plan,
            "cost_usd": state.get("cost_usd", 0.0) + cost,
            "traces": traces,
        }

    return plan_node
