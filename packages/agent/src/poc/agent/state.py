from __future__ import annotations

from typing import Any, TypedDict

from poc.core.models import Answer


class AgentState(TypedDict):
    """State flowing through the agent graph."""

    # Input
    question: str
    tenant: str
    filters: dict[str, Any]

    # Intent classification
    intent: str  # "product_issue" | "feature_request" | "comparison" | "general" | "other"
    intent_confidence: float

    # Retrieval planning
    plan: dict[str, Any]  # {filters, top_k, strategy}

    # Retrieved chunks (list of ScoredChunk-like dicts)
    retrieved: list[dict[str, Any]]

    # Clusters (list of {label, member_chunk_ids, sample_quotes})
    clusters: list[dict[str, Any]]

    # LLM-generated draft answer
    draft: str

    # Judge evaluation (+ groundedness gate appends "groundedness": float | None)
    judgement: dict[str, Any]  # {score, passed, reasoning, groundedness}

    # Final answer
    final: Answer | None

    # Accounting
    cost_usd: float
    traces: list[dict[str, Any]]  # structured trace events

    # HITL
    needs_human_review: bool
    human_feedback: str | None
