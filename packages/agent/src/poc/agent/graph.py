from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from poc.agent.nodes.cluster import make_cluster_node
from poc.agent.nodes.groundedness import make_groundedness_node
from poc.agent.nodes.hitl import make_hitl_node
from poc.agent.nodes.intent import make_intent_node
from poc.agent.nodes.judge import make_judge_node
from poc.agent.nodes.plan import make_plan_node
from poc.agent.nodes.retrieve import make_retrieve_node
from poc.agent.nodes.summarize import make_summarize_node
from poc.agent.state import AgentState
from poc.llm.provider import LLMProvider
from poc.observability.tracing import traced
from poc.retrieval.hybrid import HybridRetriever

_log = logging.getLogger(__name__)


def _make_should_review(enable_hitl: bool):
    """Edge factory: route to 'hitl' on low confidence only when HITL is enabled."""

    def _should_review(state: AgentState) -> str:
        if enable_hitl and state.get("needs_human_review"):
            return "hitl"
        return "end"

    return _should_review


def build_graph(
    *,
    llm: LLMProvider,
    retriever: HybridRetriever,
    intent_model: str,
    plan_model: str,
    summarize_model: str,
    judge_model: str,
    enable_hitl: bool = False,
) -> StateGraph:
    """Build and compile the Product Insight Agent graph.

    When ``enable_hitl`` is True the graph pauses at the ``hitl`` node (via
    ``interrupt()``) whenever the judge or groundedness gate flags low confidence,
    and is compiled with a ``MemorySaver`` checkpointer so it can be resumed. When
    False, the ``groundedness`` gate routes straight to END and no checkpointer is
    attached.
    """

    graph = StateGraph(AgentState)

    # Register nodes. Each node function is wrapped in @traced so every graph step
    # becomes an OTel span (agent.intent, agent.plan, …) without touching the node
    # bodies — the instrumentation lives entirely at the wiring layer.
    graph.add_node("intent", traced("agent.intent")(make_intent_node(llm, intent_model)))
    graph.add_node("plan", traced("agent.plan")(make_plan_node(llm, plan_model)))
    graph.add_node("retrieve", traced("agent.retrieve")(make_retrieve_node(retriever)))
    graph.add_node("cluster", traced("agent.cluster")(make_cluster_node(llm, plan_model)))
    graph.add_node(
        "summarize", traced("agent.summarize")(make_summarize_node(llm, summarize_model))
    )
    graph.add_node("judge", traced("agent.judge")(make_judge_node(llm, judge_model)))
    graph.add_node(
        "groundedness",
        traced("agent.groundedness")(make_groundedness_node(llm, judge_model)),
    )
    graph.add_node("hitl", traced("agent.hitl")(make_hitl_node()))

    # Wire edges
    graph.set_entry_point("intent")
    graph.add_edge("intent", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "cluster")
    graph.add_edge("cluster", "summarize")
    graph.add_edge("summarize", "judge")
    # Groundedness runs as a post-judge gate (S4.T3): per-claim verification of the
    # finalized answer, escalating to review when judgement.groundedness < threshold.
    graph.add_edge("judge", "groundedness")
    graph.add_conditional_edges(
        "groundedness", _make_should_review(enable_hitl), {"hitl": "hitl", "end": END}
    )
    graph.add_edge("hitl", END)

    checkpointer = MemorySaver() if enable_hitl else None
    return graph.compile(checkpointer=checkpointer)


def initial_state(
    question: str,
    *,
    tenant: str = "default",
    filters: dict[str, Any] | None = None,
) -> AgentState:
    return AgentState(
        question=question,
        tenant=tenant,
        filters=filters or {},
        intent="",
        intent_confidence=0.0,
        plan={},
        retrieved=[],
        clusters=[],
        draft="",
        judgement={},
        final=None,
        cost_usd=0.0,
        traces=[],
        needs_human_review=False,
        human_feedback=None,
    )
