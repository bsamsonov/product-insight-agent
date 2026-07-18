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


# S3.T2: which router role each graph node speaks as when routing is enabled.
# Kept at the wiring layer — nodes stay role-agnostic, router.yaml owns models.
_NODE_ROLES: dict[str, str] = {
    "intent": "classifier",
    "plan": "planner",
    "cluster": "planner",
    "summarize": "summarizer",
    "judge": "judge",
    "groundedness": "judge",
}


def build_graph(
    *,
    llm: LLMProvider,
    retriever: HybridRetriever,
    intent_model: str,
    plan_model: str,
    summarize_model: str,
    judge_model: str,
    enable_hitl: bool = False,
    router: Any | None = None,  # RoutedLLM — enables role-based routing (S3.T2)
) -> StateGraph:
    """Build and compile the Product Insight Agent graph.

    When ``enable_hitl`` is True the graph pauses at the ``hitl`` node (via
    ``interrupt()``) whenever the judge or groundedness gate flags low confidence,
    and is compiled with a ``MemorySaver`` checkpointer so it can be resumed. When
    False, the ``groundedness`` gate routes straight to END and no checkpointer is
    attached.

    When ``router`` (a :class:`~poc.llm.router.RoutedLLM`) is given, every node
    talks to the LLM through a :class:`~poc.llm.role_provider.RoleScopedLLM`
    bound to its role in ``_NODE_ROLES`` — provider+model resolution, fallback,
    response cache, and budget enforcement then live in the router. The
    ``*_model`` arguments are ignored in that mode. The judge node additionally
    receives an ``escalate``-scoped provider for low-confidence re-runs.
    """

    escalate_llm = None
    if router is not None:
        from poc.llm.role_provider import RoleScopedLLM

        node_llm = {node: RoleScopedLLM(router, role) for node, role in _NODE_ROLES.items()}
        escalate_llm = RoleScopedLLM(router, "escalate")
    else:
        node_llm = dict.fromkeys(_NODE_ROLES, llm)

    graph = StateGraph(AgentState)

    # Register nodes. Each node function is wrapped in @traced so every graph step
    # becomes an OTel span (agent.intent, agent.plan, …) without touching the node
    # bodies — the instrumentation lives entirely at the wiring layer.
    graph.add_node(
        "intent", traced("agent.intent")(make_intent_node(node_llm["intent"], intent_model))
    )
    graph.add_node("plan", traced("agent.plan")(make_plan_node(node_llm["plan"], plan_model)))
    graph.add_node("retrieve", traced("agent.retrieve")(make_retrieve_node(retriever)))
    graph.add_node(
        "cluster", traced("agent.cluster")(make_cluster_node(node_llm["cluster"], plan_model))
    )
    graph.add_node(
        "summarize",
        traced("agent.summarize")(make_summarize_node(node_llm["summarize"], summarize_model)),
    )
    graph.add_node(
        "judge",
        traced("agent.judge")(
            make_judge_node(node_llm["judge"], judge_model, escalate_llm=escalate_llm)
        ),
    )
    graph.add_node(
        "groundedness",
        traced("agent.groundedness")(make_groundedness_node(node_llm["groundedness"], judge_model)),
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
