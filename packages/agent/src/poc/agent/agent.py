from __future__ import annotations

import logging
from typing import Any

from poc.agent.graph import build_graph, initial_state
from poc.agent.state import AgentState
from poc.core.models import Answer
from poc.llm.provider import LLMProvider
from poc.observability.tracing import get_tracer
from poc.retrieval.hybrid import HybridRetriever

_log = logging.getLogger(__name__)


class ProductInsightAgent:
    """High-level agent that wraps the LangGraph graph."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        retriever: HybridRetriever,
        # model routing: fast for classification/planning, smart for synthesis & judge
        fast_model: str = "kr/claude-haiku-4.5",
        smart_model: str = "kr/claude-sonnet-4.5",
        judge_model: str = "kr/claude-sonnet-4.5",
        enable_hitl: bool = False,
    ) -> None:
        """
        Initializes the instance with the provided language model, retrieval method, and
        model configurations. The initialization constructs a graph using the specified
        models for various tasks such as intent detection, planning, summarization, and
        judgment.

        :param llm: The language model provider used for constructing the graph.
        :type llm: LLMProvider

        :param retriever: The hybrid retriever responsible for fetching information
            used in graph construction.
        :type retriever: HybridRetriever

        :param fast_model: The model used for intent, plan, and cluster tasks. Defaults
            to "kr/claude-haiku-4.5".
        :type fast_model: str

        :param smart_model: The model used for summarization. Defaults
            to "kr/claude-sonnet-4.5".
        :type smart_model: str

        :param judge_model: The model used for faithfulness evaluation. Defaults
            to "kr/claude-sonnet-4.5" — judge gates HITL, so reliability matters most here.
        :type judge_model: str
        """
        self._enable_hitl = enable_hitl
        self._graph = build_graph(
            llm=llm,
            retriever=retriever,
            intent_model=fast_model,
            plan_model=fast_model,
            summarize_model=smart_model,
            judge_model=judge_model,
            enable_hitl=enable_hitl,
        )

    async def run(
        self,
        question: str,
        *,
        tenant: str = "default",
        filters: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """
        Run the graph for a question and return the resulting AgentState.

        With HITL enabled the graph may pause at the ``hitl`` node; the returned state
        then contains ``__interrupt__``. Call :meth:`resume_review` with the same
        ``thread_id`` to approve/reject and finish the run.
        """
        state = initial_state(question, tenant=tenant, filters=filters)
        config = self._thread_config(thread_id)
        # Root span: makes every node span (agent.intent…judge) a child of one trace
        # instead of N detached traces. Langfuse then renders a single request tree.
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("tenant", tenant)
            result: AgentState = await self._graph.ainvoke(state, config=config)
            span.set_attribute("intent", result.get("intent", ""))
            return result

    async def resume_review(self, thread_id: str, *, approved: bool) -> AgentState:
        """Resume a paused HITL run with a human approve/reject decision."""
        from langgraph.types import Command

        config = self._thread_config(thread_id)
        result: AgentState = await self._graph.ainvoke(
            Command(resume={"approved": approved}), config=config
        )
        return result

    def _thread_config(self, thread_id: str | None) -> dict[str, Any] | None:
        """Build the config dict; checkpointer-backed graphs require a thread_id."""
        if not self._enable_hitl:
            return None
        return {"configurable": {"thread_id": thread_id or "default"}}

    async def get_answer(
        self,
        question: str,
        *,
        tenant: str = "default",
        filters: dict[str, Any] | None = None,
    ) -> Answer:
        """
        Asynchronously retrieves the final answer, or constructs a fallback answer based
        on the draft state, for the given question and optional filtering criteria.
        """
        state = await self.run(question, tenant=tenant, filters=filters)
        if state.get("final"):
            return state["final"]
        # Fallback: construct answer from draft
        return Answer(
            question=question,
            text=state.get("draft", "UNKNOWN: agent did not produce an answer."),
            cost_usd=state.get("cost_usd"),
        )
