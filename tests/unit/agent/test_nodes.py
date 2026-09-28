from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from poc.agent.nodes.cluster import _assign_cluster, make_cluster_node
from poc.agent.nodes.intent import make_intent_node
from poc.agent.nodes.plan import make_plan_node
from poc.agent.nodes.retrieve import make_retrieve_node
from poc.agent.nodes.summarize import make_summarize_node
from poc.agent.state import AgentState
from poc.ingestion.chunker import Chunk
from poc.llm.provider import LLMResponse
from poc.retrieval.qdrant_index import ScoredChunk


def _make_state(**kwargs) -> AgentState:
    base = AgentState(
        question="What are common shoe complaints?",
        tenant="default",
        filters={},
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
    base.update(kwargs)
    return base


def _make_mock_llm(response_content: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_response = LLMResponse(
        content=response_content,
        model="test-model",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.001,
        latency_ms=100,
        finish_reason="stop",
    )
    mock_llm.complete = AsyncMock(return_value=mock_response)
    return mock_llm


class TestIntentNode:
    async def test_valid_intent_classification(self) -> None:
        content = '{"intent": "product_issue", "confidence": 0.9, "reasoning": "complaints"}'
        mock_llm = _make_mock_llm(content)
        node = make_intent_node(mock_llm, "test-model")
        state = _make_state()
        result = await node(state)

        assert result["intent"] == "product_issue"
        assert result["intent_confidence"] == 0.9
        assert result["cost_usd"] > 0

    async def test_invalid_intent_falls_back_to_general(self) -> None:
        content = '{"intent": "INVALID_INTENT", "confidence": 0.8, "reasoning": "test"}'
        mock_llm = _make_mock_llm(content)
        node = make_intent_node(mock_llm, "test-model")
        state = _make_state()
        result = await node(state)

        # Pydantic rejects the literal, node falls back to "general"
        assert result["intent"] == "general"

    async def test_llm_failure_returns_fallback(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("network error"))
        node = make_intent_node(mock_llm, "test-model")
        state = _make_state()
        result = await node(state)

        assert result["intent"] == "general"
        assert result["intent_confidence"] == 0.5

    async def test_traces_updated(self) -> None:
        content = '{"intent": "feature_request", "confidence": 0.7, "reasoning": "suggestions"}'
        mock_llm = _make_mock_llm(content)
        node = make_intent_node(mock_llm, "test-model")
        state = _make_state()
        result = await node(state)

        traces = result["traces"]
        assert any(t["node"] == "intent" for t in traces)

    async def test_cost_accumulates(self) -> None:
        content = '{"intent": "general", "confidence": 0.8, "reasoning": "general"}'
        mock_llm = _make_mock_llm(content)
        node = make_intent_node(mock_llm, "test-model")
        state = _make_state(cost_usd=0.005)
        result = await node(state)

        assert result["cost_usd"] > 0.005


class TestPlanNode:
    async def test_generates_retrieval_plan(self) -> None:
        content = (
            '{"filters": {"lang": "en", "product_id": null, "region": null}, '
            '"top_k": 10, "strategy": "hybrid", '
            '"focus_keywords": ["comfort", "fit"], "reasoning": "shoe issues"}'
        )
        mock_llm = _make_mock_llm(content)
        node = make_plan_node(mock_llm, "test-model")
        state = _make_state(intent="product_issue")
        result = await node(state)

        plan = result["plan"]
        assert plan["strategy"] == "hybrid"
        assert plan["top_k"] == 10
        assert isinstance(plan["focus_keywords"], list)

    async def test_plan_merges_existing_filters(self) -> None:
        content = (
            '{"filters": {"lang": null, "product_id": null, "region": null}, '
            '"top_k": 5, "strategy": "dense", '
            '"focus_keywords": [], "reasoning": "test"}'
        )
        mock_llm = _make_mock_llm(content)
        node = make_plan_node(mock_llm, "test-model")
        state = _make_state(filters={"product_id": "shoe-42"})
        result = await node(state)

        assert result["plan"]["filters"].get("product_id") == "shoe-42"

    async def test_plan_failure_returns_fallback(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        node = make_plan_node(mock_llm, "test-model")
        state = _make_state()
        result = await node(state)

        plan = result["plan"]
        assert "top_k" in plan
        assert "strategy" in plan


def _make_scored_chunk(chunk_id: str, text: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, doc_id=f"doc-{chunk_id}", text=text, position=0, metadata={}),
        score=score,
    )


class TestRetrieveNode:
    async def test_retrieves_and_serializes_chunks(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_scored_chunk("c1", "comfortable shoes", 0.95),
            _make_scored_chunk("c2", "durable build", 0.80),
        ]
        node = make_retrieve_node(retriever)
        state = _make_state(plan={"top_k": 5, "filters": {}})
        result = await node(state)

        assert len(result["retrieved"]) == 2
        assert result["retrieved"][0]["chunk_id"] == "c1"
        assert result["retrieved"][0]["score"] == 0.95
        assert any(t["node"] == "retrieve" for t in result["traces"])

    async def test_empty_filters_are_dropped(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        node = make_retrieve_node(retriever)
        state = _make_state(plan={"top_k": 7, "filters": {"rating": 5, "lang": None, "x": ""}})
        await node(state)

        _, kwargs = retriever.retrieve.call_args
        assert kwargs["filters"] == {"rating": 5}
        assert kwargs["top_k"] == 7

    async def test_no_filters_passes_none(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        node = make_retrieve_node(retriever)
        state = _make_state(plan={"top_k": 10, "filters": {}})
        await node(state)

        _, kwargs = retriever.retrieve.call_args
        assert kwargs["filters"] is None

    async def test_retriever_failure_degrades_gracefully(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.side_effect = Exception("Qdrant down")
        node = make_retrieve_node(retriever)
        state = _make_state(plan={"top_k": 10, "filters": {}})
        result = await node(state)

        assert result["retrieved"] == []


class TestClusterNode:
    def test_assign_cluster_by_keywords(self) -> None:
        assert _assign_cluster("These shoes are so comfortable and fit well") == "comfort & fit"
        assert _assign_cluster("durable quality, never broken") == "durability & quality"
        assert _assign_cluster("the price is great value for money") == "value"
        assert _assign_cluster("xyzzy nothing matches here") == "other"

    def test_performance_covers_general_outdoor_gear(self) -> None:
        assert _assign_cluster("the tent stayed waterproof and sturdy all night") == "performance"
        assert _assign_cluster("lightweight bottle, stable on the trail") == "performance"

    async def test_groups_chunks_into_themes(self) -> None:
        node = make_cluster_node(MagicMock(), "test-model")
        state = _make_state(
            retrieved=[
                {"chunk_id": "c1", "text": "very comfortable fit"},
                {"chunk_id": "c2", "text": "comfort and size are great"},
                {"chunk_id": "c3", "text": "durable quality, lasts long"},
                {"chunk_id": "c4", "text": "good value for the price"},
            ]
        )
        result = await node(state)

        labels = {c["label"] for c in result["clusters"]}
        assert "comfort & fit" in labels
        assert "durability & quality" in labels
        assert "value" in labels
        comfort = next(c for c in result["clusters"] if c["label"] == "comfort & fit")
        assert set(comfort["member_chunk_ids"]) == {"c1", "c2"}
        assert comfort["size"] == 2

    async def test_empty_retrieved_yields_no_clusters(self) -> None:
        node = make_cluster_node(MagicMock(), "test-model")
        state = _make_state(retrieved=[])
        result = await node(state)

        assert result["clusters"] == []

    async def test_other_dropped_when_real_themes_exist(self) -> None:
        node = make_cluster_node(MagicMock(), "test-model")
        state = _make_state(
            retrieved=[
                {"chunk_id": "c1", "text": "comfortable fit"},
                {"chunk_id": "c2", "text": "zzz unrelated text"},
            ]
        )
        result = await node(state)

        labels = {c["label"] for c in result["clusters"]}
        assert "other" not in labels
        assert "comfort & fit" in labels


class TestSummarizeNode:
    async def test_generates_draft_with_citations(self) -> None:
        mock_llm = _make_mock_llm("Customers like comfort [c1] and durability [c2].")
        node = make_summarize_node(mock_llm, "test-model")
        state = _make_state(
            retrieved=[
                {"chunk_id": "c1", "text": "comfortable"},
                {"chunk_id": "c2", "text": "durable"},
            ],
            clusters=[{"label": "comfort & fit", "size": 1, "sample_quotes": ["comfortable"]}],
        )
        result = await node(state)

        assert "[c1]" in result["draft"]
        assert result["cost_usd"] > 0

    async def test_empty_retrieved_returns_unknown(self) -> None:
        mock_llm = _make_mock_llm("should not be called")
        node = make_summarize_node(mock_llm, "test-model")
        state = _make_state(retrieved=[])
        result = await node(state)

        assert result["draft"].startswith("UNKNOWN")
        mock_llm.complete.assert_not_called()

    async def test_llm_failure_returns_unknown(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        node = make_summarize_node(mock_llm, "test-model")
        state = _make_state(retrieved=[{"chunk_id": "c1", "text": "comfortable"}])
        result = await node(state)

        assert result["draft"] == "UNKNOWN: summarization failed."
