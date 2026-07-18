"""S2.T2 acceptance test — end-to-end agent pipeline.

AC (docs/planning/06_implementation_plan.md, S2.T2):
    on 50 synthetic reviews the graph returns an Insight with at least 3
    clusters and >=5 citations.

The graph is exercised with a fake :class:`HybridRetriever` (returns 50 themed
synthetic reviews) and a fake :class:`LLMProvider` that routes its reply by the
system prompt of each node (intent / plan / summarize / judge). No network, no
Qdrant, no real model — the test asserts the *structural* contract of the
pipeline, not model quality.
"""

from __future__ import annotations

import re
from typing import Any

from poc.agent.graph import build_graph, initial_state
from poc.agent.nodes.cluster import _CLUSTER_THEMES
from poc.ingestion.chunker import Chunk
from poc.llm.provider import LLMMessage, LLMResponse
from poc.retrieval.qdrant_index import ScoredChunk

# --- Synthetic corpus: 50 reviews spread across the clusterable themes --------

_REVIEW_TEMPLATES = [
    "These shoes are so comfortable, the fit is perfect and not too tight.",
    "Great comfort, true to size, my feet feel relaxed all day long.",
    "Durable build quality, no wear after months, nothing broken yet.",
    "The quality is excellent and durable, they last and never break.",
    "Amazing performance, great grip and traction for running and sport.",
    "Athletic performance is top notch, the grip helps me run faster.",
    "I love the look and style, the color and design are beautiful.",
    "Stylish design, the color matches everything, great aesthetic look.",
    "Good value for the price, totally worth the money, not expensive.",
    "Cheap price but high value, definitely worth every bit of money.",
    "Customer service was great, easy return and fast shipping delivery.",
    "Support helped me exchange them, shipping and delivery were quick.",
]


def _make_corpus(n: int = 50) -> list[ScoredChunk]:
    chunks: list[ScoredChunk] = []
    for i in range(n):
        template = _REVIEW_TEMPLATES[i % len(_REVIEW_TEMPLATES)]
        chunk = Chunk(
            id=f"chunk-{i:03d}",
            doc_id=f"review-{i:03d}",
            text=template,
            position=0,
            metadata={"lang": "en", "rating": (i % 5) + 1},
        )
        # Decreasing score so order is deterministic.
        chunks.append(ScoredChunk(chunk=chunk, score=1.0 - i * 0.01))
    return chunks


class _FakeRetriever:
    """Stand-in for HybridRetriever — returns the synthetic corpus."""

    def __init__(self, corpus: list[ScoredChunk]) -> None:
        self._corpus = corpus

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        return self._corpus[:top_k]


_CTX_ID_RE = re.compile(r"\[([^\]]+)\]")


class _FakeLLM:
    """Routes replies by the system prompt of the calling node."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: Any = None,
        **_: Any,
    ) -> LLMResponse:
        system = messages[0].content
        user = messages[-1].content

        if "Classify the user's question" in system:
            content = '{"intent": "product_issue", "confidence": 0.9, "reasoning": "complaints"}'
        elif "retrieval planner" in system:
            content = (
                '{"filters": {"main_category": null, "verified_purchase": null, '
                '"rating": null, "product_title": null, "product_avg_rating": null, '
                '"helpful_vote": null, "product_rating_count": null}, '
                '"top_k": 10, "strategy": "hybrid", '
                '"focus_keywords": ["comfort", "quality"], "reasoning": "shoe issues"}'
            )
        elif "product insight analyst" in system:
            # Cite every chunk id present in the context fragments → >=5 citations.
            cited = _CTX_ID_RE.findall(user)
            citations = " ".join(f"[{cid}]" for cid in cited)
            content = (
                "Summary: customers are broadly positive about comfort and durability "
                f"{citations}.\n"
                "Key themes: comfort, durability, value.\n"
                "Action items: keep current fit; monitor sole wear; highlight value pricing."
            )
        elif "quality evaluator" in system:
            content = '{"score": 0.85, "passed": true, "reasoning": "well grounded"}'
        elif "factual accuracy evaluator" in user:
            # Groundedness checker prompt is a single user message; return claims
            # array with every claim supported → score 1.0.
            content = (
                '[{"claim": "customers are positive about comfort", '
                '"verdict": "supported", "reasoning": "context backs it"}, '
                '{"claim": "durability is praised", '
                '"verdict": "supported", "reasoning": "context backs it"}]'
            )
        else:  # pragma: no cover - defensive
            content = "{}"

        return LLMResponse(
            content=content,
            model=model,
            input_tokens=50,
            output_tokens=80,
            cost_usd=0.0005,
            latency_ms=42,
            finish_reason="stop",
        )


async def test_pipeline_returns_insight_with_clusters_and_citations() -> None:
    corpus = _make_corpus(50)
    graph = build_graph(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(corpus),
        intent_model="fake-intent",
        plan_model="fake-plan",
        summarize_model="fake-summarize",
        judge_model="fake-judge",
        enable_hitl=False,
    )

    state = initial_state("What do customers say about these shoes?")
    result = await graph.ainvoke(state)

    # --- AC 1: at least 3 thematic clusters -----------------------------------
    clusters = result["clusters"]
    assert len(clusters) >= 3, f"expected >=3 clusters, got {len(clusters)}"
    labels = {c["label"] for c in clusters}
    assert labels.issubset(set(_CLUSTER_THEMES)), f"unknown cluster labels: {labels}"

    # --- AC 2: at least 5 citations in the final Answer -----------------------
    final = result["final"]
    assert final is not None, "judge node should have produced a final Answer"
    assert len(final.citations) >= 5, f"expected >=5 citations, got {len(final.citations)}"

    # Every citation must point at a really-retrieved chunk (no hallucinated ids).
    retrieved_ids = {c["chunk_id"] for c in result["retrieved"]}
    for citation in final.citations:
        assert citation.chunk_id in retrieved_ids

    # --- Sanity: the draft is a real summary, not a degradation fallback ------
    assert not result["draft"].startswith("UNKNOWN")
    assert result["judgement"]["passed"] is True

    # --- S4.T3: groundedness gate ran and recorded a per-claim score ----------
    assert result["judgement"]["groundedness"] == 1.0
    assert any(t["node"] == "groundedness" for t in result["traces"])
