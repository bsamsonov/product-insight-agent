from __future__ import annotations

import logging
from typing import Any

from poc.agent.state import AgentState
from poc.llm.provider import LLMProvider

_log = logging.getLogger(__name__)

# Simple k-means-like grouping by keyword similarity (no sklearn needed for small sets)
_CLUSTER_THEMES = {
    "comfort & fit": ["comfort", "fit", "size", "tight", "loose", "wide", "narrow"],
    "durability & quality": ["durable", "quality", "wear", "last", "broken", "defect"],
    # Sports & outdoors gear, not only running shoes: bottles, packs, tents, cleats, reels…
    "performance": [
        "performance",
        "grip",
        "traction",
        "sturdy",
        "stable",
        "waterproof",
        "lightweight",
        "effective",
        "sport",
        "athletic",
    ],
    "appearance": ["look", "style", "color", "design", "aesthetic"],
    "value": ["price", "value", "worth", "money", "cheap", "expensive"],
    "customer service": ["service", "support", "return", "exchange", "ship", "deliver"],
    "other": [],
}


def _assign_cluster(text: str) -> str:
    text_lower = text.lower()
    best_cluster = "other"
    best_count = 0
    for cluster, keywords in _CLUSTER_THEMES.items():
        if cluster == "other":
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_cluster = cluster
    return best_cluster


def make_cluster_node(llm: LLMProvider, model: str):
    """Factory: groups retrieved chunks into thematic clusters."""

    async def cluster_node(state: AgentState) -> dict:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {"clusters": [], "traces": list(state.get("traces", []))}

        # Group chunks by theme
        groups: dict[str, list[dict[str, Any]]] = {}
        for chunk in retrieved:
            label = _assign_cluster(chunk["text"])
            groups.setdefault(label, []).append(chunk)

        # Remove empty "other" if other groups exist
        if len(groups) > 1 and "other" in groups:
            del groups["other"]

        clusters = [
            {
                "label": label,
                "member_chunk_ids": [c["chunk_id"] for c in chunks],
                "sample_quotes": [c["text"][:150] for c in chunks[:2]],
                "size": len(chunks),
            }
            for label, chunks in groups.items()
        ]

        traces = list(state.get("traces", []))
        traces.append({"node": "cluster", "num_clusters": len(clusters)})

        return {
            "clusters": clusters,
            "traces": traces,
        }

    return cluster_node
