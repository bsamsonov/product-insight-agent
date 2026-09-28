"""The agent forwards the judge verdict to Langfuse as trace scores."""

from __future__ import annotations

from poc.agent import agent as agent_module


def test_score_judgement_sends_judge_and_groundedness(monkeypatch):
    sent = []
    monkeypatch.setattr(
        agent_module.langfuse_client,
        "score_current_trace",
        lambda name, value, comment=None: sent.append((name, value, comment)),
    )
    agent_module._score_judgement({"score": 0.8, "reasoning": "ok", "groundedness": 0.5})
    assert sent == [("judge_score", 0.8, "ok"), ("groundedness", 0.5, None)]


def test_score_judgement_skips_missing_values(monkeypatch):
    sent = []
    monkeypatch.setattr(
        agent_module.langfuse_client,
        "score_current_trace",
        lambda name, value, comment=None: sent.append(name),
    )
    agent_module._score_judgement({"groundedness": None})
    assert sent == []
