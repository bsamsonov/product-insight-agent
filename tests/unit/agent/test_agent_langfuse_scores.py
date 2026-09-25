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


class _FakeGraph:
    def __init__(self, state):
        self._state = state

    async def ainvoke(self, state, config=None):
        return self._state


def _agent_with_state(state):
    agent = object.__new__(agent_module.ProductInsightAgent)
    agent._enable_hitl = False
    agent._graph = _FakeGraph(state)
    return agent


async def test_ask_exposes_judge_signals():
    from poc.core.models import Answer

    final = Answer(question="q", text="a [x__0__c0]")
    agent = _agent_with_state(
        {
            "final": final,
            "intent": "complaints",
            "judgement": {"score": 0.4, "groundedness": 0.9},
            "needs_human_review": True,
        }
    )
    result = await agent.ask("q")
    assert result.answer is final
    assert result.intent == "complaints"
    assert result.judge_score == 0.4
    assert result.groundedness == 0.9
    assert result.needs_human_review is True


async def test_ask_falls_back_to_draft():
    agent = _agent_with_state({"draft": "draft text", "judgement": {}})
    result = await agent.ask("q")
    assert result.answer.text == "draft text"
    assert result.judge_score is None
    assert result.needs_human_review is False
