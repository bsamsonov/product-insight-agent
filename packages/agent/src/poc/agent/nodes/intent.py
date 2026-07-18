from __future__ import annotations

import logging
from typing import Literal

from poc.agent.state import AgentState
from poc.llm.provider import LLMMessage, LLMProvider
from pydantic import BaseModel

_log = logging.getLogger(__name__)

_VALID_INTENTS = {"product_issue", "feature_request", "comparison", "general", "other"}


class IntentClassification(BaseModel):
    intent: Literal["product_issue", "feature_request", "comparison", "general", "other"]
    confidence: float
    reasoning: str


def make_intent_node(llm: LLMProvider, model: str):
    """Factory: returns an async node function that classifies user intent."""

    async def intent_node(state: AgentState) -> dict:
        system_prompt = (
            "Classify the user's question about product reviews into one of these intents:\n"
            "- product_issue: complaints, problems, defects, negative experiences\n"
            "- feature_request: suggestions, improvements, wishlist\n"
            "- comparison: comparing products, brands, models\n"
            "- general: general questions about products, positive feedback, ratings\n"
            "- other: anything else\n\n"
            'Respond with JSON: {"intent": "<intent>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}'  # noqa: E501
        )
        user_prompt = f"Question: {state['question']}"

        try:
            response = await llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                model=model,
                max_tokens=128,
                temperature=0.0,
                response_format=IntentClassification,
            )
            result = IntentClassification.model_validate_json(response.content)
            intent = result.intent if result.intent in _VALID_INTENTS else "other"
            confidence = max(0.0, min(1.0, result.confidence))
            cost = response.cost_usd or 0.0
        except Exception as exc:
            _log.warning("Intent classification failed: %s", exc)
            intent = "general"
            confidence = 0.5
            cost = 0.0

        traces = list(state.get("traces", []))
        traces.append({"node": "intent", "intent": intent, "confidence": confidence})

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "cost_usd": state.get("cost_usd", 0.0) + cost,
            "traces": traces,
        }

    return intent_node
