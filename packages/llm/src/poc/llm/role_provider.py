"""Role-scoped adapter: LLMProvider facade over RoutedLLM (S3.T2).

Agent nodes are written against the :class:`~poc.llm.provider.LLMProvider`
protocol (``complete(messages, model=..., ...)``), while the router exposes
``route(role, messages, ...)`` and resolves provider+model from router.yaml.

Rewriting every node to know about roles would couple graph logic to routing.
Instead each node receives a ``RoleScopedLLM`` bound to its role — the classic
adapter move: the graph wiring decides "intent speaks as classifier,
summarize speaks as summarizer", and the nodes stay untouched.

The ``model`` argument nodes pass to ``complete()`` is intentionally ignored:
with routing enabled, router.yaml is the single source of truth for models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc.llm.provider import LLMMessage, LLMResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from poc.llm.router import RoutedLLM
    from pydantic import BaseModel


class RoleScopedLLM:
    """LLMProvider-compatible view of a RoutedLLM fixed to a single role."""

    def __init__(self, router: RoutedLLM, role: str) -> None:
        self._router = router
        self._role = role

    @property
    def role(self) -> str:
        return self._role

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,  # ignored: router.yaml owns model selection per role
        max_tokens: int,  # ignored: router.yaml owns max_tokens per role
        temperature: float = 0.0,
        response_format: type[BaseModel] | None = None,
        cache_control: bool = False,  # reserved for AnthropicProvider
    ) -> LLMResponse:
        return await self._router.route(
            self._role,
            messages,
            response_format=response_format,
            temperature=temperature,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError(
            "Streaming is not supported through the role router; "
            "use a direct provider for streaming endpoints."
        )
