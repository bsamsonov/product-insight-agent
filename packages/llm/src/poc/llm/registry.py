from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc.llm.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    default_model: str


REGISTRY: dict[str, ProviderConfig] = {
    "gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-flash",
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:14b",
    ),
    "omniroute": ProviderConfig(
        base_url="http://localhost:20128/v1",
        default_model="kr/claude-haiku-4.5",
    ),
}


def get_provider(name: str) -> OpenAICompatibleProvider:
    """Create a provider instance by registry name.

    Convenience factory used by *main.py* and other entry-points that need
    a named provider without going through :class:`~poc.llm.router.RoutedLLM`.

    Args:
        name: Registry key, e.g. ``"gemini"``, ``"groq"``, ``"omniroute"``.

    Returns:
        A ready-to-use :class:`~poc.llm.openai_compatible.OpenAICompatibleProvider`.
    """
    from poc.llm.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider.from_env(name)
