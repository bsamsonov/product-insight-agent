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
    # Optional named provider for any OpenAI-compatible gateway (LiteLLM, OpenRouter,
    # OmniRoute, ...). Not used as a default anywhere — see LLMSettings.default_provider.
    "omniroute": ProviderConfig(
        base_url="http://localhost:20128/v1",
        default_model="gpt-4o-mini",
    ),
}


def default_model_for(provider: str | None = None) -> str:
    """Resolve the default model name for *provider*.

    Single source of truth for "which model if none was passed explicitly": if
    *provider* is omitted, the provider itself is resolved from
    :attr:`~poc.llm.settings.LLMSettings.default_provider` (env
    ``POC_DEFAULT_PROVIDER``, falling back to ``"gemini"``), then the model comes
    from that provider's :class:`ProviderConfig.default_model`. Callers (scripts,
    the agent, the API) should use this instead of hardcoding model names.

    Args:
        provider: Registry key. Defaults to the configured default provider.

    Returns:
        The provider's configured default model name.
    """
    from poc.llm.settings import LLMSettings

    name = provider or LLMSettings().default_provider
    return REGISTRY[name].default_model


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
