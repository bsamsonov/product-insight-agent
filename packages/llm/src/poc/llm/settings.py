from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Per-provider API keys and base URLs.

    Env vars follow pattern POC_<PROVIDER>_API_KEY / POC_<PROVIDER>_BASE_URL.
    pydantic-settings maps field `gemini_api_key` → env var `POC_GEMINI_API_KEY`.
    """

    model_config = SettingsConfigDict(env_prefix="POC_", env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_base_url: str | None = None

    groq_api_key: str = ""
    groq_base_url: str | None = None

    deepseek_api_key: str = ""
    deepseek_base_url: str | None = None

    # ollama doesn't require a real key
    ollama_api_key: str = "ollama"
    ollama_base_url: str | None = None

    # Optional: any OpenAI-compatible gateway (LiteLLM, OpenRouter, OmniRoute, ...)
    omniroute_api_key: str = ""
    omniroute_base_url: str | None = None

    # Default provider used when from_env() is called without arguments. This is the
    # single source of truth for "which provider if none is specified" — resolved from
    # env var POC_DEFAULT_PROVIDER, falling back to "gemini" (matches the README
    # quickstart, which only requires a free-tier POC_GEMINI_API_KEY).
    default_provider: str = "gemini"
