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

    # local OmniRoute gateway
    omniroute_api_key: str = ""
    omniroute_base_url: str | None = None

    # default provider/model used when from_env() is called without arguments
    default_provider: str = "omniroute"
    default_model: str = "kr/claude-haiku-4.5"
