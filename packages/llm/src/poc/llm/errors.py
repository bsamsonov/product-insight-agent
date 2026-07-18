from __future__ import annotations

from poc.core.errors import DomainError


class LLMProviderError(DomainError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMRateLimitError(LLMProviderError):
    pass


class LLMTimeoutError(LLMProviderError):
    pass
