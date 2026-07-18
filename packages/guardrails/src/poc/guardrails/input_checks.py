from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


@dataclass
class InputCheckResult:
    passed: bool
    violations: list[str]
    sanitized_text: str


# PII patterns: email, phone, SSN-like, credit card patterns
_PII_PATTERNS: list[tuple[str, str, str]] = [
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
    ("phone", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    ("ssn", r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "[SSN]"),
    ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD]"),
]

# Prompt injection signatures (common attack patterns)
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+\w+\s+ai", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(?:prior|previous|above)\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(?:you|i)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if|though)\s+you", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"DAN\b", re.IGNORECASE),
]

_MAX_INPUT_LENGTH = 4000  # characters


class LengthGuard:
    """Token-aware length guard using tiktoken (falls back to len//4 estimate)."""

    def __init__(self, max_tokens: int = 8000, encoding: str = "cl100k_base") -> None:
        self.max_tokens = max_tokens
        self._encoding_name = encoding
        self._enc = None
        if _HAS_TIKTOKEN:
            with contextlib.suppress(Exception):
                self._enc = tiktoken.get_encoding(encoding)

    def _count_tokens(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Regex fallback: GPT-family average ~4 chars per token
        return len(text) // 4

    def check(self, text: str) -> tuple[bool, int]:
        """Return (is_within_limit, token_count)."""
        count = self._count_tokens(text)
        return count <= self.max_tokens, count


# Module-level singleton — created lazily to avoid import-time cost
_default_length_guard: LengthGuard | None = None


def _get_length_guard(max_tokens: int) -> LengthGuard:
    global _default_length_guard
    if _default_length_guard is None or _default_length_guard.max_tokens != max_tokens:
        _default_length_guard = LengthGuard(max_tokens=max_tokens)
    return _default_length_guard


def check_input(
    text: str,
    *,
    max_length: int = _MAX_INPUT_LENGTH,  # characters (backward compat)
    max_tokens: int = 8000,
) -> InputCheckResult:
    """Check and sanitize user input text.

    Checks:
    - Length limit (token-based via LengthGuard; character limit kept for backward compat)
    - Prompt injection patterns
    - PII redaction

    Returns InputCheckResult with sanitized text.
    Violations are informational; callers decide whether to reject.
    """
    violations: list[str] = []
    sanitized = text

    # Token-based length check (primary)
    guard = _get_length_guard(max_tokens)
    within_limit, token_count = guard.check(text)
    if not within_limit:
        violations.append(f"input_too_long_tokens:{token_count}")
        # Truncate by characters as a safe approximation
        char_limit = max_tokens * 4
        sanitized = sanitized[:char_limit]
    elif len(text) > max_length:
        # Character limit fallback (backward compat)
        violations.append(f"input_too_long:{len(text)}")
        sanitized = sanitized[:max_length]

    # Prompt injection check
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            violations.append(f"prompt_injection:{pattern.pattern[:40]}")

    # PII redaction
    for name, pattern, replacement in _PII_PATTERNS:
        new_text, count = re.subn(pattern, replacement, sanitized)
        if count > 0:
            violations.append(f"pii_redacted:{name}:{count}")
            sanitized = new_text

    passed = not any(v.startswith("prompt_injection") for v in violations)
    return InputCheckResult(passed=passed, violations=violations, sanitized_text=sanitized)
