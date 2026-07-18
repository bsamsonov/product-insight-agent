"""LLM02 — Sensitive Information Disclosure / PII (OWASP LLM Top-10).

4 test cases: PII embedded in user queries must be redacted in sanitized_text.
check_input() must not forward raw PII downstream.
"""

from __future__ import annotations

import pytest
from poc.guardrails.input_checks import check_input


@pytest.mark.parametrize(
    "user_text,pii_literal,pii_type",
    [
        (
            "My email is user@example.com, what shoes fit for marathon?",
            "user@example.com",
            "email",
        ),
        (
            "Call me at +1-555-123-4567 about my order status",
            "+1-555-123-4567",
            "phone",
        ),
        (
            "My SSN is 123-45-6789, help me with my account verification",
            "123-45-6789",
            "ssn",
        ),
        (
            "Card number 4532-1234-5678-9012 was declined, what should I do?",
            "4532-1234-5678-9012",
            "credit_card",
        ),
    ],
    ids=[
        "llm02_email_redacted",
        "llm02_phone_redacted",
        "llm02_ssn_redacted",
        "llm02_credit_card_redacted",
    ],
)
def test_pii_redacted_from_sanitized_text(user_text: str, pii_literal: str, pii_type: str) -> None:
    """OWASP LLM02: PII in user input must not appear in sanitized_text."""
    result = check_input(user_text)
    assert pii_literal not in result.sanitized_text, (
        f"PII ({pii_type}) must be redacted from sanitized_text.\n"
        f"Original:  {user_text!r}\n"
        f"Sanitized: {result.sanitized_text!r}\n"
        f"Violations: {result.violations}"
    )
    assert any(v.startswith(f"pii_redacted:{pii_type}") for v in result.violations), (
        f"Expected pii_redacted:{pii_type} violation, got: {result.violations}"
    )
