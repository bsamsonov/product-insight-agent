"""Tests for output guardrails — RefusalDetector, schema validation, hallucination signals."""

from __future__ import annotations

import json

from poc.guardrails.output_checks import RefusalDetector, check_output, retry_with_fix_prompt
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# RefusalDetector
# ---------------------------------------------------------------------------


class TestRefusalDetector:
    def setup_method(self) -> None:
        self.detector = RefusalDetector()

    # --- Positive cases: should be detected as refusals ---

    def test_cannot_answer(self) -> None:
        assert self.detector.is_refusal("I cannot answer this question.") is True

    def test_unable_to(self) -> None:
        assert self.detector.is_refusal("I'm unable to provide that information.") is True

    def test_not_able_to(self) -> None:
        assert self.detector.is_refusal("I am not able to help with that.") is True

    def test_as_an_ai(self) -> None:
        assert self.detector.is_refusal("As an AI, I must remain neutral on this topic.") is True

    def test_being_an_ai(self) -> None:
        assert self.detector.is_refusal("Being an AI, I don't have opinions.") is True

    def test_wont(self) -> None:
        assert self.detector.is_refusal("I won't share personal data.") is True

    def test_must_decline(self) -> None:
        assert self.detector.is_refusal("I must decline this request.") is True

    def test_sorry_im_not(self) -> None:
        assert self.detector.is_refusal("Sorry, I'm not able to do that.") is True

    def test_apologies_but(self) -> None:
        assert self.detector.is_refusal("Apologies, but I am not in a position to answer.") is True

    def test_this_request_is(self) -> None:
        assert self.detector.is_refusal("This request is outside my capabilities.") is True

    # --- Negative cases: normal answers that should NOT be detected ---

    def test_normal_answer_not_refusal(self) -> None:
        assert (
            self.detector.is_refusal(
                "The product received excellent reviews with 4.8 stars on average."
            )
            is False
        )

    def test_analytical_answer_not_refusal(self) -> None:
        assert (
            self.detector.is_refusal(
                "Based on the retrieved documents, customers frequently mention battery life."
            )
            is False
        )

    def test_dont_have_info_is_refusal(self) -> None:
        # "I don't have information" matches REFUSAL_PATTERNS
        assert self.detector.is_refusal("I don't have information about that product.") is True

    def test_get_standard_refusal(self) -> None:
        msg = self.detector.get_standard_refusal(trace_id="abc-123")
        assert "trace_id=abc-123" in msg
        assert len(msg) > 20

    def test_get_standard_refusal_no_trace(self) -> None:
        msg = self.detector.get_standard_refusal()
        assert "trace_id" not in msg


# ---------------------------------------------------------------------------
# check_output — schema violation
# ---------------------------------------------------------------------------


class SampleSchema(BaseModel):
    summary: str
    rating: float


def test_schema_violation_not_passed() -> None:
    invalid_json = '{"wrong_key": "value"}'
    result = check_output(invalid_json, expected_schema=SampleSchema)
    assert result.passed is False
    assert any(v.startswith("schema_violation") for v in result.violations)


def test_schema_valid_passes() -> None:
    valid_json = '{"summary": "Great product", "rating": 4.5}'
    result = check_output(valid_json, expected_schema=SampleSchema)
    assert result.passed is True
    assert not any(v.startswith("schema_violation") for v in result.violations)


def test_schema_none_no_violation() -> None:
    result = check_output("Any plain text answer", expected_schema=None)
    assert result.passed is True


# ---------------------------------------------------------------------------
# check_output — hallucination signal (informational, passed stays True)
# ---------------------------------------------------------------------------


def test_hallucination_signal_warning_but_passed() -> None:
    text = "As of my last update, the product was discontinued."
    result = check_output(text)
    assert result.passed is True  # hallucination is a warning, not a blocker
    assert any(v.startswith("hallucination_signal") for v in result.violations)


def test_hallucination_based_on_knowledge() -> None:
    text = "Based on my general knowledge, this product is popular."
    result = check_output(text)
    assert result.passed is True
    assert any(v.startswith("hallucination_signal") for v in result.violations)


def test_no_hallucination_signal_clean_text() -> None:
    result = check_output("The product has a 4.5 star rating based on 200 reviews.")
    assert not any(v.startswith("hallucination_signal") for v in result.violations)


# ---------------------------------------------------------------------------
# check_output — refusal detection (informational, passed stays True)
# ---------------------------------------------------------------------------


def test_refusal_detected_but_passed() -> None:
    """Refusal is a valid LLM outcome — check_output should still pass."""
    result = check_output("I cannot answer this question about pricing.")
    assert result.passed is True  # refusal is valid, not an error
    assert "refusal_detected" in result.violations


def test_normal_output_no_refusal() -> None:
    result = check_output("The average rating for this product category is 4.2 stars.")
    assert "refusal_detected" not in result.violations
    assert result.passed is True


# ---------------------------------------------------------------------------
# retry_with_fix_prompt — AC: on 20 synthetically broken outputs, the
# validator (one retry) repairs >= 15 without human review.
# ---------------------------------------------------------------------------


# 20 synthetically broken LLM outputs that should be coerced into SampleSchema
# {"summary": str, "rating": float} by a single fix-prompt retry.
_BROKEN_OUTPUTS: list[str] = [
    'Sure! Here is the JSON: {"summary": "Great grip", "rating": 4.5}',
    '```json\n{"summary": "Comfortable fit", "rating": 4.0}\n```',
    "{'summary': 'Runs small', 'rating': 3.0}",  # single quotes
    '{"summary": "Durable sole", "rating": "4.2"}',  # rating as string
    '{"summary": "Good value", "rating": 4,}',  # trailing comma
    '{"summary": "Waterproof", "rating": 5.0}  // verified buyer',  # trailing comment
    'Here you go:\n{"summary": "Lightweight", "rating": 4.8}',
    '{"summary": "Breathable mesh", "rating": 4.1}\nLet me know if you need more.',
    '{summary: "Stylish", rating: 3.9}',  # unquoted keys
    '{"summary": "Arch support", "rating": 4.6}',  # already valid but wrapped below
    'The answer is {"summary": "Great traction", "rating": 4.3}.',
    '{"summary": "True to size", "rating": 4.0}```',
    '{"rating": 3.5, "summary": "Average laces"}',  # reordered keys
    '{"summary": "Snug heel", "rating": 4.7, "extra": "ignore"}',  # extra key
    '{"summary": "Quick delivery", "rating": 4.9}\n\n',  # trailing whitespace
    'OUTPUT BELOW\n{"summary": "Soft cushioning", "rating": 4.4}',
    '<json>{"summary": "Wide toe box", "rating": 4.2}</json>',
    "the product is fine, no structured data here at all",  # unfixable: no data
    "ERROR: model timed out before producing output",  # unfixable: no data
    "rating four point five, summary great grip",  # unfixable: prose only
]


def _fake_fixer(broken_text: str):
    """Fake LLM repair callable.

    Simulates a model that, given the malformed output, attempts to emit clean
    JSON. It succeeds when *any* JSON-looking object is recoverable from the
    text; for prose/error outputs with no extractable fields it returns the
    text unchanged (which then fails validation -> the validator returns None).
    """
    import re

    match = re.search(r"\{.*\}", broken_text, re.DOTALL)
    if not match:
        return broken_text  # nothing to fix -> stays invalid

    candidate = match.group(0)
    # Repair common defects a real fix-prompt would handle.
    candidate = candidate.replace("'", '"')  # single -> double quotes
    candidate = re.sub(r",\s*}", "}", candidate)  # trailing comma
    candidate = re.sub(r"//.*", "", candidate)  # line comments
    candidate = re.sub(r"(\{|,)\s*([A-Za-z_]\w*)\s*:", r'\1"\2":', candidate)  # unquoted keys
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return broken_text
    # Coerce rating to float and drop unexpected keys.
    out = {"summary": str(data.get("summary", "")), "rating": float(data.get("rating", 0))}
    return json.dumps(out)


async def test_retry_fix_prompt_repairs_at_least_15_of_20() -> None:
    """AC: validator repairs >= 15 / 20 broken outputs via a single retry."""
    repaired = 0
    for broken in _BROKEN_OUTPUTS:

        async def _llm(messages, *, model, max_tokens, _broken=broken):
            # A real fix-prompt sees the original text inside the user message.
            return _fake_fixer(_broken)

        result = await retry_with_fix_prompt(
            broken, SampleSchema, _llm, model="fake-model", max_tokens=256
        )
        if isinstance(result, SampleSchema):
            repaired += 1

    assert repaired >= 15, f"Expected >=15 repaired, got {repaired}/20"


async def test_retry_fix_prompt_returns_none_when_unfixable() -> None:
    """Single retry that still yields invalid JSON returns None (no recursion)."""

    async def _bad_llm(messages, *, model, max_tokens):
        return "still not json"

    result = await retry_with_fix_prompt(
        "broken", SampleSchema, _bad_llm, model="fake-model", max_tokens=256
    )
    assert result is None


async def test_retry_fix_prompt_single_attempt_only() -> None:
    """The fixer LLM is invoked exactly once (no recursive retries)."""
    calls = 0

    async def _counting_llm(messages, *, model, max_tokens):
        nonlocal calls
        calls += 1
        return "not valid json either"

    await retry_with_fix_prompt(
        "broken", SampleSchema, _counting_llm, model="fake-model", max_tokens=256
    )
    assert calls == 1
