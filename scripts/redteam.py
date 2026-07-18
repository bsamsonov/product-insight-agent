"""Red-team runner: executes all OWASP LLM Top-10 test cases and generates a report.

Usage:
    uv run python scripts/redteam.py

The script runs pytest over tests/security/owasp/ and parses the output to produce
docs/security/redteam-report.md. Does not require pytest-json-report.
"""

from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

SECURITY_TEST_DIR = "tests/security/owasp/"
REPORT_PATH = Path("docs/security/redteam-report.md")

# Map test file prefixes to OWASP categories
_CATEGORY_MAP = {
    "test_llm01": ("LLM01", "Prompt Injection", 5),
    "test_llm02": ("LLM02", "Sensitive Information Disclosure / PII", 4),
    "test_llm05": ("LLM05", "Improper Output Handling", 3),
    "test_llm06": ("LLM06", "Excessive Agency", 3),
    "test_llm08": ("LLM08", "Vector and Embedding Weaknesses", 3),
    "test_llm09": ("LLM09", "Misinformation / Hallucination", 2),
}


def run_redteam() -> int:
    """Run pytest over security tests and return the exit code."""
    print("Running OWASP LLM Top-10 red-team test suite...")
    result = subprocess.run(
        ["uv", "run", "pytest", SECURITY_TEST_DIR, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    generate_report(result.stdout, result.returncode)
    return result.returncode


def _parse_results(stdout: str) -> dict[str, list[tuple[str, str]]]:
    """Parse pytest -v stdout into per-category results.

    Returns dict mapping category_key -> [(test_name, status), ...]
    where status is 'PASSED' | 'FAILED' | 'ERROR'.
    """
    categories: dict[str, list[tuple[str, str]]] = {k: [] for k in _CATEGORY_MAP}
    # Lines like: tests/security/owasp/test_llm01_*.py::test_name PASSED
    line_re = re.compile(r"(test_llm\d+)[^\s]*::(\S+)\s+(PASSED|FAILED|ERROR)")
    for line in stdout.splitlines():
        m = line_re.search(line)
        if m:
            prefix, test_name, status = m.group(1), m.group(2), m.group(3)
            if prefix in categories:
                categories[prefix].append((test_name, status))
    return categories


def _count_summary(stdout: str) -> tuple[int, int]:
    """Extract (passed, total) from pytest summary line."""
    m = re.search(r"(\d+) passed", stdout)
    passed = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+) failed", stdout)
    failed = int(m2.group(1)) if m2 else 0
    m3 = re.search(r"(\d+) error", stdout)
    errors = int(m3.group(1)) if m3 else 0
    total = passed + failed + errors
    return passed, total


def _status_icon(status: str) -> str:
    return "PASS" if status == "PASSED" else "FAIL"


def generate_report(pytest_stdout: str, returncode: int) -> None:
    """Generate docs/security/redteam-report.md from pytest output."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    categories = _parse_results(pytest_stdout)
    passed, total = _count_summary(pytest_stdout)

    lines: list[str] = []
    lines.append("# Red-Team Report — OWASP LLM Top-10\n")
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append("System: Product Insight Agent POC")
    lines.append("Guardrails version: `input_checks.py` + `output_checks.py`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total cases | {total} |")
    lines.append(f"| Passed (correctly handled) | {passed}/{total} |")
    lines.append(f"| Failed | {total - passed}/{total} |")
    lines.append(f"| Overall status | {'PASS' if returncode == 0 else 'FAIL'} |")
    lines.append("")

    for prefix, (cat_id, cat_name, expected_count) in _CATEGORY_MAP.items():
        results = categories.get(prefix, [])
        cat_passed = sum(1 for _, s in results if s == "PASSED")
        lines.append("---")
        lines.append("")
        lines.append(f"## {cat_id} — {cat_name} ({expected_count} cases)")
        lines.append("")
        if results:
            lines.append(f"Result: **{cat_passed}/{len(results)} passed**")
            lines.append("")
            lines.append("| Test | Status |")
            lines.append("|------|--------|")
            for test_name, status in results:
                lines.append(f"| `{test_name}` | {_status_icon(status)} |")
        else:
            lines.append("_No results collected for this category._")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Known Gaps and Production Recommendations")
    lines.append("")
    gaps = [
        ("Shell command injection not detected", "LLM06", "Command-pattern regex"),
        ("SQL injection in output not flagged", "LLM05", "SQL DDL check in output_checks"),
        ("Null bytes not stripped from input", "LLM08", "Null-byte sanitization pre-embed"),
        ("Unicode confusables bypass regex", "LLM08", "NFKC normalization pre-match"),
        ("API DELETE command not detected", "LLM06", "Scope-intent classifier"),
    ]
    lines.append("| Gap | Category | Recommended Fix |")
    lines.append("|-----|----------|-----------------|")
    for gap, cat, fix in gaps:
        lines.append(f"| {gap} | {cat} | {fix} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Generated by `scripts/redteam.py`_")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    raise SystemExit(run_redteam())
