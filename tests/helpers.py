"""Shared test helpers for farm-RAG tests."""

from __future__ import annotations

from typing import Any


def tally_checkbox_field(label: str, *texts: str) -> dict[str, Any]:
    """Build a Tally-style CHECKBOXES multi-select field payload.

    Matches the field structure used by Tally for checkbox groups,
    where multiple options exist and selected options are identified by IDs.
    """
    options = [{"id": f"opt_{i}", "text": t} for i, t in enumerate(texts)]
    return {
        "key": label.lower().replace(" ", "_"),
        "label": label,
        "type": "CHECKBOXES",
        "value": [o["id"] for o in options],
        "options": options,
    }


def tally_text_field(label: str, value: str) -> dict[str, Any]:
    """Build a Tally-style INPUT_TEXT field payload."""
    return {
        "key": label.lower().replace(" ", "_"),
        "label": label,
        "type": "INPUT_TEXT",
        "value": value,
    }


def result_rule_ids(result) -> list[str]:
    """Sorted rule IDs from an exemption result."""
    return sorted(r.rule_id for r in result.matched_rules)


def assert_verdict(result, expected_verdict: str) -> None:
    """Assert the exemption result has the expected verdict."""
    assert result.verdict == expected_verdict, (
        f"Expected {expected_verdict}, got {result.verdict}. "
        f"Matched: {[r.rule_id for r in result.matched_rules]}"
    )


def assert_exempt(result, expected_rules: list[str] | None = None) -> None:
    """Assert the exemption result is exempt.

    Optionally verify that the expected rule IDs were matched.
    """
    assert_verdict(result, "exempt")
    if expected_rules is not None:
        actual = result_rule_ids(result)
        assert actual == sorted(expected_rules), (
            f"Mismatched rules: {actual} vs {sorted(expected_rules)}"
        )


def assert_partial(result, expected_rules: list[str] | None = None) -> None:
    """Assert the exemption result is partially_exempt.

    Optionally verify that the expected rule IDs were matched.
    """
    assert_verdict(result, "partially_exempt")
    if expected_rules is not None:
        actual = result_rule_ids(result)
        assert actual == sorted(expected_rules), (
            f"Mismatched rules: {actual} vs {sorted(expected_rules)}"
        )


def assert_not_exempt(result) -> None:
    """Assert the exemption result is not_exempt with zero matched rules."""
    assert_verdict(result, "not_exempt")
    assert len(result.matched_rules) == 0, (
        f"Expected no matched rules for not_exempt, got: "
        f"{[r.rule_id for r in result.matched_rules]}"
    )


def assert_needs_review(result) -> None:
    """Assert the exemption result is needs_review."""
    assert_verdict(result, "needs_review")
