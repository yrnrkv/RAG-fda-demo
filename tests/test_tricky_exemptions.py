"""Tricky exemption scenarios — double whammies, edge cases, and rule interactions.

These tests go beyond the 15 canonical shared scenarios to exercise rule *combinations*:
when two or more exemptions fire simultaneously, the engine must resolve the verdict
correctly. Priority: full > partial > needs_review > not_exempt.

Also includes Tally webhook integration tests with tricky mock payloads (revenue band
boundaries, channel combinations, NSSP shellfish, farm-to-school flags).

Last updated: 2026-06-21
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fda_traceability_rag.exemption import (
    PRODUCE_FARM_AVG_USD,
    SMALL_RFE_MAX_FTE_EMPLOYEES,
    SurveyAnswers,
    evaluate_exemption,
)
from service.tally_mapping import map_submission, revenue_band_max
from tests.helpers import (
    assert_exempt,
    assert_needs_review,
    assert_not_exempt,
    assert_partial,
    tally_checkbox_field,
    tally_text_field,
)


# ==============================================================================
# Helpers for building SurveyAnswers
# ==============================================================================

def _a(**overrides) -> SurveyAnswers:
    """Factory: produce a SurveyAnswers with sensible defaults overridden."""
    defaults = {
        "entity_type": "produce_farm",
        "handles_ftl_food": "yes",
        "ftl_categories": ["fresh tomatoes"],
        "produce_3yr_avg_usd": None,
        "sales_channels": ["wholesale_distribution"],
        "rfe_full_time_equiv_employees": None,
        "downstream_kill_step": False,
        "rarely_consumed_raw_variety": False,
        "shellfish_under_nssp": False,
        "farm_to_school_program": False,
    }
    defaults.update(overrides)
    return SurveyAnswers(**defaults)


# ==============================================================================
# DOUBLE FULL EXEMPTIONS — two full rules fire, verdict is exempt
# ==============================================================================

def test_double_full_small_produce_farm_plus_direct_to_consumer():
    """$20k produce farm selling direct-to-consumer only.
    Both small_produce_farm (§1.1305(a)(1)(ii)) AND direct_to_consumer (§1.1305(b))
    fire. Verdict: exempt (full trumps everything).
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=20_000,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["direct_to_consumer", "small_produce_farm"])


def test_double_full_produce_farm_at_threshold_plus_direct():
    """$25,000 (exactly at threshold, inclusive) + direct-to-consumer.
    Both rules still fire — the <= boundary is inclusive.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=25_000,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["direct_to_consumer", "small_produce_farm"])


def test_double_full_fishing_vessel_plus_nssp_shellfish():
    """Fishing vessel owner handling NSSP-covered bivalve molluscan shellfish.
    Both fishing_vessel (§1.1305(m)) AND nssp_shellfish (§1.1305(f)) fire.
    """
    r = evaluate_exemption(_a(
        entity_type="fishing_vessel",
        ftl_categories=["molluscan shellfish"],
        shellfish_under_nssp=True,
    ))
    assert_exempt(r, ["fishing_vessel", "nssp_shellfish"])


def test_double_full_direct_consumer_plus_rarely_consumed_raw():
    """Farm selling direct AND growing a rarely-consumed-raw variety.
    direct_to_consumer (§1.1305(b)) + rarely_consumed_raw (§1.1305(e)).
    """
    r = evaluate_exemption(_a(
        sales_channels=["direct_consumer"],
        rarely_consumed_raw_variety=True,
    ))
    assert_exempt(r, ["direct_to_consumer", "rarely_consumed_raw"])


def test_double_full_not_on_ftl_plus_small_rfe():
    """RFE with 8 employees who ALSO doesn't handle FTL foods.
    Both not_on_ftl + small_rfe fire. Verdict: exempt.
    """
    r = evaluate_exemption(_a(
        entity_type="rfe",
        handles_ftl_food="no",
        ftl_categories=[],
        rfe_full_time_equiv_employees=8,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["not_on_ftl", "small_rfe"])


def test_double_full_nssp_plus_rarely_consumed_raw_on_distributor():
    """A distributor handling NSSP shellfish AND a rarely-consumed-raw FTL food.
    Both nssp_shellfish + rarely_consumed_raw fire, independent of entity type.
    """
    r = evaluate_exemption(_a(
        entity_type="distributor",
        ftl_categories=["molluscan shellfish", "fresh-cut vegetables"],
        shellfish_under_nssp=True,
        rarely_consumed_raw_variety=True,
    ))
    assert_exempt(r, ["nssp_shellfish", "rarely_consumed_raw"])


def test_double_full_small_rfe_plus_downstream_kill_step():
    """Small RFE (8 FTE) whose food also undergoes a downstream kill step.
    small_rfe (full) + downstream_kill_step (partial). Full trumps → exempt,
    but BOTH rules still appear in matched_rules (the engine reports all
    matching exemptions, even when full overrides partial on the verdict).
    """
    r = evaluate_exemption(_a(
        entity_type="rfe",
        rfe_full_time_equiv_employees=8,
        downstream_kill_step=True,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["downstream_kill_step", "small_rfe"])


# ==============================================================================
# FULL + PARTIAL — full always takes the verdict
# ==============================================================================

def test_full_plus_partial_small_produce_farm_plus_farm_to_school():
    """Small $20k produce farm that also participates in farm-to-school.
    small_produce_farm (full) + farm_to_school (partial). → exempt.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=20_000,
        farm_to_school_program=True,
    ))
    assert_exempt(r, ["small_produce_farm", "farm_to_school"])


def test_full_plus_partial_direct_to_consumer_plus_farm_to_school():
    """Direct-to-consumer farm that also does farm-to-school.
    direct_to_consumer (full) + farm_to_school (partial). → exempt.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=800_000,  # well above $25k, so only direct_to_consumer fires as full
        sales_channels=["direct_consumer"],
        farm_to_school_program=True,
    ))
    assert_exempt(r, ["direct_to_consumer", "farm_to_school"])


def test_full_plus_partial_not_on_ftl_plus_downstream_kill_step():
    """Distributor not handling FTL foods but with downstream kill step.
    not_on_ftl (full) + downstream_kill_step (partial). → exempt.
    """
    r = evaluate_exemption(_a(
        entity_type="distributor",
        handles_ftl_food="no",
        ftl_categories=[],
        downstream_kill_step=True,
    ))
    assert_exempt(r, ["not_on_ftl", "downstream_kill_step"])


# ==============================================================================
# TRIPLE WHAMMY — three rules fire simultaneously
# ==============================================================================

def test_triple_small_farm_direct_consumer_farm_to_school():
    """$20k produce farm selling direct AND in farm-to-school.
    small_produce_farm + direct_to_consumer (both full) + farm_to_school (partial).
    → exempt, ALL THREE rules in matched_rules.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=20_000,
        sales_channels=["direct_consumer"],
        farm_to_school_program=True,
    ))
    assert_exempt(r, ["direct_to_consumer", "small_produce_farm", "farm_to_school"])


def test_triple_at_threshold_direct_farm_to_school():
    """$25,000 (exact threshold) + direct + farm-to-school. All three fire.
    Verdict: exempt.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=25_000,
        sales_channels=["direct_consumer"],
        farm_to_school_program=True,
    ))
    assert_exempt(r, ["direct_to_consumer", "small_produce_farm", "farm_to_school"])


# ==============================================================================
# DOUBLE PARTIAL — two partials, verdict stays partially_exempt
# ==============================================================================

def test_double_partial_downstream_kill_plus_farm_to_school():
    """Manufacturer with downstream kill step AND farm-to-school.
    Both are partial → partially_exempt with 2 rules.
    """
    r = evaluate_exemption(_a(
        entity_type="manufacturer_processor",
        downstream_kill_step=True,
        farm_to_school_program=True,
    ))
    assert_partial(r, ["downstream_kill_step", "farm_to_school"])


# ==============================================================================
# PARTIAL TRUMPS NEEDS_REVIEW — unsure about FTL but partial exemptions apply
# ==============================================================================

def test_partial_trumps_needs_review_kill_step():
    """Unsure about FTL, but has a downstream kill step.
    Partial exemption wins over needs_review — the kill step exemption applies
    regardless of FTL uncertainty (it modifies obligations even if on FTL).
    """
    r = evaluate_exemption(_a(
        handles_ftl_food="unsure",
        ftl_categories=[],
        downstream_kill_step=True,
    ))
    assert_partial(r, ["downstream_kill_step"])


def test_partial_trumps_needs_review_farm_to_school():
    """Unsure about FTL, but farm-to-school program applies.
    Same logic: partial > needs_review.
    """
    r = evaluate_exemption(_a(
        handles_ftl_food="unsure",
        ftl_categories=[],
        farm_to_school_program=True,
    ))
    assert_partial(r, ["farm_to_school"])


def test_partial_trumps_needs_review_both_partials():
    """Unsure FTL + BOTH partial exemptions. → partially_exempt, 2 rules."""
    r = evaluate_exemption(_a(
        handles_ftl_food="unsure",
        ftl_categories=[],
        downstream_kill_step=True,
        farm_to_school_program=True,
    ))
    assert_partial(r, ["downstream_kill_step", "farm_to_school"])


# ==============================================================================
# BOUNDARY & EDGE CASES
# ==============================================================================

def test_produce_1_dollar_above_threshold_no_exemption():
    """$25,001 produce farm (wholesale only, no other flags).
    Just $1 above threshold → small_produce_farm does NOT fire. → not_exempt.
    """
    r = evaluate_exemption(_a(produce_3yr_avg_usd=25_001))
    assert_not_exempt(r)


def test_produce_1_dollar_above_threshold_but_direct():
    """$25,001 produce farm, but direct-to-consumer.
    small_produce_farm does NOT fire (above threshold), but direct_to_consumer DOES.
    → exempt (1 rule, not 2).
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=25_001,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["direct_to_consumer"])


def test_rfe_at_boundary_10_employees():
    """RFE with exactly 10 FTE employees — boundary is inclusive (<=).
    small_rfe fires. → exempt.
    """
    r = evaluate_exemption(_a(
        entity_type="rfe",
        rfe_full_time_equiv_employees=10,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["small_rfe"])


def test_rfe_11_employees_no_exemption():
    """RFE with 11 FTE employees — just above the 10-employee cutoff.
    small_rfe does NOT fire. → not_exempt.
    """
    r = evaluate_exemption(_a(
        entity_type="rfe",
        rfe_full_time_equiv_employees=11,
        sales_channels=["direct_consumer"],
    ))
    assert_not_exempt(r)


def test_rfe_11_employees_but_not_on_ftl():
    """RFE with 11 FTE but doesn't handle FTL foods.
    small_rfe does NOT fire, but not_on_ftl DOES. → exempt (1 rule).
    """
    r = evaluate_exemption(_a(
        entity_type="rfe",
        handles_ftl_food="no",
        ftl_categories=[],
        rfe_full_time_equiv_employees=11,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["not_on_ftl"])


def test_zero_dollar_produce_farm():
    """Produce farm with $0 produce value — should still qualify as small.
    The <= boundary includes 0.
    """
    r = evaluate_exemption(_a(produce_3yr_avg_usd=0))
    assert_exempt(r, ["small_produce_farm"])


def test_produce_avg_usd_none_no_small_farm_rule():
    """Produce farm with produce_3yr_avg_usd=None (not reported).
    The small_produce_farm predicate requires a number — absent → rule doesn't fire.
    """
    r = evaluate_exemption(_a(produce_3yr_avg_usd=None))
    assert_not_exempt(r)


# ==============================================================================
# CHANNEL EDGE CASES
# ==============================================================================

def test_mixed_channels_no_direct_consumer_exemption():
    """Farm selling through BOTH wholesale AND direct-to-consumer.
    direct_to_consumer rule requires ALL channels be direct_consumer → doesn't fire.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=500_000,
        sales_channels=["direct_consumer", "wholesale_distribution"],
    ))
    assert_not_exempt(r)


def test_empty_channels_no_direct_exemption():
    """Farm with empty sales_channels.
    direct_to_consumer requires len > 0 → doesn't fire.
    """
    r = evaluate_exemption(_a(
        produce_3yr_avg_usd=500_000,
        sales_channels=[],
    ))
    assert_not_exempt(r)


def test_other_farm_direct_to_consumer():
    """'other_farm' entity type is also a farm → direct_to_consumer fires."""
    r = evaluate_exemption(_a(
        entity_type="other_farm",
        produce_3yr_avg_usd=500_000,
        sales_channels=["direct_consumer"],
    ))
    assert_exempt(r, ["direct_to_consumer"])


# ==============================================================================
# ENTITY TYPE RULE AFFINITY
# ==============================================================================

def test_distributor_no_farm_rules():
    """Distributor cannot use farm-specific exemptions (small_produce_farm, direct_to_consumer).
    But can use entity-agnostic ones (nssp_shellfish, rarely_consumed_raw, not_on_ftl, etc.).
    """
    r = evaluate_exemption(_a(
        entity_type="distributor",
        produce_3yr_avg_usd=20_000,  # low, but irrelevant — not a farm
        sales_channels=["direct_consumer"],  # irrelevant — not a farm
    ))
    # small_produce_farm and direct_to_consumer should NOT fire because entity_type != farm
    assert_not_exempt(r)


def test_manufacturer_can_use_partial_exemptions():
    """Manufacturer/processor can use downstream_kill_step, rarely_consumed_raw, etc.
    But not farm-specific rules.
    """
    r = evaluate_exemption(_a(
        entity_type="manufacturer_processor",
        downstream_kill_step=True,
    ))
    assert_partial(r, ["downstream_kill_step"])


# ==============================================================================
# DETERMINISM
# ==============================================================================

def test_all_tricky_scenarios_are_deterministic():
    """Run each tricky scenario 5x — results must be identical every time."""
    scenarios = [
        _a(produce_3yr_avg_usd=20_000, sales_channels=["direct_consumer"]),
        _a(entity_type="fishing_vessel", ftl_categories=["molluscan shellfish"], shellfish_under_nssp=True),
        _a(produce_3yr_avg_usd=20_000, sales_channels=["direct_consumer"], farm_to_school_program=True),
        _a(handles_ftl_food="unsure", ftl_categories=[], downstream_kill_step=True, farm_to_school_program=True),
        _a(entity_type="rfe", handles_ftl_food="no", ftl_categories=[], rfe_full_time_equiv_employees=8),
    ]
    for answers in scenarios:
        first = evaluate_exemption(answers).to_dict()
        for _ in range(4):
            assert evaluate_exemption(answers).to_dict() == first


# ==============================================================================
# THRESHOLD VALIDATION
# ==============================================================================

def test_threshold_values_vs_shared_fixture():
    """Ensure our Python constants match the shared canonical fixture."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).resolve().parents[1] / "shared" / "exemption-scenarios.json"
    thresholds = json.loads(fixture_path.read_text())["thresholds"]

    assert PRODUCE_FARM_AVG_USD == thresholds["produce_farm_avg_usd"], (
        f"PRODUCE_FARM_AVG_USD ({PRODUCE_FARM_AVG_USD}) != fixture ({thresholds['produce_farm_avg_usd']})"
    )
    assert SMALL_RFE_MAX_FTE_EMPLOYEES == thresholds["small_rfe_max_fte_employees"]


# ==============================================================================
# TALLY WEBHOOK INTEGRATION — tricky payloads through the full pipeline
# ==============================================================================

# ---------- tricky Tally payloads ----------
TRICKY_TALLY_SMALL_DIRECT = {
    "submissionId": "TRICKY-001",
    "fields": [
        tally_text_field("First Name", "Alice"),
        tally_text_field("Last Name", "SmallFarm"),
        tally_text_field("Best email to reach you", "alice@smallfarm.example"),
        tally_text_field("Phone number", "555-0100"),
        tally_text_field("State of your main farm", "Vermont"),
        tally_text_field("Annual Farm Revenue", "Under $25,000"),
        tally_checkbox_field("Main Crop Type (Select all that apply)", "Tomatoes", "Cucumbers"),
        tally_checkbox_field(
            "Distribution Channel (Select all that apply)",
            "Direct to consumers (farmers markets, CSA, farm stand, online)",
        ),
    ],
}

TRICKY_TALLY_BOUNDARY_BAND = {
    "submissionId": "TRICKY-002",
    "fields": [
        tally_text_field("First Name", "Bob"),
        tally_text_field("Last Name", "Boundary"),
        tally_text_field("Best email to reach you", "bob@boundary.example"),
        tally_text_field("Phone number", "555-0101"),
        tally_text_field("State of your main farm", "Oregon"),
        # "Under $25,000" max = $25,000, inclusive → small produce farm fires
        tally_text_field("Annual Farm Revenue", "Under $25,000"),
        tally_checkbox_field("Main Crop Type (Select all that apply)", "Leafy greens"),
        tally_checkbox_field(
            "Distribution Channel (Select all that apply)",
            "Wholesalers or distributors",
        ),
    ],
}

TRICKY_TALLY_MIXED_CHANNEL_BIG = {
    "submissionId": "TRICKY-003",
    "fields": [
        tally_text_field("First Name", "Carol"),
        tally_text_field("Last Name", "MidSize"),
        tally_text_field("Best email to reach you", "carol@midsize.example"),
        tally_text_field("Phone number", "555-0102"),
        tally_text_field("State of your main farm", "California"),
        # $1M-$5M band → max $5M → far above $25k
        tally_text_field("Annual Farm Revenue", "$1M – $5M"),
        tally_checkbox_field("Main Crop Type (Select all that apply)", "Leafy greens (fresh-cut)", "Herbs"),
        tally_checkbox_field(
            "Distribution Channel (Select all that apply)",
            "Wholesalers or distributors",
            "Direct to consumers (farmers markets, CSA, farm stand, online)",
        ),
    ],
}

TRICKY_TALLY_SHELLFISH_NSSP = {
    "submissionId": "TRICKY-004",
    "fields": [
        tally_text_field("First Name", "Dave"),
        tally_text_field("Last Name", "OysterFarm"),
        tally_text_field("Best email to reach you", "dave@shellfish.example"),
        tally_text_field("Phone number", "555-0103"),
        tally_text_field("State of your main farm", "Washington"),
        tally_text_field("Annual Farm Revenue", "Under $25,000"),
        tally_checkbox_field("Main Crop Type (Select all that apply)", "Molluscan shellfish (oysters, clams, mussels)"),
        tally_checkbox_field(
            "Distribution Channel (Select all that apply)",
            "Wholesalers or distributors",
        ),
    ],
}


def test_tricky_webhook_small_direct_to_consumer():
    """Tally payload: $25k-or-under farm, tomatoes, direct-to-consumer only.
    small_produce_farm + direct_to_consumer → exempt.
    """
    import service.app as app_module

    app_module.TALLY_SIGNING_SECRET = ""  # explicit opt-out for test
    client = TestClient(app_module.app)

    body = json.dumps({"data": TRICKY_TALLY_SMALL_DIRECT}).encode()
    resp = client.post("/webhooks/tally", content=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["verdict"] == "exempt"
    assert sorted(payload["mappedAnswers"]["salesChannels"]) == ["direct_consumer"]
    # small_produce_farm fires because rev_max = 25000 <= 25000
    rule_ids = [r["ruleId"] for r in payload["result"]["matchedRules"]]
    assert "direct_to_consumer" in rule_ids
    assert "small_produce_farm" in rule_ids


def test_tricky_webhook_boundary_band():
    """Tally payload: Under $25,000 band, leafy greens, wholesale only.
    Revenue band max = 25,000 → exactly at threshold → small_produce_farm fires.
    → exempt.
    """
    import service.app as app_module

    app_module.TALLY_SIGNING_SECRET = ""
    client = TestClient(app_module.app)

    body = json.dumps({"data": TRICKY_TALLY_BOUNDARY_BAND}).encode()
    resp = client.post("/webhooks/tally", content=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["verdict"] == "exempt"
    rule_ids = [r["ruleId"] for r in payload["result"]["matchedRules"]]
    assert "small_produce_farm" in rule_ids
    assert payload["leadRow"]["verdict"] == "exempt"
    assert payload["leadRow"]["email"] == "bob@boundary.example"


def test_tricky_webhook_mixed_channel_big_farm():
    """Tally payload: $1M-$5M farm, leafy greens, mixed channels.
    Revenue band max = 5,000,000 → far above $25k.
    Mixed channels → direct_to_consumer doesn't fire (not ALL direct).
    → not_exempt.
    """
    import service.app as app_module

    app_module.TALLY_SIGNING_SECRET = ""
    client = TestClient(app_module.app)

    body = json.dumps({"data": TRICKY_TALLY_MIXED_CHANNEL_BIG}).encode()
    resp = client.post("/webhooks/tally", content=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["verdict"] == "not_exempt"
    assert payload["leadRow"]["verdict"] == "not_exempt"
    assert "Carol" in payload["customerEmail"]["body"]


def test_tricky_webhook_shellfish_nssp():
    """Tally payload: small shellfish farm handling molluscan shellfish.
    The mapper detects 'molluscan_shellfish' in ftlCategories → sets
    shellfishUnderNssp=True. Small revenue + NSSP shellfish → double full
    exemption (small_produce_farm + nssp_shellfish).
    → exempt.
    """
    import service.app as app_module

    app_module.TALLY_SIGNING_SECRET = ""
    client = TestClient(app_module.app)

    body = json.dumps({"data": TRICKY_TALLY_SHELLFISH_NSSP}).encode()
    resp = client.post("/webhooks/tally", content=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["verdict"] == "exempt"
    rule_ids = [r["ruleId"] for r in payload["result"]["matchedRules"]]
    assert "nssp_shellfish" in rule_ids
    assert "small_produce_farm" in rule_ids
    assert payload["mappedAnswers"]["shellfishUnderNssp"] is True


def test_tricky_webhook_slack_and_email_outputs():
    """Verify the full webhook pipeline produces correct Slack + email + lead row."""
    import service.app as app_module

    app_module.TALLY_SIGNING_SECRET = ""
    client = TestClient(app_module.app)

    body = json.dumps({"data": TRICKY_TALLY_SMALL_DIRECT}).encode()
    resp = client.post("/webhooks/tally", content=body)
    assert resp.status_code == 200
    payload = resp.json()

    # Customer email
    assert "Hi Alice" in payload["customerEmail"]["body"]
    assert "exempt from FSMA 204" in payload["customerEmail"]["subject"]
    assert "not legal advice" in payload["customerEmail"]["body"]

    # Lead row
    assert payload["leadRow"]["firstName"] == "Alice"
    assert payload["leadRow"]["verdict"] == "exempt"
    assert "Tomatoes" in payload["leadRow"]["crops"]
    # Lead row channels: raw Tally labels (not mapped), so use case-insensitive check
    assert "direct to consumers" in payload["leadRow"]["channels"].lower()

    # Slack text
    assert "New FSMA 204 lead" in payload["slackText"]
    assert "Alice SmallFarm" in payload["slackText"]
    assert "exempt" in payload["slackText"]


# ==============================================================================
# M15 VERIFICATION — TALLY_SIGNING_SECRET behavior (fixed)
# ==============================================================================

def test_webhook_refuses_when_secret_is_none():
    """M15: When TALLY_SIGNING_SECRET is None (unset), the webhook MUST refuse.
    The old behavior silently skipped verification; now it's a 500 error.
    """
    import service.app as app_module

    original = app_module.TALLY_SIGNING_SECRET
    try:
        app_module.TALLY_SIGNING_SECRET = None
        client = TestClient(app_module.app)
        resp = client.post("/webhooks/tally", content=b"{}")
        assert resp.status_code == 500
        assert "not configured" in resp.json()["detail"].lower()
    finally:
        app_module.TALLY_SIGNING_SECRET = original


def test_webhook_accepts_when_secret_is_empty_string():
    """M15: When TALLY_SIGNING_SECRET is '' (explicit opt-out), the webhook
    accepts requests without signature verification (trusted-network mode).
    """
    import service.app as app_module

    original = app_module.TALLY_SIGNING_SECRET
    try:
        app_module.TALLY_SIGNING_SECRET = ""
        client = TestClient(app_module.app)
        resp = client.post("/webhooks/tally", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
    finally:
        app_module.TALLY_SIGNING_SECRET = original


def test_webhook_rejects_invalid_signature_with_secret_set():
    """M15: When a real secret is set, invalid signatures are rejected (401)."""
    import base64
    import hashlib
    import hmac

    import service.app as app_module

    original = app_module.TALLY_SIGNING_SECRET
    try:
        app_module.TALLY_SIGNING_SECRET = "correct-secret"
        client = TestClient(app_module.app)

        # Valid signature
        body = json.dumps({"data": TRICKY_TALLY_SMALL_DIRECT}).encode()
        sig = base64.b64encode(hmac.new(b"correct-secret", body, hashlib.sha256).digest()).decode()
        resp = client.post("/webhooks/tally", content=body, headers={"tally-signature": sig})
        assert resp.status_code == 200

        # Bad signature
        resp_bad = client.post("/webhooks/tally", content=body, headers={"tally-signature": "wrong"})
        assert resp_bad.status_code == 401

        # Missing signature
        resp_missing = client.post("/webhooks/tally", content=body)
        assert resp_missing.status_code == 401
    finally:
        app_module.TALLY_SIGNING_SECRET = original


# ==============================================================================
# M16 VERIFICATION — CORS origin behavior (fixed)
# ==============================================================================

def test_cors_with_no_origins_configured():
    """M16: When ALLOWED_ORIGINS is empty, CORS middleware denies all origins.
    There should be no 'access-control-allow-origin' header for a disallowed origin.
    """
    import service.app as app_module

    original = list(app_module._ALLOWED_ORIGINS)
    try:
        app_module._ALLOWED_ORIGINS.clear()
        client = TestClient(app_module.app)
        resp = client.get("/health", headers={"Origin": "https://evil.example"})
        # When allow_origins is empty, CORSMiddleware does NOT echo back the Origin.
        acao = resp.headers.get("access-control-allow-origin")
        assert acao != "https://evil.example", (
            f"CORS should NOT echo back disallowed origin, got {acao}"
        )
    finally:
        app_module._ALLOWED_ORIGINS.clear()
        app_module._ALLOWED_ORIGINS.extend(original)


def test_cors_with_specific_origin():
    """M16: When ALLOWED_ORIGINS is set explicitly, only that origin gets through."""
    import service.app as app_module

    original = list(app_module._ALLOWED_ORIGINS)
    try:
        app_module._ALLOWED_ORIGINS.clear()
        app_module._ALLOWED_ORIGINS.extend(["https://farmersfront.com"])
        client = TestClient(app_module.app)

        # Allowed origin
        resp = client.get("/health", headers={"Origin": "https://farmersfront.com"})
        acao = resp.headers.get("access-control-allow-origin")
        assert acao == "https://farmersfront.com"

        # Disallowed origin
        resp2 = client.get("/health", headers={"Origin": "https://evil.example"})
        acao2 = resp2.headers.get("access-control-allow-origin")
        assert acao2 != "https://evil.example"
    finally:
        app_module._ALLOWED_ORIGINS.clear()
        app_module._ALLOWED_ORIGINS.extend(original)


# ==============================================================================
# M24 VERIFICATION — User-Agent no longer impersonates FDA
# ==============================================================================

def test_user_agent_points_to_repo():
    """M24: The User-Agent in sources.py must reference the actual project,
    not fda.gov (which implied FDA operated the bot).
    """
    from fda_traceability_rag.sources import _get as sources_get

    # We can test the constant directly
    import inspect

    source = inspect.getsource(sources_get)
    assert "+https://www.fda.gov/" not in source, (
        "User-Agent still contains fda.gov URL — it should reference the actual project"
    )
    assert "+https://github.com/CrossCraftAI/farm-RAG" in source


# ==============================================================================
# REVENUE BAND PARSING CORNER CASES
# ==============================================================================

def test_revenue_band_exactly_25000():
    """Revenue band 'Under $25,000' → max = 25,000."""
    assert revenue_band_max("Under $25,000") == 25_000


def test_revenue_band_25001_to_50000():
    """Revenue band '$25,001 – $50,000' → max = 50,000 → > threshold."""
    r = revenue_band_max("$25,001 – $50,000")
    assert r == 50_000
    assert r > PRODUCE_FARM_AVG_USD


def test_revenue_band_open_ended_5m_plus():
    """Revenue band '$5M+' → None (open-ended, no finite max)."""
    assert revenue_band_max("$5M+") is None


def test_revenue_band_single_value():
    """Revenue band '$10,000' → max is 10,000 (no range)."""
    assert revenue_band_max("$10,000") == 10_000


def test_revenue_band_none_or_empty():
    """Revenue band None or empty string → None."""
    assert revenue_band_max(None) is None
    assert revenue_band_max("") is None


# ==============================================================================
# FTL CATEGORY LOOKUP REGRESSION
# ==============================================================================

def test_molluscan_shellfish_detected():
    """Verify the FTL lookup correctly identifies molluscan shellfish variants."""
    from fda_traceability_rag.ftl import crops_on_ftl

    assert "molluscan_shellfish" in crops_on_ftl(["Molluscan shellfish (oysters, clams, mussels)"])
    assert "molluscan_shellfish" in crops_on_ftl(["Oysters"])
    assert "molluscan_shellfish" in crops_on_ftl(["Clams"])


def test_unsure_ftl_when_no_crops():
    """When no crops are selected in the Tally form, FTL status is 'unsure'."""
    data = {
        "submissionId": "NO-CROPS",
        "fields": [
            tally_text_field("Annual Farm Revenue", "Under $25,000"),
        ],
    }
    answers, _ = map_submission(data)
    assert answers["handlesFtlFood"] == "unsure"
    assert answers["ftlCategories"] == []
