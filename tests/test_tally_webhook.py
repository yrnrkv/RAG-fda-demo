"""Tests for the Tally lead-gen → exemption flow (FTL lookup, mapper, webhook).

The webhook fixture mirrors the real two-row example (Oscar Kwan): a 1,000+ acre
California leafy-greens farm doing $1M-$5M across wholesale/grocery/direct. Expected
deterministic verdict: not_exempt (on the FTL, far above the $25k threshold, not
direct-to-consumer only).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from fda_traceability_rag.ftl import crops_on_ftl, ftl_category_for
from service.tally_mapping import map_submission, revenue_band_max
from tests.helpers import tally_checkbox_field as _checkbox, tally_text_field as _text


OSCAR_DATA = {
    "submissionId": "QoAEVB8",
    "respondentId": "vGdz79Q",
    "fields": [
        _text("First Name", "Oscar"),
        _text("Last Name", "Kwan"),
        _text("Best email to reach you", "oscar@binox.com.hk"),
        _text("Phone number", "96998921"),
        _text("State of your main farm", "California"),
        _text("Farm Size", "1,000 acres +"),
        _text("Annual Farm Revenue", "$1M – $5M"),
        _checkbox(
            "Main Crop Type (Select all that apply)",
            "Leafy greens (fresh-cut)",
            "Herbs",
            "Leafy greens",
        ),
        _checkbox(
            "Which steps do you handle on your farm? (Select all that apply)",
            "Cooling before packing",
            "Initial packing",
        ),
        _checkbox(
            "Distribution Channel (Select all that apply)",
            "Wholesalers or distributors",
            "Direct to consumers (farmers markets, CSA, farm stand, online)",
            "National or regional grocery stores (Walmart, Kroger, etc.)",
        ),
        _text("What's your #1 headache right now with FSMA compliance?", "Paperwork burden"),
        _text(
            "How much would you willing to pay per month for a tool ...",
            "$60 – $99/month",
        ),
    ],
}


def test_ftl_lookup_leafy_greens_and_herbs():
    assert ftl_category_for("Leafy greens (fresh-cut)") == "leafy_greens"
    assert ftl_category_for("Herbs") == "herbs"
    assert ftl_category_for("Wheat") is None
    assert "leafy_greens" in crops_on_ftl(["Leafy greens", "Wheat"])


def test_revenue_band_max():
    assert revenue_band_max("$1M – $5M") == 5_000_000
    assert revenue_band_max("Under $25,000") == 25_000
    assert revenue_band_max("$5M+") is None  # open-ended


def test_oscar_maps_to_not_exempt():
    answers, profile = map_submission(OSCAR_DATA)
    assert answers["handlesFtlFood"] == "yes"
    assert "leafy_greens" in answers["ftlCategories"]
    assert "wholesale_distribution" in answers["salesChannels"]
    assert "direct_consumer" in answers["salesChannels"]
    assert answers["produce3yrAvgUsd"] == 5_000_000  # far above $25k

    from fda_traceability_rag.exemption import SurveyAnswers, evaluate_exemption

    result = evaluate_exemption(SurveyAnswers.from_camel(answers))
    assert result.verdict == "not_exempt"
    assert profile["email"] == "oscar@binox.com.hk"
    assert profile["willingnessToPay"] == "$60 – $99/month"


def test_direct_only_small_farm_is_exempt():
    data = {
        "submissionId": "X",
        "fields": [
            _text("Annual Farm Revenue", "Under $25,000"),
            _checkbox("Main Crop Type (Select all that apply)", "Tomatoes"),
            _checkbox(
                "Distribution Channel (Select all that apply)",
                "Direct to consumers (farmers markets, CSA, farm stand, online)",
            ),
        ],
    }
    answers, _ = map_submission(data)
    from fda_traceability_rag.exemption import SurveyAnswers, evaluate_exemption

    result = evaluate_exemption(SurveyAnswers.from_camel(answers))
    # direct-to-consumer only -> exempt under 1.1305(b)
    assert result.verdict == "exempt"
    assert any(r.rule_id == "direct_to_consumer" for r in result.matched_rules)


def test_webhook_endpoint_with_signature():
    from fastapi.testclient import TestClient

    import service.app as app_module

    app_module.TALLY_SIGNING_SECRET = "shh"
    client = TestClient(app_module.app)

    body = json.dumps({"data": OSCAR_DATA}).encode()
    sig = base64.b64encode(hmac.new(b"shh", body, hashlib.sha256).digest()).decode()

    bad = client.post("/webhooks/tally", content=body, headers={"tally-signature": "wrong"})
    assert bad.status_code == 401

    ok = client.post("/webhooks/tally", content=body, headers={"tally-signature": sig})
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["verdict"] == "not_exempt"
    assert "Oscar" in payload["customerEmail"]["body"]
    assert payload["leadRow"]["email"] == "oscar@binox.com.hk"
    assert payload["leadRow"]["verdict"] == "not_exempt"
    assert "New FSMA 204 lead" in payload["slackText"]

    app_module.TALLY_SIGNING_SECRET = ""  # reset for other tests
