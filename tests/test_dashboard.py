"""
Tests for the case-detail rendering helpers in app/dashboard.py.

This file previously had no test coverage at all, despite two rounds of
stored-XSS fixes here (merchant_id in commit 8a28add, then a sibling gap in
kyc_status/business_category/chargebacks_30d/refunds_30d found in a later
review) -- exactly the kind of critical-path-with-no-coverage gap a future
edit could silently reintroduce. merchant_context_display_values() was
pulled out of render_case_detail specifically so this escaping logic is
testable without a running Streamlit app.
"""

import app.dashboard as dashboard

BASE_VIEW = {
    "merchant_id": "merchant_1001",
    "kyc_status": "complete",
    "business_category": "electronics",
    "chargebacks_30d": 2,
    "refunds_30d": 5,
    "account_age_days": 400,
    "daily_txn_volume": 12000.0,
    "avg_30d_txn_volume": 11000.0,
    "avg_ticket_size": 250.0,
}

XSS_PAYLOAD = "<img src=x onerror=alert(document.cookie)>"


def test_valid_row_renders_without_escaping_artifacts():
    ctx = dashboard.merchant_context_display_values(BASE_VIEW)
    assert ctx["merchant"] == "merchant_1001"
    assert ctx["kyc"] == "Complete"
    assert ctx["category"] == "Electronics"
    assert ctx["chargebacks"] == "2"
    assert ctx["refunds"] == "5"
    assert ctx["account_age"] == "400 days"
    assert "12,000.00" in ctx["daily_txn"]


def test_merchant_id_xss_payload_is_escaped():
    view = dict(BASE_VIEW, merchant_id=XSS_PAYLOAD)
    ctx = dashboard.merchant_context_display_values(view)
    assert "<img" not in ctx["merchant"]
    assert "&lt;img" in ctx["merchant"]


def test_business_category_xss_payload_is_escaped():
    # Reachable via Batch Scoring: a CSV row whose business_category fails
    # find_missing_or_invalid's allow-list check is still logged and still
    # displayed here (that's what "needs_manual_review" means), not
    # discarded -- so this field is exactly as attacker-reachable as
    # merchant_id.
    view = dict(BASE_VIEW, business_category=XSS_PAYLOAD)
    ctx = dashboard.merchant_context_display_values(view)
    assert "<img" not in ctx["category"].lower()
    assert "&lt;" in ctx["category"]  # .title() case-shifts the tag text but escaping still holds


def test_kyc_status_xss_payload_is_escaped():
    view = dict(BASE_VIEW, kyc_status=XSS_PAYLOAD)
    ctx = dashboard.merchant_context_display_values(view)
    assert "<img" not in ctx["kyc"].lower()


def test_chargebacks_and_refunds_xss_payload_is_escaped():
    # find_missing_or_invalid has no bounds/type check strong enough to
    # reject a string here reaching the audit log via a raw batch upload of
    # an invalid row -- these fields "look" numeric but must be escaped too.
    view = dict(BASE_VIEW, chargebacks_30d=XSS_PAYLOAD, refunds_30d=XSS_PAYLOAD)
    ctx = dashboard.merchant_context_display_values(view)
    assert "<img" not in ctx["chargebacks"]
    assert "<img" not in ctx["refunds"]


def test_non_numeric_metric_field_does_not_crash_and_is_escaped():
    # A batch CSV row with a non-numeric account_age_days used to crash the
    # whole case-detail render with an uncaught ValueError from the ":.0f"
    # format spec. It must degrade to an escaped string instead.
    view = dict(BASE_VIEW, account_age_days="not-a-number", daily_txn_volume=XSS_PAYLOAD)
    ctx = dashboard.merchant_context_display_values(view)
    assert ctx["account_age"] == "not-a-number"
    assert "<img" not in ctx["daily_txn"]


def test_none_values_render_as_placeholder():
    view = dict(BASE_VIEW, merchant_id=None, kyc_status=None, business_category=None,
                chargebacks_30d=None, refunds_30d=None, account_age_days=None,
                daily_txn_volume=None, avg_30d_txn_volume=None, avg_ticket_size=None)
    ctx = dashboard.merchant_context_display_values(view)
    assert all(v == "—" for v in ctx.values())
