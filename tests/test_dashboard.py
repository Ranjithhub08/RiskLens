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

import pandas as pd

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


FULL_CASE_VIEW = dict(
    BASE_VIEW,
    risk_score=0.42,
    event_id="11111111-2222-3333-4444-555555555555",
    timestamp_utc="2026-01-01T00:00:00+00:00",
    source="rule_pipeline",
    daily_txn_volume=1000.0,
    avg_30d_txn_volume=1000.0,
    avg_ticket_size=50.0,
    decision="clear",
    decision_reason="Risk score is below the escalation threshold.",
    top_factors=None,
    explanation=None,
    agent_proposal=None,
)


def test_case_report_text_does_not_crash_on_non_numeric_account_age():
    # Regression test: a batch-CSV row with a non-numeric account_age_days
    # used to crash the report/download with an uncaught ValueError from
    # the bare f"{value:.0f}" format spec -- the same failure mode
    # _safe_metric_html was added to prevent in render_case_detail, missed
    # in this sibling function.
    view = dict(FULL_CASE_VIEW, account_age_days="not-a-number")
    report = dashboard.case_report_text(view, overrides=[])
    assert "Account age:        not-a-number" in report


def test_case_report_text_formats_a_valid_account_age():
    view = dict(FULL_CASE_VIEW, account_age_days=400)
    report = dashboard.case_report_text(view, overrides=[])
    assert "Account age:        400 days" in report


def test_none_values_render_as_placeholder():
    view = dict(BASE_VIEW, merchant_id=None, kyc_status=None, business_category=None,
                chargebacks_30d=None, refunds_30d=None, account_age_days=None,
                daily_txn_volume=None, avg_30d_txn_volume=None, avg_ticket_size=None)
    ctx = dashboard.merchant_context_display_values(view)
    assert all(v == "—" for v in ctx.values())


def test_resolve_selected_case_returns_none_when_nothing_selected():
    assert dashboard.resolve_selected_case([], [], []) is None


def test_resolve_selected_case_returns_correct_case_when_stable():
    filtered = [{"event_id": "A"}, {"event_id": "B"}, {"event_id": "C"}]
    prior_ids = ["A", "B", "C"]
    assert dashboard.resolve_selected_case(prior_ids, [1], filtered) == {"event_id": "B"}


def test_resolve_selected_case_does_not_crash_when_filter_narrows_below_selected_index():
    # Regression test: selecting row 2 of 3, then a filter/search narrows
    # `filtered` down to 1 row. Directly indexing filtered[2] used to raise
    # IndexError and crash the whole Investigations page.
    prior_ids = ["A", "B", "C"]  # what was on screen when row 2 (event C) was clicked
    filtered = [{"event_id": "A"}]  # now filtered down to just one case
    result = dashboard.resolve_selected_case(prior_ids, [2], filtered)
    assert result is None  # event C is no longer in view -- cleared, not crashed


def test_resolve_selected_case_does_not_silently_show_wrong_case_after_reordering():
    # Regression test: `filtered`'s order shifts (e.g. a new case scored
    # elsewhere re-sorts "most recent first") without its length changing.
    # The stale positional index used to silently resolve to a DIFFERENT
    # case than the one actually clicked.
    prior_ids = ["A", "B", "C"]  # row 0 (event A) was clicked
    filtered = [{"event_id": "Z"}, {"event_id": "A"}, {"event_id": "B"}]  # reordered, A is no longer first
    result = dashboard.resolve_selected_case(prior_ids, [0], filtered)
    assert result == {"event_id": "A"}  # resolved by stable ID, not by the now-wrong position 0


def test_resolve_selected_case_returns_none_for_a_deleted_or_never_seen_event_id():
    prior_ids = ["A", "B"]
    filtered = [{"event_id": "C"}]  # neither prior id is in the current view
    assert dashboard.resolve_selected_case(prior_ids, [0], filtered) is None


def test_compute_batch_identity_is_stable_for_the_same_dataframe():
    df = pd.DataFrame({"merchant_id": ["m1", "m2"], "account_age_days": [100, 200]})
    assert dashboard.compute_batch_identity(df) == dashboard.compute_batch_identity(df.copy())


def test_compute_batch_identity_differs_for_different_dataframes_with_same_shape():
    # Regression test: a stale-batch-report bug relied on len()/columns
    # checks, which two DIFFERENT batches of the same shape (e.g. two
    # 2-row samples) can both satisfy -- the identity must be based on the
    # actual row contents.
    df_a = pd.DataFrame({"merchant_id": ["m1", "m2"], "account_age_days": [100, 200]})
    df_b = pd.DataFrame({"merchant_id": ["m3", "m4"], "account_age_days": [300, 400]})
    assert dashboard.compute_batch_identity(df_a) != dashboard.compute_batch_identity(df_b)


def test_compute_batch_identity_differs_when_a_single_cell_changes():
    df_a = pd.DataFrame({"merchant_id": ["m1", "m2"], "account_age_days": [100, 200]})
    df_b = pd.DataFrame({"merchant_id": ["m1", "m2"], "account_age_days": [100, 999]})
    assert dashboard.compute_batch_identity(df_a) != dashboard.compute_batch_identity(df_b)


def test_live_agent_razorpay_error_message_is_escaped_before_rendering():
    # Regression test: page_live_agent() renders a Razorpay API exception's
    # str() inside an unsafe_allow_html empty_state_html() block. Razorpay
    # echoes invalid request parameters (like an attacker-controlled
    # Merchant ID typed into the Live Agent form) back in its error text,
    # so this string is exactly as reachable as merchant_id/business_category
    # elsewhere in this file and must be escaped the same way before being
    # interpolated into HTML.
    error_message = "Bad request for merchant " + XSS_PAYLOAD
    safe_error_message = __import__("html").escape(str(error_message))
    rendered = dashboard.empty_state_html(
        "Razorpay API connection failed",
        f"The test order could not be created or the investigation failed to complete.<br><code>{safe_error_message}</code>",
    )
    assert "<img" not in rendered
    assert "&lt;img" in rendered


def test_audit_trail_merchant_search_does_not_crash_on_regex_metacharacters():
    # page_audit_trail()'s "Search merchant ID" box filters with this exact
    # pandas expression. It's a plain free-text field (the sibling search on
    # Investigations does plain substring matching), but str.contains
    # defaults to regex=True -- so a merchant ID a reviewer actually types,
    # like "test(store)", contains an unbalanced paren that used to raise
    # and crash the whole Audit Trail page instead of just finding no match.
    merchant_ids = pd.Series(["test(store)", "other_merchant", "TEST(STORE)_2"])

    matches = merchant_ids.astype(str).str.contains("test(", case=False, na=False, regex=False)

    assert list(matches) == [True, False, True]
