"""
Tests for model/feedback.py -- turning human overrides into training rows,
and safely promoting a retrained candidate.

train_candidate_with_feedback() itself (the actual retrain) is deliberately
NOT covered by an automated test here: it trains a real XGBoost model on the
full 6000-row training set, which would add several real seconds to every
test run for the sake of re-testing logic model/train.py's own training path
already exercises. It was verified manually end to end (real overrides in a
real audit log, producing a candidate with different metrics than the
deployed model) while building this feature. What IS covered here is the
part that's actually risky to get wrong: turning an override into the
correct labeled row (including across both pipelines' different event
shapes), and making sure promotion never touches a path it wasn't
explicitly given.
"""

import json

import pytest

from audit.audit_log import get_connection, log_event, log_override
from gating.decision_engine import DECISION_CLEAR
from model import feedback as feedback_module
from model.feedback import build_feedback_rows, promote_candidate

RULE_SNAPSHOT = {
    "account_age_days": 900,
    "kyc_status": "complete",
    "business_category": "services",
    "daily_txn_volume": 9000,
    "avg_30d_txn_volume": 9000,
    "total_txns_30d": 500,
    "chargebacks_30d": 0,
    "refunds_30d": 2,
    "avg_ticket_size": 18,
}


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test_feedback_audit.db"))
    yield connection
    connection.close()


class DummyModel:
    """Standing in for a fitted XGBClassifier -- joblib needs a module-level
    (importable) class to pickle, so this can't be defined inside the test."""


def test_build_feedback_rows_maps_clear_override_to_not_risky(conn):
    event_id = log_event(conn, "M1", RULE_SNAPSHOT, 0.5, None, "explanation", "escalate", "reason")
    log_override(conn, event_id, "escalate", DECISION_CLEAR, "Confirmed legitimate with the merchant.")

    rows = build_feedback_rows(conn)
    assert len(rows) == 1
    assert rows.iloc[0]["is_risky"] == 0
    assert rows.iloc[0]["business_category"] == "services"
    assert rows.iloc[0]["merchant_id"] == "M1"


def test_build_feedback_rows_dedupes_to_the_latest_override_per_case(conn):
    """Regression test: a case can be overridden more than once (app/
    dashboard.py's render_override_section explicitly supports "Record
    another override" so a reviewer can correct their own earlier mistake).
    build_feedback_rows used to emit one training row per override ROW, not
    per case -- a stale, superseded override and its correction became two
    feedback rows with IDENTICAL features but OPPOSITE is_risky labels, and
    the stale one was never dropped, silently injecting label noise into
    every future retrain. Only the reviewer's most recent verdict on a case
    should survive into the training set."""
    event_id = log_event(conn, "M1", RULE_SNAPSHOT, 0.85, None, "explanation", "escalate", "reason")
    log_override(conn, event_id, "escalate", "escalate", "first, mistaken override")
    log_override(conn, event_id, "escalate", DECISION_CLEAR, "self-corrected: confirmed legitimate")

    rows = build_feedback_rows(conn)
    assert len(rows) == 1
    assert rows.iloc[0]["is_risky"] == 0  # the corrected label, not the stale one


def test_build_feedback_rows_maps_non_clear_override_to_risky(conn):
    event_id = log_event(conn, "M1", RULE_SNAPSHOT, 0.1, None, "explanation", "clear", "reason")
    log_override(conn, event_id, "clear", "flag_for_compliance_review", "Actually risky on closer review.")

    rows = build_feedback_rows(conn)
    assert rows.iloc[0]["is_risky"] == 1


def test_build_feedback_rows_extracts_features_from_agent_pipeline_trace(conn):
    """
    agent_pipeline events don't store the full merchant profile in
    input_snapshot (only merchant_id + amount) -- the rest only shows up
    inside agent_trace's get_merchant_context result. This locks in that
    build_feedback_rows knows to look there instead of giving up.
    """
    snapshot = {"transaction": {"merchant_id": "M2", "daily_txn_volume": 5000, "razorpay_order_id": "order_x"}}
    trace = [
        {"tool": "get_merchant_context", "arguments": {}, "result": {**RULE_SNAPSHOT, "daily_txn_volume": 5000}},
        {"tool": "score_transaction_risk", "arguments": {}, "result": {"risk_score": 0.3}},
    ]
    event_id = log_event(
        conn, "M2", snapshot, 0.3, None, "explanation", "escalate", "reason",
        source="agent_pipeline", agent_proposal={"recommended_decision": "clear", "reasoning": "x"}, agent_trace=trace,
    )
    log_override(conn, event_id, "escalate", DECISION_CLEAR, "Reviewed and cleared.")

    rows = build_feedback_rows(conn)
    assert len(rows) == 1
    assert rows.iloc[0]["business_category"] == "services"
    assert rows.iloc[0]["daily_txn_volume"] == 5000


def test_build_feedback_rows_skips_incomplete_agent_case(conn):
    """No get_merchant_context step means the feature row can't be completed
    -- must be skipped rather than guessed at with missing fields."""
    snapshot = {"transaction": {"merchant_id": "M3", "daily_txn_volume": 5000, "razorpay_order_id": "order_y"}}
    trace = [{"tool": "score_transaction_risk", "arguments": {}, "result": {"risk_score": 0.3}}]
    event_id = log_event(
        conn, "M3", snapshot, 0.3, None, "explanation", "escalate", "reason",
        source="agent_pipeline", agent_trace=trace,
    )
    log_override(conn, event_id, "escalate", DECISION_CLEAR, "Reviewed.")

    rows = build_feedback_rows(conn)
    assert rows.empty


def test_promote_candidate_never_touches_the_real_deployed_model(tmp_path, monkeypatch):
    """
    promote_candidate overwrites the live model file (and the Models page's
    metrics/chart JSON) -- if this test used the real paths, running the
    suite would silently clobber the project's actual deployed artifacts as
    a side effect. Redirect every path it writes to before calling it.
    """
    fake_model_path = tmp_path / "candidate_model.joblib"
    fake_threshold_path = tmp_path / "candidate_threshold.json"
    fake_metrics_path = tmp_path / "metrics.json"
    fake_chart_data_path = tmp_path / "chart_data.json"
    monkeypatch.setattr(feedback_module, "MODEL_PATH", str(fake_model_path))
    monkeypatch.setattr(feedback_module, "THRESHOLD_PATH", str(fake_threshold_path))
    monkeypatch.setattr(feedback_module, "METRICS_PATH", str(fake_metrics_path))
    monkeypatch.setattr(feedback_module, "CHART_DATA_PATH", str(fake_chart_data_path))
    monkeypatch.setattr(feedback_module, "ARTIFACT_DIR", str(tmp_path))

    candidate_metrics = {"threshold": 0.61, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.7}
    candidate_artifacts = {
        "test_rows": 100,
        "test_positive_rate": 0.1,
        "confusion_matrix": [[80, 10], [5, 5]],
        "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.7},
        "shap_global_importance": [{"feature": "account_age_days", "mean_abs_shap": 0.2}],
    }

    promote_candidate(DummyModel(), 0.61, candidate_metrics, candidate_artifacts)

    assert fake_model_path.exists()
    with open(fake_threshold_path) as f:
        assert json.load(f)["xgboost_threshold"] == 0.61
    with open(fake_metrics_path) as f:
        metrics = json.load(f)
        assert metrics["xgboost"] == candidate_metrics
        assert metrics["test_rows"] == 100
    with open(fake_chart_data_path) as f:
        chart_data = json.load(f)
        assert chart_data["confusion_matrix"]["matrix"] == [[80, 10], [5, 5]]
        assert chart_data["roc_curve"]["xgboost"]["auc"] == 0.7
        assert chart_data["shap_global_importance"][0]["feature"] == "account_age_days"
