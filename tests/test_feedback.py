"""
Tests for model/feedback.py -- turning human overrides into training rows,
and safely promoting a retrained candidate.

The actual model.fit() inside train_candidate_with_feedback() is
deliberately NOT covered by an automated test here: it trains a real
XGBoost model on the full 6000-row training set, which would add several
real seconds to every test run for the sake of re-testing logic
model/train.py's own training path already exercises. It was verified
manually end to end (real overrides in a real audit log, producing a
candidate with different metrics than the deployed model) while building
this feature. What IS covered here is the part that's actually risky to
get wrong: turning an override into the correct labeled row (including
across both pipelines' different event shapes), making sure promotion
never touches a path it wasn't explicitly given, and -- since this is
exactly where a real bug was found and fixed -- that feedback rows
actually land in the TRAINING split rather than being silently sorted
into the held-out test split, which split_original_and_fold_in_feedback
is pulled out specifically so this can be tested without the slow fit().
"""

import json

import pandas as pd
import pytest

from audit.audit_log import get_connection, log_event, log_override
from gating.decision_engine import DECISION_CLEAR
from model import feedback as feedback_module
from model.feedback import build_feedback_rows, promote_candidate, split_original_and_fold_in_feedback

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


def test_build_feedback_rows_excludes_manual_review_overrides_entirely(conn):
    # Regression test: overriding a case to "needs_manual_review" is a
    # reviewer asserting UNCERTAINTY ("I can't decide this myself"), not
    # confirming the merchant is risky -- it used to be treated exactly
    # like an "escalate"/"flag" override (is_risky=1), silently injecting a
    # mislabeled row asserting confirmed risk for an explicitly-undetermined
    # case into every future retrain.
    event_id = log_event(conn, "M1", RULE_SNAPSHOT, 0.5, None, "explanation", "escalate", "reason")
    log_override(conn, event_id, "escalate", "needs_manual_review", "Not enough information to decide either way.")

    rows = build_feedback_rows(conn)
    assert rows.empty


def test_build_feedback_rows_survives_an_overflow_sized_stored_field(conn):
    # Regression test: pd.DataFrame([features]) (without dtype=object) can
    # raise OverflowError itself during pandas' default type-inference on a
    # huge Python int stored in a past event's input_snapshot -- reachable
    # because a reviewer can override ANY case, including one that was
    # originally routed to needs_manual_review precisely because it
    # contained such a value in the first place (see pipeline.py's
    # identical dtype=object fix). Without it, this crashes
    # build_feedback_rows entirely -- not just skipping this one bad
    # override -- since it happens while iterating every override.
    snapshot = dict(RULE_SNAPSHOT, chargebacks_30d=int("9" * 400))
    event_id = log_event(conn, "M1", snapshot, None, None, None, "needs_manual_review", "invalid input")
    log_override(conn, event_id, "needs_manual_review", DECISION_CLEAR, "Reviewed manually, looks fine.")

    rows = build_feedback_rows(conn)  # must not raise
    assert rows.empty  # the huge value is still invalid, so it's correctly skipped -- just without crashing


def test_build_feedback_rows_does_not_silently_truncate_at_the_default_page_size(conn, monkeypatch):
    # Regression test: get_all_overrides' own default limit (1000) exists
    # for a bounded-page caller like a dashboard KPI -- the exact same
    # class of bug already found and fixed once there (commit 4d07816).
    # build_feedback_rows must ask for every override, not accept that
    # same default, or retraining (and the "N usable overrides" count)
    # would silently exclude anything older once volume passes 1000.
    seen_limits = []
    real_get_all_overrides = feedback_module.get_all_overrides

    def spy(conn, limit=1000):
        seen_limits.append(limit)
        return real_get_all_overrides(conn, limit=limit)

    monkeypatch.setattr(feedback_module, "get_all_overrides", spy)

    event_id = log_event(conn, "M1", RULE_SNAPSHOT, 0.5, None, "explanation", "escalate", "reason")
    log_override(conn, event_id, "escalate", DECISION_CLEAR, "Confirmed legitimate.")

    build_feedback_rows(conn)

    assert seen_limits, "get_all_overrides was never called"
    assert all(limit > 1000 for limit in seen_limits)


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


def _make_original_df(n=100):
    """A minimal but realistic original_df: n rows spanning a fixed
    historical date range, half risky/half not, in RAW_REQUIRED_COLUMNS
    shape plus is_risky/snapshot_date."""
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "snapshot_date": dates,
            "account_age_days": [900] * n,
            "kyc_status": ["complete"] * n,
            "business_category": ["services"] * n,
            "daily_txn_volume": [9000.0] * n,
            "avg_30d_txn_volume": [9000.0] * n,
            "total_txns_30d": [500] * n,
            "chargebacks_30d": [0] * n,
            "refunds_30d": [2] * n,
            "avg_ticket_size": [18.0] * n,
            "is_risky": [i % 2 for i in range(n)],
        }
    )


def _make_feedback_df(n=1, timestamp="2026-08-29T00:00:00"):
    """A feedback row shaped like build_feedback_rows' output -- stamped
    with a real-world override timestamp, which is always later than
    original_df's fixed historical date range."""
    return pd.DataFrame(
        {
            "snapshot_date": [timestamp] * n,
            "account_age_days": [900] * n,
            "kyc_status": ["complete"] * n,
            "business_category": ["services"] * n,
            "daily_txn_volume": [9000.0] * n,
            "avg_30d_txn_volume": [9000.0] * n,
            "total_txns_30d": [500] * n,
            "chargebacks_30d": [0] * n,
            "refunds_30d": [2] * n,
            "avg_ticket_size": [18.0] * n,
            "is_risky": [1] * n,
            "merchant_id": [f"FEEDBACK_{i}" for i in range(n)],
        }
    )


def test_feedback_rows_land_in_training_split_not_test_split():
    """Regression test for the bug this function was written to fix:
    feedback rows are timestamped with the override's real-world date,
    always later than original_df's historical range. Sorting a
    concatenation of the two by snapshot_date and slicing by fixed
    fractions put every feedback row at the very end -- i.e. in the test
    split, never train -- so a human's corrections had zero effect on what
    the candidate model actually learned."""
    original_df = _make_original_df(n=100)
    feedback_df = _make_feedback_df(n=1)

    train_df, val_df, test_df = split_original_and_fold_in_feedback(original_df, feedback_df)

    assert "FEEDBACK_0" in train_df["merchant_id"].values
    assert "FEEDBACK_0" not in val_df.get("merchant_id", pd.Series(dtype=object)).values
    assert "FEEDBACK_0" not in test_df.get("merchant_id", pd.Series(dtype=object)).values


def test_feedback_rows_do_not_shrink_the_held_out_test_split():
    """val_df/test_df must be derived from original_df alone, so the
    before/after comparison stays evaluated on a stable, genuinely
    held-out slice across retrains -- not one that grows/shrinks or gets
    contaminated by whatever feedback happens to exist yet."""
    original_df = _make_original_df(n=100)

    train_no_fb, val_no_fb, test_no_fb = split_original_and_fold_in_feedback(original_df, pd.DataFrame())
    train_with_fb, val_with_fb, test_with_fb = split_original_and_fold_in_feedback(original_df, _make_feedback_df(n=5))

    assert len(val_with_fb) == len(val_no_fb)
    assert len(test_with_fb) == len(test_no_fb)
    assert len(train_with_fb) == len(train_no_fb) + 5


def test_split_with_no_feedback_returns_original_split_unchanged():
    original_df = _make_original_df(n=100)
    train_df, val_df, test_df = split_original_and_fold_in_feedback(original_df, pd.DataFrame())
    assert len(train_df) + len(val_df) + len(test_df) == 100
    assert "merchant_id" not in train_df.columns  # original_df has no merchant_id column


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


def test_promote_candidate_leaves_no_partial_state_on_mid_promotion_failure(tmp_path, monkeypatch):
    """Regression test: promote_candidate used to write
    joblib.dump(candidate_model, MODEL_PATH) directly to the final path as
    its very first step, followed by four more independent direct writes
    with no rollback -- a crash partway through left a brand-new model
    permanently on disk paired with STALE threshold/metrics/chart_data from
    the previous model (a silently wrong gate, not a visible crash). Every
    artifact is now written to a temp file first and swapped into place
    only after every temp write succeeds, so a failure partway through must
    leave the real files completely untouched."""
    fake_model_path = tmp_path / "candidate_model.joblib"
    fake_threshold_path = tmp_path / "candidate_threshold.json"
    fake_metrics_path = tmp_path / "metrics.json"
    fake_chart_data_path = tmp_path / "chart_data.json"
    monkeypatch.setattr(feedback_module, "MODEL_PATH", str(fake_model_path))
    monkeypatch.setattr(feedback_module, "THRESHOLD_PATH", str(fake_threshold_path))
    monkeypatch.setattr(feedback_module, "METRICS_PATH", str(fake_metrics_path))
    monkeypatch.setattr(feedback_module, "CHART_DATA_PATH", str(fake_chart_data_path))
    monkeypatch.setattr(feedback_module, "ARTIFACT_DIR", str(tmp_path))

    # Pre-existing "old model's" artifacts -- these must survive untouched.
    fake_model_path.write_text("OLD MODEL BYTES")
    fake_threshold_path.write_text(json.dumps({"xgboost_threshold": 0.50}))
    fake_metrics_path.write_text(json.dumps({"xgboost": {"f1": 0.1}}))
    fake_chart_data_path.write_text(json.dumps({"confusion_matrix": {"matrix": [[1, 0], [0, 1]]}}))

    real_json_dump = json.dump
    call_count = {"n": 0}

    def flaky_json_dump(obj, fp, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:  # the chart_data temp write, after model/threshold/metrics temp writes already succeeded
            raise OSError("simulated crash mid-promotion")
        return real_json_dump(obj, fp, **kwargs)

    monkeypatch.setattr(feedback_module.json, "dump", flaky_json_dump)

    candidate_metrics = {"threshold": 0.61, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.7}
    candidate_artifacts = {
        "test_rows": 100,
        "test_positive_rate": 0.1,
        "confusion_matrix": [[80, 10], [5, 5]],
        "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.7},
        "shap_global_importance": [{"feature": "account_age_days", "mean_abs_shap": 0.2}],
    }

    with pytest.raises(OSError, match="simulated crash"):
        promote_candidate(DummyModel(), 0.61, candidate_metrics, candidate_artifacts)

    # The real files are exactly what they were before -- the old model, not
    # a new model silently paired with a stale threshold.
    assert fake_model_path.read_text() == "OLD MODEL BYTES"
    assert json.loads(fake_threshold_path.read_text())["xgboost_threshold"] == 0.50
    assert json.loads(fake_metrics_path.read_text())["xgboost"]["f1"] == 0.1
    assert json.loads(fake_chart_data_path.read_text())["confusion_matrix"]["matrix"] == [[1, 0], [0, 1]]

    # No leftover temp files from the failed attempt. (Temp files carry a
    # unique per-call suffix now -- see the concurrent-promotion regression
    # test below -- so match on the ".tmp_promote." prefix rather than an
    # exact ".tmp_promote" suffix.)
    assert list(tmp_path.glob("*.tmp_promote.*")) == []


def test_promote_candidate_uses_unique_temp_paths_so_concurrent_promotions_cannot_collide(tmp_path, monkeypatch):
    # Regression test: temp paths used to be a FIXED name
    # (f"{final_path}.tmp_promote") with no per-call uniqueness. Two
    # reviewers promoting at close to the same time -- realistic under
    # Streamlit's shared-server, thread-per-session model -- could write to
    # the very same temp file, and one promotion's failure-path cleanup
    # (`os.remove` on every tmp path) could delete a temp file that still
    # belonged to the OTHER, still-in-progress promotion. Capturing every
    # temp path actually written to disk across two "concurrent" calls
    # (simulated here by hooking joblib.dump) must show two distinct sets.
    monkeypatch.setattr(feedback_module, "MODEL_PATH", str(tmp_path / "candidate_model.joblib"))
    monkeypatch.setattr(feedback_module, "THRESHOLD_PATH", str(tmp_path / "candidate_threshold.json"))
    monkeypatch.setattr(feedback_module, "METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setattr(feedback_module, "CHART_DATA_PATH", str(tmp_path / "chart_data.json"))
    monkeypatch.setattr(feedback_module, "ARTIFACT_DIR", str(tmp_path))

    seen_model_tmp_paths = []
    real_joblib_dump = feedback_module.joblib.dump

    def recording_joblib_dump(obj, path, *args, **kwargs):
        seen_model_tmp_paths.append(path)
        return real_joblib_dump(obj, path, *args, **kwargs)

    monkeypatch.setattr(feedback_module.joblib, "dump", recording_joblib_dump)

    candidate_metrics = {"threshold": 0.61, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.7}
    candidate_artifacts = {
        "test_rows": 100,
        "test_positive_rate": 0.1,
        "confusion_matrix": [[80, 10], [5, 5]],
        "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.7},
        "shap_global_importance": [{"feature": "account_age_days", "mean_abs_shap": 0.2}],
    }

    promote_candidate(DummyModel(), 0.61, candidate_metrics, candidate_artifacts)
    promote_candidate(DummyModel(), 0.61, candidate_metrics, candidate_artifacts)

    assert len(seen_model_tmp_paths) == 2
    assert seen_model_tmp_paths[0] != seen_model_tmp_paths[1]
    # Neither call's temp path is left behind -- both promotions succeeded
    # and swapped their (distinct) temp files into place.
    assert list(tmp_path.glob("*.tmp_promote.*")) == []
