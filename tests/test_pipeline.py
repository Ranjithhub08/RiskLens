import os
import tempfile

import pytest

from audit.audit_log import get_all_events, get_connection
from explainability.explain import RiskExplainer
from pipeline import load_model, score_record

MODEL_PATH = "model/artifacts/xgb_model.joblib"

GOOD_RECORD = {
    "merchant_id": "TEST_M1",
    "account_age_days": 30,
    "kyc_status": "incomplete",
    "business_category": "electronics",
    "daily_txn_volume": 50000.0,
    "avg_30d_txn_volume": 8000.0,  # big spike -> should push risk up
    "total_txns_30d": 100,
    "chargebacks_30d": 8,
    "refunds_30d": 15,
    "avg_ticket_size": 80.0,
}

LOW_RISK_RECORD = {
    "merchant_id": "TEST_M2",
    "account_age_days": 900,
    "kyc_status": "complete",
    "business_category": "services",
    "daily_txn_volume": 9000.0,
    "avg_30d_txn_volume": 9000.0,
    "total_txns_30d": 500,
    "chargebacks_30d": 0,
    "refunds_30d": 2,
    "avg_ticket_size": 18.0,
}

INCOMPLETE_RECORD = {
    "merchant_id": "TEST_M3",
    "account_age_days": 100,
    # kyc_status intentionally missing
    "business_category": "fashion",
    "daily_txn_volume": 5000.0,
    # avg_30d_txn_volume intentionally missing
    "total_txns_30d": 50,
    "chargebacks_30d": 1,
    "refunds_30d": 1,
    "avg_ticket_size": 100.0,
}


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model artifact not found -- run model/train.py first")
    return load_model()


@pytest.fixture(scope="module")
def explainer(model):
    return RiskExplainer(model)


@pytest.fixture
def temp_conn():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = get_connection(path)
    yield conn
    conn.close()
    os.remove(path)


def test_high_risk_pattern_scores_higher_than_low_risk_pattern(model, explainer, temp_conn):
    high = score_record(GOOD_RECORD, model, explainer, temp_conn)
    low = score_record(LOW_RISK_RECORD, model, explainer, temp_conn)
    assert high["risk_score"] is not None
    assert low["risk_score"] is not None
    assert high["risk_score"] > low["risk_score"]


def test_every_scored_record_gets_an_explanation(model, explainer, temp_conn):
    result = score_record(GOOD_RECORD, model, explainer, temp_conn)
    assert result["explanation"] is not None
    assert len(result["explanation"]) > 0
    assert result["top_factors"] is not None
    assert len(result["top_factors"]) > 0


def test_incomplete_record_fails_safe_to_manual_review(model, explainer, temp_conn):
    result = score_record(INCOMPLETE_RECORD, model, explainer, temp_conn)
    assert result["decision"] == "needs_manual_review"
    assert result["risk_score"] is None
    assert len(result["missing_fields"]) > 0


def test_every_scoring_call_writes_exactly_one_audit_event(model, explainer, temp_conn):
    score_record(GOOD_RECORD, model, explainer, temp_conn)
    score_record(LOW_RISK_RECORD, model, explainer, temp_conn)
    score_record(INCOMPLETE_RECORD, model, explainer, temp_conn)
    events = get_all_events(temp_conn)
    assert len(events) == 3


def test_decision_always_one_of_the_bounded_set(model, explainer, temp_conn):
    valid_decisions = {"clear", "escalate", "flag_for_compliance_review", "needs_manual_review"}
    for record in [GOOD_RECORD, LOW_RISK_RECORD, INCOMPLETE_RECORD]:
        result = score_record(record, model, explainer, temp_conn)
        assert result["decision"] in valid_decisions


def test_absurdly_large_but_finite_field_fails_safe_instead_of_crashing_xgboost(model, explainer, temp_conn):
    # A finite, non-negative float like 1e40 passes a plain "is it a valid
    # non-negative number" check, but XGBoost's predict_proba represents
    # features as float32 internally (max ~3.4e38) and raises an uncaught
    # XGBoostError on anything past that -- which used to skip log_event
    # entirely and silently drop the audit trail for the attempt.
    record = dict(GOOD_RECORD, account_age_days=1e40)
    result = score_record(record, model, explainer, temp_conn)
    assert result["decision"] == "needs_manual_review"
    assert result["risk_score"] is None
    events = get_all_events(temp_conn)
    assert len(events) == 1


def test_absurdly_large_integer_field_fails_safe_instead_of_crashing_pandas(model, explainer, temp_conn):
    # A Python int with hundreds of digits overflows a C double. Without
    # dtype=object on the DataFrame construction, this raised OverflowError
    # inside pandas itself, before find_missing_or_invalid ever ran.
    record = dict(GOOD_RECORD, chargebacks_30d=int("9" * 400))
    result = score_record(record, model, explainer, temp_conn)
    assert result["decision"] == "needs_manual_review"
    assert result["risk_score"] is None
    events = get_all_events(temp_conn)
    assert len(events) == 1


def test_list_valued_field_fails_safe_instead_of_crashing_on_ambiguous_truth_value(model, explainer, temp_conn):
    # A hallucinated/malformed caller (e.g. an LLM tool call) could hand a
    # list where a scalar is expected. pd.isna() on a list/array returns an
    # array, and evaluating that array's truthiness raises "ValueError: The
    # truth value of an array... is ambiguous" instead of failing safe.
    record = dict(GOOD_RECORD, account_age_days=[1, 2, 3])
    result = score_record(record, model, explainer, temp_conn)
    assert result["decision"] == "needs_manual_review"
    assert result["risk_score"] is None
    events = get_all_events(temp_conn)
    assert len(events) == 1


def test_belt_and_suspenders_nan_category_reports_one_reason_not_six(model, explainer, temp_conn, monkeypatch):
    # Regression test: transform_features sets ALL SIX category_* one-hot
    # columns to NaN together whenever business_category is NaN, so the
    # belt-and-suspenders branch in score_record (reached only if a value
    # slips past find_missing_or_invalid but still produces a NaN feature)
    # used to report "business_category (unrecognized value): <category>"
    # once per one-hot column -- six duplicate, confusing entries -- instead
    # of naming the field once. find_missing_or_invalid already rejects any
    # NaN/unrecognized business_category before this branch is reachable in
    # practice, so it's forced here by monkeypatching validation to let one
    # through.
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "find_missing_or_invalid", lambda df: [])
    record = dict(GOOD_RECORD, business_category=None)
    result = score_record(record, model, explainer, temp_conn)

    assert result["decision"] == "needs_manual_review"
    assert result["risk_score"] is None
    category_reasons = [f for f in result["missing_fields"] if "business_category" in f]
    assert category_reasons == ["business_category (unrecognized value)"]
