"""
End-to-end scoring pipeline: one record in, one audited decision out.

This is the single function both the Streamlit dashboard and the optional
API call. Keeping the orchestration logic in one place (rather than
duplicating it in app/dashboard.py and api/main.py) is what keeps this
"no bugs, everything works properly" -- there's only one place the flow
described in ARCHITECTURE.md section 5 can go wrong, and it's covered by
tests/test_pipeline.py.
"""

import os

import joblib
import pandas as pd

from audit.audit_log import get_connection, log_event
from explainability.explain import RiskExplainer
from features.features import find_missing_or_invalid, transform_features
from gating.decision_engine import decide_for_record

MODEL_PATH = "model/artifacts/xgb_model.joblib"
THRESHOLD_PATH = "model/artifacts/decision_threshold.json"


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def score_record(record: dict, model, explainer: RiskExplainer, conn) -> dict:
    """
    record: a dict of raw merchant snapshot fields (see features.RAW_REQUIRED_COLUMNS)
    model: the loaded XGBoost model
    explainer: a RiskExplainer wrapping that model
    conn: an open audit_log sqlite connection

    Returns a dict describing the full outcome, and writes one audit event
    regardless of which branch (scored vs. missing-data fail-safe) is hit.
    """
    df = pd.DataFrame([record])
    missing = find_missing_or_invalid(df)

    risk_score = None
    explanation = None
    top_factors = None

    if not missing:
        X = transform_features(df)
        if X.isna().any(axis=None):
            # Belt-and-suspenders: find_missing_or_invalid should have already
            # caught anything that would produce a NaN feature, but if a value
            # was present, non-empty, and still produced NaN after transform
            # (e.g. a value type we didn't anticipate), fail safe rather than
            # score on it.
            missing = [
                col.replace("category_", "business_category (unrecognized value): ")
                for col in X.columns[X.isna().any()].tolist()
            ]
        else:
            risk_score = float(model.predict_proba(X)[:, 1][0])
            explained = explainer.explain_row(X)
            explanation = explained["explanation"]
            top_factors = explained["top_factors"]

    gating_result = decide_for_record(missing, risk_score)

    event_id = log_event(
        conn,
        merchant_id=record.get("merchant_id"),
        input_snapshot=record,
        risk_score=risk_score,
        top_factors=top_factors,
        explanation=explanation,
        decision=gating_result.decision,
        decision_reason=gating_result.reason,
    )

    return {
        "event_id": event_id,
        "risk_score": risk_score,
        "explanation": explanation,
        "top_factors": top_factors,
        "decision": gating_result.decision,
        "decision_reason": gating_result.reason,
        "missing_fields": missing,
    }
