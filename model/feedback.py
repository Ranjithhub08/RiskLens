"""
Turns human overrides (audit/audit_log.py's human_overrides table) into
training data, and retrains a CANDIDATE model on the original training set
plus that feedback -- without silently replacing the live production model.

This deliberately mirrors the "propose, then a human decides" pattern used
everywhere else in RiskLens: the agent proposes a decision and the
deterministic gate is what actually decides; here, a retrain proposes a
candidate model and its measured before/after impact, and a human reviewer
explicitly promotes it (see promote_candidate) only if the numbers look
better. Retraining always happens on demand; promoting to production is a
separate, deliberate, auditable step.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, roc_curve
from xgboost import XGBClassifier

from audit.audit_log import get_all_overrides, get_event_by_id
from features.features import RAW_REQUIRED_COLUMNS, find_missing_or_invalid, transform_features
from gating.decision_engine import DECISION_CLEAR
from model.train import (
    RAW_DATA_PATH,
    TRAIN_FRACTION,
    VAL_FRACTION,
    best_threshold_for_f1,
    evaluate,
)

ARTIFACT_DIR = "model/artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.joblib")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "decision_threshold.json")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")
CHART_DATA_PATH = os.path.join(ARTIFACT_DIR, "chart_data.json")
# The raw (pre-transform) held-out test rows that actually produced whatever
# model is currently deployed. Written by promote_candidate, read by the
# dashboard's Threshold Explorer (see app/dashboard.py's _test_set_predictions)
# instead of it independently re-deriving a test split from
# data/raw/merchant_snapshots.csv -- see promote_candidate's docstring for why
# that independent re-derivation goes stale the moment feedback rows are
# involved.
TEST_SNAPSHOT_PATH = os.path.join(ARTIFACT_DIR, "deployed_test_snapshot.csv")


def _extract_features_from_event(event: dict) -> dict:
    """
    Pull a full raw feature row (same shape as data/raw/merchant_snapshots.csv)
    out of a logged audit event, regardless of which pipeline produced it.

    rule_pipeline events store the complete record directly in input_snapshot.
    agent_pipeline events only store the transaction (merchant_id, amount) in
    input_snapshot -- the rest of the merchant profile was fetched live during
    the investigation and only appears inside agent_trace's get_merchant_context
    tool result, so we look there instead.
    """
    snapshot = event.get("input_snapshot")
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except (TypeError, json.JSONDecodeError):
            snapshot = {}
    snapshot = snapshot or {}

    source = event.get("source") or "rule_pipeline"
    row = {"merchant_id": event.get("merchant_id")}

    if source == "rule_pipeline":
        for col in RAW_REQUIRED_COLUMNS:
            row[col] = snapshot.get(col)
        return row

    row["daily_txn_volume"] = (snapshot.get("transaction") or {}).get("daily_txn_volume")
    agent_trace = event.get("agent_trace")
    if isinstance(agent_trace, str):
        try:
            agent_trace = json.loads(agent_trace)
        except (TypeError, json.JSONDecodeError):
            agent_trace = None
    for step in agent_trace or []:
        if step.get("tool") == "get_merchant_context" and isinstance(step.get("result"), dict):
            ctx = step["result"]
            for col in RAW_REQUIRED_COLUMNS:
                if col != "daily_txn_volume":
                    row[col] = ctx.get(col)
            break
    return row


def build_feedback_rows(conn) -> pd.DataFrame:
    """
    One row per human-overridden CASE (event_id), in the exact schema
    model/train.py expects -- ready to append onto the original training
    data for a retrain.

    A case can be overridden more than once -- app/dashboard.py's
    render_override_section explicitly supports "Record another override"
    on a case that already has one, so a reviewer can correct their own
    earlier override. get_all_overrides(conn) returns every override ROW,
    most-recent-first, not one per event -- iterating it directly used to
    turn a self-correction into two feedback rows with IDENTICAL features
    but OPPOSITE is_risky labels, and the stale, superseded row was never
    dropped, silently injecting label noise into every future retrain
    proportional to how many times a case was re-reviewed. Deduplicating to
    the latest override per event_id here (relying on that most-recent-first
    ordering) keeps only the reviewer's current verdict on each case.

    Label mapping: a reviewer correcting a case to "clear" is a ground-truth
    NOT-risky example (is_risky=0); correcting it to anything else (escalate,
    flag_for_compliance_review, needs_manual_review) is a ground-truth risky
    example (is_risky=1). This collapses RiskLens's 4-way decision into the
    model's binary target -- the same simplification the deterministic gate
    already makes by thresholding a single risk probability.

    Overrides whose original event can't be found, whose feature row is
    incomplete (e.g. an agent case where get_merchant_context was never
    called), or whose kyc_status/business_category isn't a value the model
    actually recognizes (e.g. a typo that reached us through the batch-CSV
    or API paths, which don't constrain those fields to a dropdown the way
    the manual-entry form does) are skipped rather than guessed at -- see
    skipped_count on the caller side if you need to report how many were
    dropped.

    That second check matters because this is exactly the situation
    find_missing_or_invalid was built for: a case with an unrecognized
    category originally failed to score at all and was routed to manual
    review with risk_score=None -- but a human reviewer can still override
    ANY case's decision (see app/dashboard.py's render_override_section),
    including one like this. Without re-validating here, that override
    would carry its invalid category straight into a retrain: since the
    value is present (not None), the None-check above wouldn't catch it,
    and transform_features would one-hot-encode it as all-zero -- a
    merchant that silently belongs to no business category at all -- the
    same silent-corruption failure mode this function's docstring already
    guards against, just reachable through this second path instead.
    """
    latest_override_by_event = {}
    for override in get_all_overrides(conn):
        # most-recent-first ordering means the first override seen here for
        # a given event_id is the reviewer's latest word on that case --
        # setdefault keeps only that one and ignores any older, superseded
        # override on the same event.
        latest_override_by_event.setdefault(override["event_id"], override)

    rows = []
    for event_id, override in latest_override_by_event.items():
        event = get_event_by_id(conn, event_id)
        if not event:
            continue
        features = _extract_features_from_event(event)
        if any(features.get(col) is None for col in RAW_REQUIRED_COLUMNS):
            continue
        if find_missing_or_invalid(pd.DataFrame([features])):
            continue
        features["is_risky"] = 0 if override["overridden_decision"] == DECISION_CLEAR else 1
        features["snapshot_date"] = override["timestamp_utc"]
        rows.append(features)
    return pd.DataFrame(rows)


def train_candidate_with_feedback(conn) -> dict:
    """
    Retrains on the ORIGINAL raw training data plus every usable human
    override collected so far, and returns metrics for both the
    currently-deployed model and this new candidate evaluated on the SAME
    held-out test split, so the comparison is apples-to-apples. Never
    touches the live model file -- see promote_candidate for that.
    """
    original_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["snapshot_date"])
    feedback_df = build_feedback_rows(conn)
    total_overrides = len(get_all_overrides(conn))

    if not feedback_df.empty:
        feedback_df["snapshot_date"] = pd.to_datetime(feedback_df["snapshot_date"], utc=True).dt.tz_localize(None)
        combined_df = pd.concat([original_df, feedback_df], ignore_index=True, sort=False)
    else:
        combined_df = original_df
    combined_df = combined_df.sort_values("snapshot_date").reset_index(drop=True)

    n = len(combined_df)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    train_df = combined_df.iloc[:train_end]
    val_df = combined_df.iloc[train_end:val_end]
    test_df = combined_df.iloc[val_end:]

    X_train, y_train = transform_features(train_df), train_df["is_risky"].values
    X_val, y_val = transform_features(val_df), val_df["is_risky"].values
    X_test, y_test = transform_features(test_df), test_df["is_risky"].values

    # Currently-deployed model, scored on this SAME test split.
    current_model = joblib.load(MODEL_PATH)
    current_threshold = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            current_threshold = json.load(f).get("xgboost_threshold", 0.5)
    current_prob = current_model.predict_proba(X_test)[:, 1]
    current_metrics = evaluate("current (deployed)", y_test, current_prob, current_threshold)

    # Candidate: same hyperparameters model/train.py uses, trained on the
    # combined (original + feedback) data.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    candidate = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
    )
    candidate.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    candidate_threshold = best_threshold_for_f1(y_val, candidate.predict_proba(X_val)[:, 1])
    candidate_prob = candidate.predict_proba(X_test)[:, 1]
    candidate_metrics = evaluate("candidate (with feedback)", y_test, candidate_prob, candidate_threshold)

    # Precomputed here (not inside promote_candidate) so promoting is just a
    # file write, not a second round of training/SHAP work -- everything a
    # promotion needs to keep the Models page's charts in sync with whatever
    # actually gets deployed is captured from the SAME candidate fit above.
    y_pred = (candidate_prob >= candidate_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, candidate_prob)
    shap_values = shap.TreeExplainer(candidate).shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_global = sorted(
        (
            {"feature": name, "mean_abs_shap": float(val)}
            for name, val in zip(X_test.columns.tolist(), mean_abs_shap)
        ),
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )

    return {
        "total_overrides": total_overrides,
        "feedback_rows_used": len(feedback_df),
        "combined_rows": n,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_model": candidate,
        "candidate_threshold": candidate_threshold,
        # Raw (pre-transform) test rows -- exactly what X_test/y_test above
        # were derived from. Passed through to promote_candidate so it can be
        # persisted alongside the model: see TEST_SNAPSHOT_PATH's comment for
        # why the dashboard needs this rather than re-deriving a test split.
        "candidate_test_df": test_df,
        "candidate_artifacts": {
            "test_rows": len(X_test),
            "test_positive_rate": float(y_test.mean()),
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": candidate_metrics["roc_auc"]},
            "shap_global_importance": shap_global,
        },
    }


def promote_candidate(
    candidate_model,
    candidate_threshold: float,
    candidate_metrics: dict,
    candidate_artifacts: dict,
    candidate_test_df: pd.DataFrame = None,
) -> None:
    """
    Explicit, human-triggered step: overwrite the live model + threshold
    with a candidate that's already been reviewed, AND refresh the Models
    page's own metrics.json/chart_data.json to match -- otherwise the page
    would keep showing the old model's confusion matrix/ROC/SHAP chart right
    after promoting a model that no longer produces them, which reads as a
    bug even though it's actually cosmetic staleness. The baseline logistic
    regression comparison is left untouched since it wasn't retrained here;
    only the XGBoost side of the report changes.

    candidate_test_df (the raw rows train_candidate_with_feedback actually
    evaluated this candidate on) is persisted to TEST_SNAPSHOT_PATH for the
    same reason: once feedback rows are mixed in and the combined data is
    re-sorted by date and re-split, that test partition is no longer the
    same rows -- or even the same row COUNT -- as an independent call to
    model/train.py's load_and_split() would produce from the original raw
    CSV alone. Without persisting it, the dashboard's Threshold Explorer
    (which needs raw per-row predictions to answer "what if the threshold
    were X", not just the single aggregate confusion matrix/ROC curve saved
    above) would keep re-deriving its own, now-mismatched test set -- so the
    same page could show two different confusion matrices for the same
    model at the same threshold, one from this file's real evaluation and
    one from that stale re-derivation. Optional (defaults to None) so any
    existing caller that hasn't been updated yet still works; the Threshold
    Explorer falls back to load_and_split() when no snapshot exists yet,
    which is exactly correct for a fresh model that was never retrained
    with feedback.

    Deliberately separate from training the candidate -- retraining always
    happens on demand, promoting only happens if a person looks at the
    before/after numbers and decides it's actually better.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Every artifact below is first written to a `.tmp_promote` file
    # alongside its real path, and only swapped into place with os.replace()
    # (atomic on the same filesystem) once ALL of them have been written
    # successfully -- see the try/except below. This used to write
    # joblib.dump(candidate_model, MODEL_PATH) directly to the final path,
    # as the very first step, followed by four more independent direct
    # writes with no rollback: a crash partway through (disk full, an
    # OOM-kill, any interrupted process) left a brand-new model permanently
    # on disk paired with a STALE decision_threshold.json/metrics.json/
    # chart_data.json from the previous model. That's not a crash a
    # reviewer would notice on the next run -- it's a silently wrong gate,
    # since a threshold tuned for one model's probability distribution gets
    # applied to a different model's scores with no error anywhere.
    metrics_report = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics_report = json.load(f)
    metrics_report["test_rows"] = candidate_artifacts["test_rows"]
    metrics_report["test_positive_rate"] = candidate_artifacts["test_positive_rate"]
    metrics_report["xgboost"] = candidate_metrics
    if "baseline_logistic_regression" in metrics_report:
        metrics_report["lift_over_baseline_f1"] = candidate_metrics["f1"] - metrics_report["baseline_logistic_regression"]["f1"]

    chart_data = {}
    if os.path.exists(CHART_DATA_PATH):
        with open(CHART_DATA_PATH) as f:
            chart_data = json.load(f)
    chart_data["confusion_matrix"] = {"labels": ["Not risky", "Risky"], "matrix": candidate_artifacts["confusion_matrix"]}
    chart_data.setdefault("roc_curve", {})["xgboost"] = candidate_artifacts["roc_curve"]
    chart_data["shap_global_importance"] = candidate_artifacts["shap_global_importance"]

    pending = {
        MODEL_PATH: ("joblib", candidate_model),
        THRESHOLD_PATH: ("json", {"xgboost_threshold": candidate_threshold}),
        METRICS_PATH: ("json", metrics_report),
        CHART_DATA_PATH: ("json", chart_data),
    }
    if candidate_test_df is not None:
        pending[TEST_SNAPSHOT_PATH] = ("csv", candidate_test_df)

    # Computed upfront (not appended to as each write succeeds) so the
    # cleanup below can find every temp path that might exist on disk --
    # including one left behind by a write that failed PARTWAY THROUGH
    # itself: `open(tmp_path, "w")` inside the json branch below already
    # creates the (empty or partially written) file before `json.dump` runs,
    # so a failure inside that `with` block leaves a real file on disk that
    # was never going to get recorded by a "track it after it succeeds"
    # dict -- the first version of this fix cleaned up only successfully
    # completed temp writes and silently left exactly that kind of orphaned
    # `.tmp_promote` file behind.
    tmp_paths = {final_path: f"{final_path}.tmp_promote" for final_path in pending}
    try:
        # Phase 1: write every artifact to a temp path. If any write fails
        # here, none of the real files have been touched yet.
        for final_path, (kind, payload) in pending.items():
            tmp_path = tmp_paths[final_path]
            if kind == "joblib":
                joblib.dump(payload, tmp_path)
            elif kind == "json":
                with open(tmp_path, "w") as f:
                    json.dump(payload, f, indent=2)
            elif kind == "csv":
                payload.to_csv(tmp_path, index=False)

        # Phase 2: every temp write succeeded -- atomically swap them all
        # into place. Model and threshold go first and back-to-back (the
        # pair whose mutual consistency actually drives gating decisions);
        # the purely-cosmetic report artifacts follow.
        for final_path in [MODEL_PATH, THRESHOLD_PATH, METRICS_PATH, TEST_SNAPSHOT_PATH, CHART_DATA_PATH]:
            if final_path in tmp_paths:
                os.replace(tmp_paths[final_path], final_path)
    except Exception:
        for tmp_path in tmp_paths.values():
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        raise
