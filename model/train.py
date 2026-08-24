"""
Train and evaluate the risk model.

- Time-based train/test split (train on earlier snapshots, test on later
  ones) rather than a random split, because a risk model should be judged
  on its ability to generalize forward in time, not just to held-out rows
  scattered across the same period.
- Trains an XGBoost classifier plus a logistic regression baseline, so the
  metrics report shows the lift XGBoost provides over a simple model.
- Saves: the trained model, the feature manifest, a metrics.json report,
  and evaluation plots (confusion matrix, ROC curve, SHAP summary) used by
  the dashboard's "Model performance" tab.

Run:
    python3 model/train.py
"""

import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.features import FEATURE_COLUMNS, save_feature_manifest, transform_features

RAW_DATA_PATH = "data/raw/merchant_snapshots.csv"
ARTIFACT_DIR = "model/artifacts"
# Time-ordered 3-way split: train on the earliest data, tune the decision
# threshold on validation, report final numbers on test -- so the reported
# metrics aren't inflated by picking a threshold that happens to fit the
# test set.
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2  # remaining 0.2 is test


def load_and_split():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["snapshot_date"])
    df = df.sort_values("snapshot_date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    splits = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        splits[name] = (transform_features(split_df), split_df["is_risky"].values)
    return splits


def best_threshold_for_f1(y_true, y_prob):
    """Pick the probability threshold that maximizes F1 on the given (validation) set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    print(f"\n[{name}] threshold={threshold:.3f} (tuned on validation set)")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    return metrics


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    save_feature_manifest(os.path.join(ARTIFACT_DIR, "feature_manifest.json"))

    splits = load_and_split()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)} | Test rows: {len(X_test)}")
    print(
        f"Train positive rate: {y_train.mean():.3f} | "
        f"Val positive rate: {y_val.mean():.3f} | "
        f"Test positive rate: {y_test.mean():.3f}"
    )

    # --- Baseline: logistic regression ------------------------------------
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train.fillna(0), y_train)
    baseline_val_prob = baseline.predict_proba(X_val.fillna(0))[:, 1]
    baseline_threshold = best_threshold_for_f1(y_val, baseline_val_prob)
    baseline_test_prob = baseline.predict_proba(X_test.fillna(0))[:, 1]
    baseline_metrics = evaluate("LogisticRegression (baseline)", y_test, baseline_test_prob, baseline_threshold)

    # --- Main model: XGBoost, with early stopping against the validation set
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"XGBoost stopped at {model.best_iteration + 1} trees (early stopping on validation aucpr)")

    xgb_val_prob = model.predict_proba(X_val)[:, 1]
    xgb_threshold = best_threshold_for_f1(y_val, xgb_val_prob)
    xgb_test_prob = model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate("XGBoost", y_test, xgb_test_prob, xgb_threshold)

    # --- Save model + metrics ------------------------------------------------
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "xgb_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "decision_threshold.json"), "w") as f:
        json.dump({"xgboost_threshold": xgb_threshold}, f, indent=2)

    metrics_report = {
        "test_rows": len(X_test),
        "test_positive_rate": float(y_test.mean()),
        "baseline_logistic_regression": baseline_metrics,
        "xgboost": xgb_metrics,
        "lift_over_baseline_f1": xgb_metrics["f1"] - baseline_metrics["f1"],
    }
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)
    print("\nSaved metrics report to model/artifacts/metrics.json")

    # --- Plots for the dashboard's "Model performance" tab ------------------
    y_pred = (xgb_test_prob >= xgb_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Not risky", "Risky"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix (XGBoost, threshold={xgb_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_test, xgb_test_prob, ax=ax)
    ax.set_title("ROC Curve (XGBoost)")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig = plt.figure(figsize=(6, 4))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots: confusion_matrix.png, roc_curve.png, shap_summary.png")

    # --- Raw chart data for the dashboard's interactive (Altair) charts -----
    # The dashboard renders its own hoverable/tooltipped charts rather than
    # embedding these static PNGs -- it needs the underlying numbers, not
    # just the pictures. Every value here is computed directly from the
    # held-out test set, same as the PNGs above; nothing is invented.
    baseline_fpr, baseline_tpr, _ = roc_curve(y_test, baseline_test_prob)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_test_prob)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_global = sorted(
        (
            {"feature": name, "mean_abs_shap": float(val)}
            for name, val in zip(X_test.columns.tolist(), mean_abs_shap)
        ),
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )

    chart_data = {
        "confusion_matrix": {
            "labels": ["Not risky", "Risky"],
            # rows = actual, columns = predicted, same convention as sklearn's confusion_matrix
            "matrix": cm.tolist(),
        },
        "roc_curve": {
            "xgboost": {"fpr": xgb_fpr.tolist(), "tpr": xgb_tpr.tolist(), "auc": xgb_metrics["roc_auc"]},
            "baseline_logistic_regression": {
                "fpr": baseline_fpr.tolist(),
                "tpr": baseline_tpr.tolist(),
                "auc": baseline_metrics["roc_auc"],
            },
        },
        "shap_global_importance": shap_global,
    }
    with open(os.path.join(ARTIFACT_DIR, "chart_data.json"), "w") as f:
        json.dump(chart_data, f, indent=2)
    print("Saved model/artifacts/chart_data.json (for interactive dashboard charts)")

    print("\nDone. Model + artifacts are in model/artifacts/")


if __name__ == "__main__":
    main()
