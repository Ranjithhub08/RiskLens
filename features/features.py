"""
Shared feature engineering.

Both training (model/train.py) and inference (used by the dashboard/API)
import `transform_features` from here, so the model is always scored on
data shaped exactly the way it was trained on. This is the single most
common source of silent bugs in ML systems (train/inference feature
skew), so we deliberately have only one implementation.
"""

import json
import os
import numpy as np
import pandas as pd

EPS = 1e-6

# Fixed category list, saved alongside the model, so one-hot encoding is
# identical at train time and inference time even if a batch at inference
# happens to be missing a category.
BUSINESS_CATEGORIES = ["electronics", "fashion", "services", "travel", "subscriptions", "other"]

RAW_REQUIRED_COLUMNS = [
    "account_age_days",
    "kyc_status",
    "business_category",
    "daily_txn_volume",
    "avg_30d_txn_volume",
    "total_txns_30d",
    "chargebacks_30d",
    "refunds_30d",
    "avg_ticket_size",
]

FEATURE_COLUMNS = [
    "account_age_days",
    "kyc_complete",
    "volume_change_pct",
    "chargeback_rate",
    "refund_rate",
    "avg_ticket_size",
] + [f"category_{c}" for c in BUSINESS_CATEGORIES]


class MissingFieldsError(ValueError):
    """Raised when a raw record is missing fields required for scoring."""

    def __init__(self, missing_fields):
        self.missing_fields = missing_fields
        super().__init__(f"Missing required fields: {missing_fields}")


def validate_raw(df: pd.DataFrame) -> list:
    """Return the list of required columns missing from df (empty list = OK)."""
    return [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]


def find_missing_or_invalid(df: pd.DataFrame) -> list:
    """
    Return the required raw column names that are either absent, or present
    but null/empty, for the (single-row) record in df. This is what the
    pipeline uses to decide whether it's safe to score at all -- it names
    the actual field, rather than reporting a generic "invalid data" reason
    after the fact.
    """
    problems = []
    for col in RAW_REQUIRED_COLUMNS:
        if col not in df.columns:
            problems.append(col)
            continue
        val = df[col].iloc[0] if len(df) else None
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            problems.append(col)
        elif isinstance(val, str) and val.strip() == "":
            problems.append(col)
    return problems


def _col_or_nan(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if present, else a NaN-filled Series with df's index.

    Using plain `df.get(name, np.nan)` is a trap: when the column is absent,
    pandas returns the scalar np.nan rather than a Series, which breaks any
    subsequent .astype()/arithmetic call. This helper always returns a Series.
    """
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index)


def transform_features(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Turn raw merchant snapshot rows into the model's numeric feature matrix.

    If `strict` is True (default), raises MissingFieldsError when required
    raw columns are absent -- this is what lets the gating layer route
    incomplete records to "needs manual review" instead of silently
    producing a bogus score.
    """
    missing = validate_raw(df)
    if missing and strict:
        raise MissingFieldsError(missing)

    out = pd.DataFrame(index=df.index)

    out["account_age_days"] = pd.to_numeric(_col_or_nan(df, "account_age_days"), errors="coerce")

    if "kyc_status" in df.columns:
        kyc = df["kyc_status"]
    else:
        kyc = pd.Series([None] * len(df), index=df.index)
    out["kyc_complete"] = (kyc == "complete").astype(float)
    out.loc[kyc.isna(), "kyc_complete"] = np.nan

    daily_vol = pd.to_numeric(_col_or_nan(df, "daily_txn_volume"), errors="coerce")
    avg_vol = pd.to_numeric(_col_or_nan(df, "avg_30d_txn_volume"), errors="coerce")
    out["volume_change_pct"] = (daily_vol - avg_vol) / (avg_vol.abs() + EPS)

    total_txns = pd.to_numeric(_col_or_nan(df, "total_txns_30d"), errors="coerce").clip(lower=1)
    chargebacks = pd.to_numeric(_col_or_nan(df, "chargebacks_30d"), errors="coerce")
    refunds = pd.to_numeric(_col_or_nan(df, "refunds_30d"), errors="coerce")
    out["chargeback_rate"] = chargebacks / total_txns
    out["refund_rate"] = refunds / total_txns

    out["avg_ticket_size"] = pd.to_numeric(_col_or_nan(df, "avg_ticket_size"), errors="coerce")

    if "business_category" in df.columns:
        category = df["business_category"]
    else:
        category = pd.Series([None] * len(df), index=df.index)
    for c in BUSINESS_CATEGORIES:
        out[f"category_{c}"] = (category == c).astype(float)
        out.loc[category.isna(), f"category_{c}"] = np.nan

    # Any row with NaNs after transformation (from bad/missing raw values that
    # weren't caught by validate_raw, e.g. an empty string) is left as NaN on
    # purpose -- the gating layer checks for this and routes to manual review
    # rather than the model silently treating NaN as zero.
    out = out[FEATURE_COLUMNS]
    return out


def save_feature_manifest(path: str = "model/artifacts/feature_manifest.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"feature_columns": FEATURE_COLUMNS, "business_categories": BUSINESS_CATEGORIES},
            f,
            indent=2,
        )
