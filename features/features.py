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


_VALID_KYC_STATUSES = {"complete", "incomplete"}
_VALID_BUSINESS_CATEGORIES = {c.lower() for c in BUSINESS_CATEGORIES}

# Every other required field is a non-negative count or amount -- a negative
# chargebacks_30d or avg_ticket_size can't come from anything real, and
# should fail safe to manual review exactly like a missing value would,
# rather than flowing into chargeback_rate/refund_rate below as a
# nonsensical negative rate the model has never seen in training.
_NUMERIC_NONNEGATIVE_COLUMNS = {
    "account_age_days",
    "daily_txn_volume",
    "avg_30d_txn_volume",
    "total_txns_30d",
    "chargebacks_30d",
    "refunds_30d",
    "avg_ticket_size",
}


def find_missing_or_invalid(df: pd.DataFrame) -> list:
    """
    Return the required raw column names that are either absent, present but
    null/empty, or -- for the two categorical fields -- present but not a
    value the model actually recognizes, for the (single-row) record in df.
    This is what the pipeline uses to decide whether it's safe to score at
    all -- it names the actual field, rather than reporting a generic
    "invalid data" reason after the fact.

    An unrecognized kyc_status/business_category (a typo, or a category the
    model was never trained on) matters here specifically because
    transform_features one-hot-encodes these fields: a value that doesn't
    match any known category silently produces an all-zero encoding rather
    than a NaN, which would otherwise slip past this check and the model
    would score it anyway as if no category applied -- a wrong answer with
    no warning, which is worse than failing safe to manual review. Casing
    differences (e.g. "Complete", "ELECTRONICS") are normalized here and in
    transform_features rather than rejected, since those aren't actually
    invalid data, just differently formatted valid data.

    The categorical checks compare str(val) rather than requiring val
    already be a str -- an earlier version only checked kyc_status/
    business_category when isinstance(val, str) was true, so a
    business_category of 42 or a kyc_status of 1 (a hallucinated tool
    argument, or a batch CSV cell pandas parsed as a number) skipped the
    check entirely and reached the same silent all-zero encoding this
    function exists to prevent. Numeric fields get the equivalent
    treatment: a value that can't convert to a finite, non-negative number
    is flagged too, instead of being coerced downstream into a negative
    rate with no warning.
    """
    problems = []
    for col in RAW_REQUIRED_COLUMNS:
        if col not in df.columns:
            problems.append(col)
            continue
        val = df[col].iloc[0] if len(df) else None
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            problems.append(col)
            continue
        if isinstance(val, str) and val.strip() == "":
            problems.append(col)
            continue
        if col == "kyc_status":
            if str(val).strip().lower() not in _VALID_KYC_STATUSES:
                problems.append(col)
            continue
        if col == "business_category":
            if str(val).strip().lower() not in _VALID_BUSINESS_CATEGORIES:
                problems.append(col)
            continue
        if col in _NUMERIC_NONNEGATIVE_COLUMNS:
            try:
                num = float(val)
            except (TypeError, ValueError):
                problems.append(col)
                continue
            if not np.isfinite(num) or num < 0:
                problems.append(col)
    # Cross-field check: chargebacks/refunds can never exceed the total
    # transaction count they're drawn from. Each field passes the
    # individual non-negative check above on its own (e.g.
    # chargebacks_30d=500, total_txns_30d=1 are each independently a valid
    # non-negative number), but together they produce a chargeback_rate
    # far outside anything the model saw in training -- the same class of
    # "value the model has never seen" problem that
    # agent/merchant_context.py already guards against for *simulated*
    # merchants, but which nothing previously caught for a real batch
    # upload row, manual investigation form entry, or agent tool call.
    if "chargebacks_30d" not in problems and "total_txns_30d" not in problems:
        try:
            if float(df["chargebacks_30d"].iloc[0]) > float(df["total_txns_30d"].iloc[0]):
                problems.append("chargebacks_30d")
        except (TypeError, ValueError):
            pass
    if "refunds_30d" not in problems and "total_txns_30d" not in problems:
        try:
            if float(df["refunds_30d"].iloc[0]) > float(df["total_txns_30d"].iloc[0]):
                problems.append("refunds_30d")
        except (TypeError, ValueError):
            pass
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
    # Case/whitespace-normalized before comparing -- "Complete" or " complete "
    # is still valid input (find_missing_or_invalid already rejected anything
    # that isn't recognized even after normalizing), it just shouldn't be
    # silently scored as "incomplete" only because of formatting.
    kyc_norm = kyc.astype(str).str.strip().str.lower()
    out["kyc_complete"] = (kyc_norm == "complete").astype(float)
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
    # Same case/whitespace normalization as kyc_status above, and for the
    # same reason: a differently-formatted but valid category (e.g.
    # "Electronics") should still one-hot-encode correctly instead of
    # matching none of the known categories and silently scoring as if no
    # category applied at all.
    category_norm = category.astype(str).str.strip().str.lower()
    for c in BUSINESS_CATEGORIES:
        out[f"category_{c}"] = (category_norm == c).astype(float)
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
