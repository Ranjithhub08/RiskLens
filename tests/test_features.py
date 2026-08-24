import pandas as pd
import pytest

from features.features import (
    FEATURE_COLUMNS,
    MissingFieldsError,
    find_missing_or_invalid,
    transform_features,
    validate_raw,
)

VALID_ROW = {
    "merchant_id": 1,
    "account_age_days": 400,
    "kyc_status": "complete",
    "business_category": "fashion",
    "daily_txn_volume": 10000.0,
    "avg_30d_txn_volume": 9000.0,
    "total_txns_30d": 300,
    "chargebacks_30d": 2,
    "refunds_30d": 10,
    "avg_ticket_size": 33.3,
}


def test_transform_features_produces_expected_columns():
    df = pd.DataFrame([VALID_ROW])
    out = transform_features(df)
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(out) == 1


def test_transform_features_no_unexpected_nans_on_valid_input():
    df = pd.DataFrame([VALID_ROW])
    out = transform_features(df)
    assert not out.isna().any(axis=None)


def test_volume_change_pct_calculation():
    df = pd.DataFrame([VALID_ROW])
    out = transform_features(df)
    expected = (10000.0 - 9000.0) / 9000.0
    assert out["volume_change_pct"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_chargeback_rate_calculation():
    df = pd.DataFrame([VALID_ROW])
    out = transform_features(df)
    expected = 2 / 300
    assert out["chargeback_rate"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_kyc_encoding():
    complete = pd.DataFrame([VALID_ROW])
    incomplete_row = dict(VALID_ROW, kyc_status="incomplete")
    incomplete = pd.DataFrame([incomplete_row])

    assert transform_features(complete)["kyc_complete"].iloc[0] == 1.0
    assert transform_features(incomplete)["kyc_complete"].iloc[0] == 0.0


def test_business_category_one_hot():
    df = pd.DataFrame([VALID_ROW])  # category = "fashion"
    out = transform_features(df)
    assert out["category_fashion"].iloc[0] == 1.0
    assert out["category_electronics"].iloc[0] == 0.0


def test_validate_raw_detects_missing_columns():
    df = pd.DataFrame([{"merchant_id": 1, "account_age_days": 400}])
    missing = validate_raw(df)
    assert "kyc_status" in missing
    assert "daily_txn_volume" in missing


def test_transform_features_strict_raises_on_missing_columns():
    df = pd.DataFrame([{"merchant_id": 1}])
    with pytest.raises(MissingFieldsError):
        transform_features(df, strict=True)


def test_transform_features_non_strict_does_not_raise():
    df = pd.DataFrame([{"merchant_id": 1}])
    out = transform_features(df, strict=False)
    # Missing numeric inputs should surface as NaN, not crash the pipeline.
    assert out.isna().any(axis=None)


def test_find_missing_or_invalid_flags_absent_column():
    df = pd.DataFrame([{k: v for k, v in VALID_ROW.items() if k != "kyc_status"}])
    problems = find_missing_or_invalid(df)
    assert "kyc_status" in problems


def test_find_missing_or_invalid_flags_null_value():
    row = dict(VALID_ROW, kyc_status=None)
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "kyc_status" in problems


def test_find_missing_or_invalid_flags_empty_string():
    row = dict(VALID_ROW, business_category="")
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "business_category" in problems


def test_find_missing_or_invalid_empty_on_valid_row():
    df = pd.DataFrame([VALID_ROW])
    assert find_missing_or_invalid(df) == []
