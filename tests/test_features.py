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


def test_find_missing_or_invalid_flags_non_string_category_and_kyc():
    # A hallucinated tool argument or a batch-CSV cell pandas parsed as a
    # number is just as unrecognized as a mistyped string -- an earlier
    # version of find_missing_or_invalid only checked kyc_status/
    # business_category when isinstance(val, str) was true, so a non-string
    # value like this skipped the check entirely and reached
    # transform_features' one-hot encoding, which silently produces an
    # all-zero row for it instead of raising.
    row = dict(VALID_ROW, kyc_status=1, business_category=42)
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "kyc_status" in problems
    assert "business_category" in problems


def test_find_missing_or_invalid_flags_negative_numeric_fields():
    # chargebacks_30d/refunds_30d/etc. are counts and amounts -- negative
    # values can't come from anything real and would otherwise flow into
    # chargeback_rate/refund_rate as a nonsensical negative rate with no
    # warning.
    row = dict(VALID_ROW, chargebacks_30d=-10, avg_ticket_size=-1.0)
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "chargebacks_30d" in problems
    assert "avg_ticket_size" in problems


def test_find_missing_or_invalid_flags_non_numeric_string_in_numeric_field():
    row = dict(VALID_ROW, total_txns_30d="not-a-number")
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "total_txns_30d" in problems


def test_find_missing_or_invalid_allows_zero_numeric_fields():
    # Zero is a legitimate value (a brand-new merchant with 0 chargebacks,
    # or 0 days old) -- only negative/non-finite/non-numeric should be
    # rejected.
    row = dict(VALID_ROW, chargebacks_30d=0, account_age_days=0)
    df = pd.DataFrame([row])
    assert find_missing_or_invalid(df) == []


def test_find_missing_or_invalid_empty_on_valid_row():
    df = pd.DataFrame([VALID_ROW])
    assert find_missing_or_invalid(df) == []


def test_find_missing_or_invalid_flags_chargebacks_exceeding_total_txns():
    # Each field is individually a valid non-negative number, but
    # chargebacks_30d > total_txns_30d is impossible -- you can't have more
    # chargebacks than transactions. Without a cross-field check this would
    # flow straight into chargeback_rate as e.g. 500.0 (50,000%), a value
    # nowhere near anything the model saw in training.
    row = dict(VALID_ROW, chargebacks_30d=500, total_txns_30d=1)
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "chargebacks_30d" in problems
    assert "total_txns_30d" not in problems  # total_txns_30d=1 is valid on its own


def test_find_missing_or_invalid_flags_refunds_exceeding_total_txns():
    row = dict(VALID_ROW, refunds_30d=50, total_txns_30d=3)
    df = pd.DataFrame([row])
    problems = find_missing_or_invalid(df)
    assert "refunds_30d" in problems


def test_find_missing_or_invalid_allows_chargebacks_equal_to_total_txns():
    # Every single transaction charging back is unusual but not impossible
    # -- equal should be allowed, only strictly greater is invalid.
    row = dict(VALID_ROW, chargebacks_30d=3, total_txns_30d=3, refunds_30d=0)
    df = pd.DataFrame([row])
    assert find_missing_or_invalid(df) == []


def test_find_missing_or_invalid_flags_absurdly_large_finite_numeric_value():
    # A finite, non-negative float like 1e40 is otherwise a "valid" number,
    # but it's many orders of magnitude past anything real and past
    # float32's safe range -- XGBoost's predict_proba raises an uncaught
    # error on a value this large rather than a graceful NaN/prediction.
    row = dict(VALID_ROW, account_age_days=1e40)
    df = pd.DataFrame([row])
    assert "account_age_days" in find_missing_or_invalid(df)


def test_find_missing_or_invalid_flags_integer_too_large_for_a_float():
    # A Python int with hundreds of digits overflows a C double: float(val)
    # raises OverflowError, not TypeError/ValueError -- this must be caught
    # by the same except clause as any other malformed numeric value.
    row = dict(VALID_ROW, chargebacks_30d=int("9" * 400))
    df = pd.DataFrame([row], dtype=object)
    assert "chargebacks_30d" in find_missing_or_invalid(df)


def test_find_missing_or_invalid_flags_list_valued_field_without_crashing():
    # A hallucinated/malformed caller (e.g. an LLM tool call) could hand a
    # list where a scalar is expected. pd.isna() on a list/array returns an
    # array rather than a bool, and evaluating an array's truthiness raises
    # ValueError -- this must be rejected as invalid, not crash.
    row = dict(VALID_ROW, account_age_days=[1, 2, 3])
    df = pd.DataFrame([row], dtype=object)
    assert "account_age_days" in find_missing_or_invalid(df)


def test_find_missing_or_invalid_allows_realistic_large_values():
    # A very large but entirely plausible merchant (huge daily volume,
    # long-running account) must not get caught by the new upper bound.
    row = dict(VALID_ROW, account_age_days=10000, daily_txn_volume=5_000_000.0, avg_ticket_size=250_000.0)
    df = pd.DataFrame([row])
    assert find_missing_or_invalid(df) == []
