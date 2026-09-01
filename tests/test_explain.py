"""
Tests for explainability/explain.py's plain-language phrasing.

Regression coverage for a bug found in the 6th review round: _phrase_for
used to pick a feature's "elevated"/"low" (etc.) WORDING purely from that
row's SHAP sign. SHAP sign reflects a feature's marginal contribution given
this row's specific combination of every other feature -- it frequently
disagrees with whether the feature's raw value is actually high or low, so
the old code produced factually false claims like "the chargeback rate is
elevated" for a merchant whose chargeback rate was near the bottom of the
observed range. Measured against the real trained model, ~31% of the real
6000-row dataset got at least one such factually-contradicted phrase.

The fix decouples the two questions: shap_value still (and only) decides
which heading a factor is listed under (raising vs. lowering the score);
the phrase's actual wording is grounded in the feature's own value via
_FEATURE_IS_ELEVATED.
"""

import joblib
import pandas as pd
import pytest

from explainability.explain import RiskExplainer, _phrase_for
from features.features import transform_features

MODEL_PATH = "model/artifacts/xgb_model.joblib"


def test_phrase_for_uses_raw_value_not_shap_sign_for_wording():
    # A LOW chargeback rate (0.4%, well under the 2% "elevated" threshold
    # data/raw/generate_data.py's own label rule uses) with a POSITIVE
    # SHAP value used to be worded "elevated" -- factually false.
    phrase = _phrase_for("chargeback_rate", shap_value=0.05, raw_value=0.004)
    assert "low" in phrase
    assert "elevated" not in phrase


def test_phrase_for_uses_raw_value_not_shap_sign_for_wording_other_direction():
    # A HIGH refund rate (28.6%, well over the 8% "elevated" threshold)
    # with a NEGATIVE SHAP value used to be worded "low" -- also false,
    # and in the dangerous direction (understating real risk).
    phrase = _phrase_for("refund_rate", shap_value=-0.03, raw_value=0.286)
    assert "elevated" in phrase
    assert phrase != "the refund rate is low"


def test_phrase_for_volume_spike_wording_matches_actual_value():
    spiked = _phrase_for("volume_change_pct", shap_value=-0.1, raw_value=1.5)
    assert "spiked" in spiked

    normal = _phrase_for("volume_change_pct", shap_value=0.1, raw_value=0.05)
    assert "normal pattern" in normal


def test_phrase_for_falls_back_to_shap_sign_when_raw_value_missing():
    # No raw_value provided (e.g. an unexpected caller) -- must not crash,
    # and falls back to the old SHAP-sign-only behavior rather than
    # erroring.
    phrase = _phrase_for("chargeback_rate", shap_value=0.05, raw_value=None)
    assert "elevated" in phrase


def test_phrase_for_category_features_unaffected_by_the_fix():
    # Category features make an attribution claim ("associated with
    # higher risk"), not a factual claim about a measured value -- these
    # correctly stay tied to SHAP sign and shouldn't be touched by the
    # raw-value wording fix.
    phrase = _phrase_for("category_electronics", shap_value=0.2, raw_value=1.0)
    assert "associated with higher risk" in phrase


@pytest.mark.skipif(not __import__("os").path.exists(MODEL_PATH), reason="requires a trained model artifact")
def test_no_contradicted_elevated_or_low_phrase_across_a_real_sample():
    # Integration-level regression test against the actual shipped model:
    # for every top-factor phrase mentioning "elevated" or "is low" for a
    # feature with a known threshold, the feature's own raw value must
    # actually be on that side of the threshold -- i.e. the wording must
    # never contradict the real data, regardless of what SHAP said.
    from explainability.explain import _FEATURE_IS_ELEVATED

    model = joblib.load(MODEL_PATH)
    explainer = RiskExplainer(model)

    rows = [
        {
            "merchant_id": i, "account_age_days": 200 + i * 7, "kyc_status": "complete",
            "business_category": "services", "daily_txn_volume": 9000.0 + i * 500,
            "avg_30d_txn_volume": 9000.0, "total_txns_30d": 300, "chargebacks_30d": i % 5,
            "refunds_30d": (i * 2) % 20, "avg_ticket_size": 30.0 + i,
        }
        for i in range(30)
    ]
    df = pd.DataFrame(rows)
    X = transform_features(df)

    for i in range(len(X)):
        result = explainer.explain_row(X.iloc[[i]])
        for factor in result["top_factors"]:
            name = factor["feature"]
            if name not in _FEATURE_IS_ELEVATED:
                continue
            raw_value = X[name].iloc[i]
            is_elevated = _FEATURE_IS_ELEVATED[name](raw_value)
            phrase = _phrase_for(name, factor["shap_value"], raw_value)
            if "elevated" in phrase or "spiked" in phrase or "incomplete" in phrase or "new" in phrase or "unusually large" in phrase:
                assert is_elevated, f"{name}={raw_value} worded as elevated but isn't"
            if "is low" in phrase or "normal pattern" in phrase or "established" in phrase or "typical" in phrase:
                assert not is_elevated, f"{name}={raw_value} worded as low/typical but is actually elevated"
