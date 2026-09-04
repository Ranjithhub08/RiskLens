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

Follow-up bug found live in a later round: decoupling those two questions
opened a new one -- the wording and the bucket can now point opposite ways
for the *same* factor (a genuine volume spike with a small net
risk-REDUCING SHAP contribution, for instance), producing a flatly
self-contradictory sentence like "factors lowering the risk score:
transaction volume has spiked well above this merchant's own 30-day
average". Confirmed this happens on ~6% of top-factor mentions even with a
correctly-trained model, not just a stale/undertrained one.
RiskExplainer.explain_row now calls this out explicitly via
_factor_phrase's "unusual combination" caveat instead of silently
producing a phrase that contradicts its own heading.
"""

import joblib
import pandas as pd
import pytest

from explainability.explain import RiskExplainer, _factor_phrase, _phrase_for
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


def test_factor_phrase_flags_a_spike_whose_shap_sign_disagrees_with_its_raw_value():
    # Found live: a genuine volume spike (raw_value well past the 0.8
    # "elevated" cutoff) can carry a NEGATIVE SHAP contribution in
    # combination with a merchant's other signals -- _phrase_for correctly
    # words that as "spiked" (grounded in the raw value, not the SHAP sign,
    # per its own docstring), but explain_row used to then file that exact
    # phrase under "factors lowering the risk score" with no acknowledgment
    # of the tension -- a flatly self-contradictory sentence ("... lowering
    # the risk score: transaction volume has spiked...") that reads as a
    # bug even though every individual fact in it is true. _factor_phrase
    # must call out the conflict instead of silently returning a phrase
    # that contradicts the bucket it's about to be filed under.
    phrase = _factor_phrase("volume_change_pct", shap_value=-0.1, raw_value=1.5)
    assert "spiked" in phrase
    assert "unusual combination" in phrase


def test_factor_phrase_flags_the_opposite_direction_conflict_too():
    # A raw value that's clearly NOT elevated (in line with normal pattern)
    # but carries a POSITIVE SHAP value (pushing risk up) is the same
    # conflict in the other direction -- must also be flagged.
    phrase = _factor_phrase("volume_change_pct", shap_value=0.1, raw_value=0.05)
    assert "normal pattern" in phrase
    assert "unusual combination" in phrase


def test_factor_phrase_does_not_flag_when_wording_and_shap_sign_agree():
    # Sanity check the flag isn't just always-on: when the raw value and
    # SHAP sign point the same way, no caveat should be added.
    spiking_and_raising = _factor_phrase("volume_change_pct", shap_value=0.2, raw_value=1.5)
    assert "unusual combination" not in spiking_and_raising

    normal_and_lowering = _factor_phrase("volume_change_pct", shap_value=-0.2, raw_value=0.05)
    assert "unusual combination" not in normal_and_lowering


def test_factor_phrase_is_unaffected_for_features_with_no_elevated_concept():
    # category_* features have no _FEATURE_IS_ELEVATED entry (their phrase
    # is an attribution claim, not a factual one -- see _phrase_for) and so
    # can never conflict; must never carry the caveat.
    phrase = _factor_phrase("category_electronics", shap_value=0.2, raw_value=1.0)
    assert "unusual combination" not in phrase


def test_explain_row_surfaces_the_caveat_from_a_real_scored_case():
    # End-to-end check against the exact scenario that surfaced this bug
    # live: a merchant whose transaction is a ~30x spike over its own
    # 30-day average. Whichever way this particular model's SHAP happens
    # to sign that feature, the resulting sentence must be internally
    # consistent -- either the spike phrase and its bucket agree, or the
    # conflict is called out explicitly. It must never silently contradict
    # itself.
    model = joblib.load(MODEL_PATH)
    explainer = RiskExplainer(model)

    row = pd.DataFrame([{
        "merchant_id": "m1", "account_age_days": 682, "kyc_status": "complete",
        "business_category": "travel", "daily_txn_volume": 100000.0,
        "avg_30d_txn_volume": 3290.04, "total_txns_30d": 2078,
        "chargebacks_30d": 8, "refunds_30d": 31, "avg_ticket_size": 3007.65,
    }])
    X = transform_features(row)
    result = explainer.explain_row(X)

    volume_change_pct = X["volume_change_pct"].iloc[0]
    assert volume_change_pct > 0.8, "test fixture must actually be a genuine spike"

    volume_factor = next(f for f in result["top_factors"] if f["feature"] == "volume_change_pct")
    if volume_factor["shap_value"] < 0:
        assert "unusual combination" in result["explanation"]
    assert "spiked" in result["explanation"]


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
