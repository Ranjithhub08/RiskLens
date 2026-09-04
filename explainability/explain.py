"""
Per-prediction explainability.

Wraps SHAP's TreeExplainer so any single case can be explained, and
translates the top contributing features into a plain-language sentence --
this is what turns "risk score: 0.81" into something a merchant or a
reviewer can actually act on.
"""

from typing import Optional

import numpy as np
import pandas as pd
import shap

# Human-readable phrasing for each engineered feature, in both directions
# (pushed risk up / pushed risk down). Kept as a lookup table rather than
# generated dynamically, so the wording is predictable and reviewable --
# an important property for a system whose whole point is explainability.
FEATURE_PHRASES = {
    "account_age_days": {
        "up": "the account is relatively new",
        "down": "the account has an established history",
    },
    "kyc_complete": {
        "up": "KYC documentation is incomplete",
        "down": "KYC documentation is complete",
    },
    "volume_change_pct": {
        "up": "transaction volume has spiked well above this merchant's own 30-day average",
        "down": "transaction volume is in line with this merchant's normal pattern",
    },
    "chargeback_rate": {
        "up": "the chargeback rate over the last 30 days is elevated",
        "down": "the chargeback rate is low",
    },
    "refund_rate": {
        "up": "the refund rate over the last 30 days is elevated",
        "down": "the refund rate is low",
    },
    "avg_ticket_size": {
        "up": "the average transaction size is unusually large",
        "down": "the average transaction size is typical",
    },
}


# Whether a feature's OWN VALUE counts as "elevated"/"new"/"spiked" etc.,
# independent of any single row's SHAP sign -- see _phrase_for's docstring
# for why these must be decoupled. Matched to data/raw/generate_data.py's
# own labeling rule where one exists (volume_change_pct, chargeback_rate,
# refund_rate, account_age_days, kyc_complete); avg_ticket_size isn't part
# of that rule (the label doesn't depend on it at all), so its threshold is
# instead the ~90th percentile actually observed in the real training data
# (data/raw/merchant_snapshots.csv's avg_ticket_size distribution).
_FEATURE_IS_ELEVATED = {
    "account_age_days": lambda v: v < 60,
    "kyc_complete": lambda v: v < 0.5,
    "volume_change_pct": lambda v: v > 0.8,
    "chargeback_rate": lambda v: v > 0.02,
    "refund_rate": lambda v: v > 0.08,
    "avg_ticket_size": lambda v: v > 2500,
}


def _wording_direction(feature_name: str, raw_value: float = None) -> Optional[str]:
    """Which way a feature's phrase reads on its own facts -- "up" for
    elevated/spiked/new/incomplete/unusually-large, "down" for low/typical/
    established/complete -- grounded in the feature's own raw value via
    _FEATURE_IS_ELEVATED, independent of any single row's SHAP sign. None
    for a feature with no such lookup entry (raw value missing/non-finite,
    or a feature -- like business category -- whose phrase is an
    attribution claim rather than a factual one, and so is tied to SHAP
    sign instead; see _phrase_for).
    """
    if feature_name not in _FEATURE_IS_ELEVATED:
        return None
    if raw_value is None or not np.isfinite(raw_value):
        return None
    return "up" if _FEATURE_IS_ELEVATED[feature_name](raw_value) else "down"


def _phrase_for(feature_name: str, shap_value: float, raw_value: float = None) -> str:
    """
    shap_value decides which heading a factor appears under (raising vs.
    lowering the score) -- that's a correct use of SHAP sign, since it's
    exactly what "did this factor push the score up or down" means.

    But SHAP sign must NOT be used to decide the WORDING of the phrase
    itself (whether to say "elevated"/"new"/"spiked" vs. "low"/
    "established"/"typical") -- a feature's SHAP value reflects its
    marginal contribution to THIS row's score GIVEN this row's specific
    combination of every other feature, on a model trained on
    deliberately noisy synthetic data (see gating/decision_engine.py's own
    comment that the training data "only weakly separates risky from
    clean merchants"). That frequently disagrees with whether the raw
    value is actually high or low -- e.g. a chargeback_rate of 0.4% (near
    the bottom of the real range) can still carry a positive SHAP value in
    combination with other features, and the old code would have called
    that "elevated" -- a factually false, and in the opposite-of-low-risk
    direction, misleading statement for the reviewer relying on this as
    the system's core explainability output. So the phrase's factual claim
    is grounded in the feature's own value via _FEATURE_IS_ELEVATED
    instead, while shap_value still and only controls bucket placement.
    """
    direction = "up" if shap_value > 0 else "down"
    if feature_name in FEATURE_PHRASES:
        wording_direction = _wording_direction(feature_name, raw_value)
        wording = wording_direction if wording_direction is not None else direction
        return FEATURE_PHRASES[feature_name][wording]
    if feature_name.startswith("category_"):
        category = feature_name.replace("category_", "")
        if direction == "up":
            return f"the business category ('{category}') is statistically associated with higher risk"
        return f"the business category ('{category}') is not a risk driver here"
    return f"'{feature_name}' contributed to the {'higher' if direction == 'up' else 'lower'} score"


def _factor_phrase(feature_name: str, shap_value: float, raw_value: float = None) -> str:
    """The final display phrase for one top factor, including a caveat when
    the raw-value-grounded wording _phrase_for produces conflicts with the
    bucket ("raising"/"lowering") this row's SHAP sign is about to file it
    under.

    _phrase_for's wording is deliberately grounded in the feature's own raw
    value rather than this row's SHAP sign (see its docstring -- that's
    what keeps it from calling a near-zero chargeback rate "elevated" just
    because it happened to carry a positive SHAP value here). But that
    means the wording and the bucket can now point opposite ways for the
    same factor: a genuine volume spike, in combination with this
    merchant's other signals, can still carry a small NEGATIVE net
    contribution to this particular score. Left alone, that produces a
    flatly self-contradictory sentence -- "factors lowering the risk
    score: transaction volume has spiked well above this merchant's own
    30-day average" -- which is exactly what surfaced in a live demo and
    reads as a bug even though every individual fact in it is true.
    Confirmed this isn't limited to a stale/undertrained model either:
    ~6% of top-factor mentions on a correctly-retrained model hit this
    same conflict on a 2000-row sample. Rather than hide the tension by
    picking one side, say both facts plainly.

    Factored out of RiskExplainer.explain_row so this logic is directly
    testable with synthetic (feature, shap_value, raw_value) inputs,
    without needing a real trained model + SHAP run to manufacture a
    conflicting row.
    """
    phrase = _phrase_for(feature_name, shap_value, raw_value)
    wording_direction = _wording_direction(feature_name, raw_value)
    bucket_direction = "up" if shap_value > 0 else "down"
    if wording_direction is not None and wording_direction != bucket_direction:
        phrase = f"{phrase} (an unusual combination -- its net effect on this particular score ran the other way)"
    return phrase


class RiskExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain_row(self, X_row: pd.DataFrame, top_k: int = 3) -> dict:
        """
        X_row: a single-row DataFrame of engineered features (same shape the
        model was trained on).

        Returns a dict with the raw SHAP contributions and a plain-language
        explanation string built from the top_k contributing factors.
        """
        if len(X_row) != 1:
            raise ValueError("explain_row expects exactly one row")

        shap_values = self.explainer.shap_values(X_row)
        row_shap = np.asarray(shap_values)[0]
        feature_names = X_row.columns.tolist()

        contributions = sorted(
            zip(feature_names, row_shap),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        top = [pair for pair in contributions[:top_k] if abs(pair[1]) > 1e-6]

        # Important: a feature's *magnitude* of contribution doesn't tell you
        # its direction. We split top factors into ones that pushed the score
        # up (raised risk) and ones that pushed it down (lowered risk), and
        # say so explicitly -- otherwise a low-risk case whose top factors
        # were all risk-*reducing* would get worded as if it had been
        # flagged, which is backwards and misleading in a demo or an audit.
        raising = [_factor_phrase(name, val, X_row[name].iloc[0]) for name, val in top if val > 0]
        lowering = [_factor_phrase(name, val, X_row[name].iloc[0]) for name, val in top if val < 0]

        if not raising and not lowering:
            explanation = "No single factor stood out; the score reflects a combination of small effects."
        else:
            parts = []
            if raising:
                parts.append("factors raising the risk score: " + "; ".join(raising))
            if lowering:
                parts.append("factors lowering the risk score: " + "; ".join(lowering))
            explanation = "Primary drivers of this score -- " + "; ".join(parts) + "."
            explanation = explanation[0].upper() + explanation[1:]

        return {
            "top_factors": [{"feature": name, "shap_value": float(val)} for name, val in top],
            "explanation": explanation,
        }
