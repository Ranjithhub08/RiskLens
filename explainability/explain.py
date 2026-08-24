"""
Per-prediction explainability.

Wraps SHAP's TreeExplainer so any single case can be explained, and
translates the top contributing features into a plain-language sentence --
this is what turns "risk score: 0.81" into something a merchant or a
reviewer can actually act on.
"""

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


def _phrase_for(feature_name: str, shap_value: float) -> str:
    direction = "up" if shap_value > 0 else "down"
    if feature_name in FEATURE_PHRASES:
        return FEATURE_PHRASES[feature_name][direction]
    if feature_name.startswith("category_"):
        category = feature_name.replace("category_", "")
        if direction == "up":
            return f"the business category ('{category}') is statistically associated with higher risk"
        return f"the business category ('{category}') is not a risk driver here"
    return f"'{feature_name}' contributed to the {'higher' if direction == 'up' else 'lower'} score"


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
        raising = [_phrase_for(name, val) for name, val in top if val > 0]
        lowering = [_phrase_for(name, val) for name, val in top if val < 0]

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
