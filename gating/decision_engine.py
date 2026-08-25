"""
Gating and decision engine.

Deliberately plain Python rules, not another model -- this is the layer
that bounds the system's authority. It never freezes an account itself; it
only ever produces one of a fixed set of recommendations, each carrying a
human-readable reason. Thresholds are configuration (not hard-coded deep in
logic) so they can be tuned and audited independently of the model.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# Configurable thresholds -- see docs/ARCHITECTURE.md section 4.5 for the
# reasoning behind these values.
#
# These are NOT arbitrary round numbers -- the trained model's predicted
# probabilities are naturally compressed into a fairly narrow band (roughly
# 0.33-0.69 on held-out data, not the full 0-1 range a probability could in
# principle span), because the synthetic training data only weakly separates
# risky from clean merchants. Thresholds of 0.40 / 0.75 were an earlier,
# untested guess that sat almost entirely *inside* that compressed band --
# in practice that meant nearly every record landed in "escalate" or
# "needs manual review", and "clear" and "flag" almost never fired at all
# (see model/artifacts/metrics.json and the note in decide_from_score below).
#
# The values below were instead picked from where the model's own held-out
# validation scores actually fall (see model/train.py's load_and_split /
# the Threshold Explorer on the Models page): ESCALATE_THRESHOLD sits just
# above the dense cluster of "ordinary" scores (~80th percentile), and
# FLAG_THRESHOLD sits at the start of the long high-risk tail (~90th
# percentile). Re-run model/train.py and re-check this distribution any
# time the training data changes -- these are tuned to the current model,
# not universal constants.
ESCALATE_THRESHOLD = 0.50
FLAG_THRESHOLD = 0.62

# If the model's score sits within this band around the escalate threshold,
# treat it as "too close to call confidently" and fail safe to manual
# review rather than trusting the raw number. Kept small relative to the
# thresholds above precisely because the model's score range is itself
# narrow -- a wide band here would swallow most of the "clear" bucket.
LOW_CONFIDENCE_BAND = 0.02

DECISION_CLEAR = "clear"
DECISION_ESCALATE = "escalate"
DECISION_FLAG = "flag_for_compliance_review"
DECISION_MANUAL_REVIEW = "needs_manual_review"

# A stable identifier for this gate implementation, surfaced in the UI and
# audit trail so every decision can be traced to the exact rule set that
# produced it. Bump this if the thresholds or logic below ever change.
GATE_VERSION = "DETERMINISTIC-GATE-01"


@dataclass
class GatingResult:
    decision: str
    reason: str
    risk_score: Optional[float] = None
    thresholds_used: dict = field(default_factory=lambda: {
        "escalate_threshold": ESCALATE_THRESHOLD,
        "flag_threshold": FLAG_THRESHOLD,
    })


def decide_from_score(risk_score: float) -> GatingResult:
    """
    Map a risk probability to a bounded decision. This function assumes the
    score itself is trustworthy (i.e. the record was already validated and
    successfully scored) -- missing/invalid input is handled separately by
    decide_for_record below, before a score is even produced.
    """
    if risk_score is None:
        return GatingResult(
            decision=DECISION_MANUAL_REVIEW,
            reason="No risk score available.",
            risk_score=None,
        )

    # NaN/inf can't come from a properly-validated, properly-scored record
    # (pipeline.py checks for NaN features before scoring), but this
    # function has no way to enforce that from here -- every comparison
    # below is False for NaN, so without this guard a NaN score used to
    # fall through to the final `else` and get labeled
    # flag_for_compliance_review with a reason claiming it "exceeds" a
    # numeric threshold, which is simply untrue for NaN. Fail safe instead,
    # with a reason that doesn't assert a comparison that never happened.
    if not math.isfinite(risk_score):
        return GatingResult(
            decision=DECISION_MANUAL_REVIEW,
            reason=f"Risk score ({risk_score!r}) is not a valid number; routed for manual review.",
            risk_score=risk_score,
        )

    # Rounded before comparing: ESCALATE_THRESHOLD - LOW_CONFIDENCE_BAND
    # (0.50 - 0.02) isn't exactly representable in binary floating point --
    # it evaluates to 0.020000000000000018, not 0.02 -- so a raw
    # `abs(risk_score - ESCALATE_THRESHOLD) <= LOW_CONFIDENCE_BAND` missed
    # a score of exactly 0.48 by that sliver and let it fall through to
    # "clear" instead of the intended "too close to call, needs manual
    # review". That's the wrong direction for a fail-safe boundary, so the
    # comparison is rounded to kill the float noise instead of comparing
    # raw floats at the exact edge.
    if round(abs(risk_score - ESCALATE_THRESHOLD), 6) <= LOW_CONFIDENCE_BAND:
        return GatingResult(
            decision=DECISION_MANUAL_REVIEW,
            reason=(
                f"Risk score ({risk_score:.2f}) is too close to the escalation threshold "
                f"({ESCALATE_THRESHOLD}) to act on automatically."
            ),
            risk_score=risk_score,
        )

    if risk_score < ESCALATE_THRESHOLD:
        return GatingResult(
            decision=DECISION_CLEAR,
            reason=f"Risk score ({risk_score:.2f}) is below the escalation threshold.",
            risk_score=risk_score,
        )
    if risk_score <= FLAG_THRESHOLD:
        return GatingResult(
            decision=DECISION_ESCALATE,
            reason=(
                f"Risk score ({risk_score:.2f}) is elevated; routed to a human reviewer "
                "with the full explanation attached."
            ),
            risk_score=risk_score,
        )
    return GatingResult(
        decision=DECISION_FLAG,
        reason=(
            f"Risk score ({risk_score:.2f}) exceeds the compliance-review threshold "
            f"({FLAG_THRESHOLD}); routed as high priority."
        ),
        risk_score=risk_score,
    )


def decide_for_record(missing_fields: list, risk_score: Optional[float] = None) -> GatingResult:
    """
    The entry point used by the pipeline. If required raw fields were
    missing (caught upstream by features.validate_raw), this fails safe to
    manual review before ever trusting a model score.
    """
    if missing_fields:
        return GatingResult(
            decision=DECISION_MANUAL_REVIEW,
            reason=f"Cannot score automatically -- missing or invalid required fields: {missing_fields}.",
            risk_score=None,
        )
    return decide_from_score(risk_score)
