"""
Gating and decision engine.

Deliberately plain Python rules, not another model -- this is the layer
that bounds the system's authority. It never freezes an account itself; it
only ever produces one of a fixed set of recommendations, each carrying a
human-readable reason. Thresholds are configuration (not hard-coded deep in
logic) so they can be tuned and audited independently of the model.
"""

from dataclasses import dataclass, field
from typing import Optional

# Configurable thresholds -- see docs/ARCHITECTURE.md section 4.5 for the
# reasoning behind these starting values.
ESCALATE_THRESHOLD = 0.40
FLAG_THRESHOLD = 0.75

# If the model's score sits within this band around the escalate threshold,
# treat it as "too close to call confidently" and fail safe to manual
# review rather than trusting the raw number.
LOW_CONFIDENCE_BAND = 0.05

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

    if abs(risk_score - ESCALATE_THRESHOLD) <= LOW_CONFIDENCE_BAND:
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
