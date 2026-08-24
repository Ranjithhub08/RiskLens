from gating.decision_engine import (
    DECISION_CLEAR,
    DECISION_ESCALATE,
    DECISION_FLAG,
    DECISION_MANUAL_REVIEW,
    ESCALATE_THRESHOLD,
    FLAG_THRESHOLD,
    decide_for_record,
    decide_from_score,
)


def test_low_score_clears():
    result = decide_from_score(0.10)
    assert result.decision == DECISION_CLEAR


def test_mid_score_escalates():
    result = decide_from_score((ESCALATE_THRESHOLD + FLAG_THRESHOLD) / 2)
    assert result.decision == DECISION_ESCALATE


def test_high_score_flags_for_compliance():
    result = decide_from_score(0.95)
    assert result.decision == DECISION_FLAG


def test_score_exactly_at_boundary_fails_safe_to_manual_review():
    # Right at the escalate threshold -- inside the low-confidence band --
    # should not be auto-decided.
    result = decide_from_score(ESCALATE_THRESHOLD)
    assert result.decision == DECISION_MANUAL_REVIEW


def test_none_score_fails_safe_to_manual_review():
    result = decide_from_score(None)
    assert result.decision == DECISION_MANUAL_REVIEW


def test_missing_fields_short_circuits_to_manual_review_even_with_a_score():
    # Even if a score somehow existed, missing required fields must win --
    # the system should never trust a score computed from incomplete data.
    result = decide_for_record(missing_fields=["kyc_status"], risk_score=0.9)
    assert result.decision == DECISION_MANUAL_REVIEW


def test_no_missing_fields_uses_the_score():
    result = decide_for_record(missing_fields=[], risk_score=0.1)
    assert result.decision == DECISION_CLEAR


def test_decision_reason_is_never_empty():
    for score in [0.0, 0.2, ESCALATE_THRESHOLD, 0.6, FLAG_THRESHOLD, 1.0]:
        result = decide_from_score(score)
        assert result.reason and len(result.reason) > 0
