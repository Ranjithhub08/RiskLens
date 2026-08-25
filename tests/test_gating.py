from gating.decision_engine import (
    DECISION_CLEAR,
    DECISION_ESCALATE,
    DECISION_FLAG,
    DECISION_MANUAL_REVIEW,
    ESCALATE_THRESHOLD,
    FLAG_THRESHOLD,
    LOW_CONFIDENCE_BAND,
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


def test_nan_score_fails_safe_to_manual_review_with_an_honest_reason():
    # Every numeric comparison in decide_from_score is False for NaN, so
    # without an explicit isfinite() guard a NaN score used to fall through
    # to the final `else` branch and get labeled flag_for_compliance_review
    # with a reason claiming it "exceeds" a numeric threshold -- a
    # fabricated comparison in what's supposed to be an honest audit trail.
    result = decide_from_score(float("nan"))
    assert result.decision == DECISION_MANUAL_REVIEW
    assert "exceeds" not in result.reason.lower()


def test_infinite_score_fails_safe_to_manual_review():
    result = decide_from_score(float("inf"))
    assert result.decision == DECISION_MANUAL_REVIEW


def test_low_confidence_band_lower_edge_is_inclusive_despite_float_precision():
    # ESCALATE_THRESHOLD - LOW_CONFIDENCE_BAND (0.50 - 0.02) isn't exactly
    # representable in binary floating point -- it evaluates to
    # 0.020000000000000018, not 0.02 -- so a raw, unrounded
    # abs(risk_score - ESCALATE_THRESHOLD) <= LOW_CONFIDENCE_BAND missed a
    # score of exactly 0.48 by that sliver and let it fall through to
    # "clear" instead of the intended "too close to call, needs manual
    # review" -- the wrong direction for a fail-safe boundary.
    lower_edge = ESCALATE_THRESHOLD - LOW_CONFIDENCE_BAND
    result = decide_from_score(lower_edge)
    assert result.decision == DECISION_MANUAL_REVIEW


def test_low_confidence_band_upper_edge_is_inclusive():
    upper_edge = ESCALATE_THRESHOLD + LOW_CONFIDENCE_BAND
    result = decide_from_score(upper_edge)
    assert result.decision == DECISION_MANUAL_REVIEW


def test_just_outside_low_confidence_band_is_not_manual_review():
    # Sanity check that the rounded comparison didn't widen the band --
    # comfortably outside it should still auto-decide.
    result = decide_from_score(ESCALATE_THRESHOLD - LOW_CONFIDENCE_BAND - 0.05)
    assert result.decision == DECISION_CLEAR


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
