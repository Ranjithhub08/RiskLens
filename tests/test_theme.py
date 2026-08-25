"""
Tests for the risk-score display helpers in app/theme.py.

gating/decision_engine.decide_from_score treats a NaN/inf risk score the
same as a missing one -- routed to needs_manual_review with an honest
reason, never silently compared against a numeric threshold. These display
functions previously only guarded against `score is None`, so a NaN score
(every comparison against NaN is False) fell through to the "High risk"
branch instead -- the visual opposite of what the gate itself decided for
that same score.
"""

import app.theme as theme


def test_risk_label_for_none_is_unscored():
    label, _ = theme.risk_label_for_score(None)
    assert label == "Unscored"


def test_risk_label_for_nan_is_unscored_not_high_risk():
    label, _ = theme.risk_label_for_score(float("nan"))
    assert label == "Unscored"


def test_risk_label_for_infinite_is_unscored():
    label, _ = theme.risk_label_for_score(float("inf"))
    assert label == "Unscored"


def test_risk_label_for_ordinary_scores_is_unaffected():
    assert theme.risk_label_for_score(0.10)[0] == "Low risk"
    assert theme.risk_label_for_score(0.55)[0] == "Elevated risk"
    assert theme.risk_label_for_score(0.90)[0] == "High risk"


def test_risk_scale_html_draws_no_marker_for_nan():
    # min(1.0, float('nan')) is 1.0 in Python -- without the isfinite guard
    # this used to draw the marker at the far right of the scale (100%,
    # reading as "confirmed maximum risk") for a score the gate itself
    # couldn't act on at all.
    html_out = theme.risk_scale_html(float("nan"))
    assert "rl-scale-marker" not in html_out


def test_risk_scale_html_draws_no_marker_for_none():
    html_out = theme.risk_scale_html(None)
    assert "rl-scale-marker" not in html_out


def test_risk_scale_html_draws_a_marker_for_an_ordinary_score():
    html_out = theme.risk_scale_html(0.5)
    assert "rl-scale-marker" in html_out
