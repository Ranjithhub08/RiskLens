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
    assert theme.risk_label_for_score(0.52)[0] == "Elevated risk"
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


# model_comparison_table_html's "winner" dot previously used a bare `xg >=
# bl` comparison, which meant an exact tie was always credited to the LEFT
# column as though it had won. That's most visible (and most misleading) on
# the Models page's "Retrain from feedback" section: a single extra
# feedback row folded into 6000+ original rows can easily leave a candidate
# model performing IDENTICALLY to what's already deployed, and the table
# was claiming the candidate beat the deployed model on every metric when
# it hadn't beaten it on any of them.
def _metrics(precision, recall, f1, roc_auc, threshold=0.5):
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "threshold": threshold}


def _body_rows_html(html_out: str) -> str:
    """Strips the trailing "&#9679; marks the stronger model" caption (which
    always contains one literal &#9679; of its own) so tests can count only
    the winner dots actually placed in the table body."""
    return html_out.split("<tbody>")[1].split("</tbody>")[0]


def test_model_comparison_table_marks_no_winner_on_an_exact_tie():
    tied = _metrics(0.398, 0.519, 0.451, 0.756)
    html_out = theme.model_comparison_table_html(tied, tied)
    assert _body_rows_html(html_out).count("&#9679;") == 0


def test_model_comparison_table_marks_no_winner_on_a_tie_at_display_precision():
    # 0.3981 and 0.3979 both print as "0.398" -- a win dot next to two
    # numbers that read identically would look like the table is
    # contradicting itself, so ties are judged at the same 3-decimal
    # precision the numbers are actually displayed at.
    left = _metrics(0.3981, 0.519, 0.451, 0.756)
    right = _metrics(0.3979, 0.519, 0.451, 0.756)
    html_out = theme.model_comparison_table_html(left, right)
    assert _body_rows_html(html_out).count("&#9679;") == 0


def test_model_comparison_table_still_marks_a_genuine_winner():
    stronger = _metrics(0.50, 0.519, 0.451, 0.756)
    weaker = _metrics(0.30, 0.519, 0.451, 0.756)
    html_out = theme.model_comparison_table_html(stronger, weaker)
    body = _body_rows_html(html_out)
    # Exactly one dot: only Precision differs, and it favors `stronger`.
    assert body.count("&#9679;") == 1
    precision_row = body.split("Precision</td>")[1].split("</tr>")[0]
    assert "0.500 &#9679;" in precision_row
    assert "0.300 &#9679;" not in precision_row
