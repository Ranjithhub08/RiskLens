"""
Tests for model/train.py's threshold-selection helper.

best_threshold_for_f1 is used both by a plain `python3 model/train.py` run
and by model/feedback.py's retrain-with-feedback flow (which imports it
directly) -- a bug here silently affects the decision threshold shipped to
production in either path.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

from model import train as train_module
from model.train import DEFAULT_THRESHOLD, best_threshold_for_f1, clear_stale_test_snapshot


def test_best_threshold_for_f1_picks_a_sensible_threshold_on_separable_data():
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50)
    # Positives get systematically higher scores than negatives.
    y_prob = np.concatenate([rng.uniform(0.0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
    threshold = best_threshold_for_f1(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    # A threshold that actually separates the two classes should get
    # (near-)perfect F1 on data this clean.
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    assert f1 > 0.9


def test_best_threshold_for_f1_falls_back_to_default_when_validation_set_has_no_positives():
    # Regression test: with y_true all zeros, precision is 0 and recall is
    # (degenerately) defined as 1 at every threshold sklearn considers, so
    # F1 ties at 0 everywhere. np.nanargmax on an all-zero array used to
    # silently return index 0 -- the LOWEST probability the model happened
    # to output on this split -- which is not a real threshold choice, just
    # an artifact of a degenerate split. It must fall back to a safe
    # default instead of shipping that as the production decision
    # threshold.
    rng = np.random.RandomState(0)
    y_true = np.zeros(30, dtype=int)
    y_prob = rng.uniform(0.0, 1.0, 30)
    assert best_threshold_for_f1(y_true, y_prob) == DEFAULT_THRESHOLD


def test_best_threshold_for_f1_maximizes_recall_when_validation_set_has_no_negatives():
    # Not a degenerate case like "no positives": with every validation
    # example positive, precision is trivially 1.0 at every threshold, so
    # F1 varies purely with recall and is genuinely maximized by predicting
    # everything positive (the lowest observed probability becomes the
    # threshold). That's a real, meaningful answer here, not an artifact --
    # only the "no positives at all" case (tested above) is degenerate.
    rng = np.random.RandomState(0)
    y_true = np.ones(30, dtype=int)
    y_prob = rng.uniform(0.0, 1.0, 30)
    threshold = best_threshold_for_f1(y_true, y_prob)
    assert threshold <= y_prob.min()


def test_best_threshold_for_f1_handles_empty_input():
    assert best_threshold_for_f1(np.array([]), np.array([])) == DEFAULT_THRESHOLD


def test_confusion_matrix_call_with_explicit_labels_stays_2x2_on_a_single_class_split():
    # Regression test for the exact `confusion_matrix(y_test, y_pred,
    # labels=[0, 1])` call used in both model/train.py's main() and
    # model/feedback.py's train_candidate_with_feedback(): a time-based
    # test split can land entirely on one class (e.g. a short, quiet
    # period with zero risky merchants). Without labels=[0, 1],
    # confusion_matrix collapses to a 1x1 matrix here, which crashes
    # ConfusionMatrixDisplay(display_labels=["Not risky", "Risky"]) in
    # train.py and corrupts the "confusion_matrix" entry written to
    # chart_data.json in feedback.py.
    y_test = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    assert cm.shape == (2, 2)


def test_clear_stale_test_snapshot_removes_an_existing_snapshot(tmp_path, monkeypatch):
    # Regression test: a plain `python3 model/train.py` run regenerates
    # xgb_model.joblib directly but, before this fix, left any PREVIOUS
    # feedback-augmented promotion's deployed_test_snapshot.csv untouched
    # on disk -- so the dashboard's Threshold Explorer (which prefers this
    # snapshot whenever it exists) kept showing predictions from the old
    # model/test-split pairing instead of the model that was just trained.
    fake_snapshot = tmp_path / "deployed_test_snapshot.csv"
    fake_snapshot.write_text("merchant_id,is_risky\nm1,0\n")
    monkeypatch.setattr(train_module, "TEST_SNAPSHOT_PATH", str(fake_snapshot))

    clear_stale_test_snapshot()

    assert not fake_snapshot.exists()


def test_clear_stale_test_snapshot_is_a_no_op_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "TEST_SNAPSHOT_PATH", str(tmp_path / "does_not_exist.csv"))
    clear_stale_test_snapshot()  # must not raise


def test_train_and_feedback_agree_on_the_test_snapshot_path():
    # model/train.py's TEST_SNAPSHOT_PATH is a deliberately duplicated
    # literal (see the comment above its definition) rather than an import
    # from model/feedback.py, specifically to avoid a circular import.
    # Duplication like that only stays safe as long as the two values
    # actually match -- if they ever drift apart, clear_stale_test_snapshot
    # would silently delete (or fail to delete) the wrong file.
    from model.feedback import TEST_SNAPSHOT_PATH as feedback_path

    assert train_module.TEST_SNAPSHOT_PATH == feedback_path
