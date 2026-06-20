import numpy as np

from credit_default.modeling import (
    build_threshold_table,
    calibration_bin_table,
    choose_threshold_by_f1,
    expected_calibration_error,
)


def test_threshold_selection_uses_best_f1():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    table = build_threshold_table(y_true, y_score, thresholds=np.array([0.3, 0.5, 0.7]))

    assert choose_threshold_by_f1(table) == 0.5
    assert set(["precision", "recall", "specificity", "balanced_accuracy", "f1"]).issubset(
        table.columns
    )


def test_calibration_bin_table_and_ece():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    table = calibration_bin_table(y_true, y_score, n_bins=2)

    assert len(table) == 2
    assert expected_calibration_error(y_true, y_score, n_bins=2) >= 0
