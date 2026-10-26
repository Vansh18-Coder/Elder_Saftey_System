import numpy as np
import pytest

from src.evaluation.thresholds import select_threshold_for_precision


def test_threshold_selection_meets_target_and_maximizes_recall():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.8, 0.6, 0.7, 0.9])

    threshold, report = select_threshold_for_precision(
        labels,
        probabilities,
        target_precision=1.0,
    )

    assert threshold == pytest.approx(0.9)
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(1 / 3)


def test_threshold_selection_rejects_validation_without_both_classes():
    with pytest.raises(ValueError, match="both normal and fall"):
        select_threshold_for_precision([0, 0], [0.2, 0.3])


def test_threshold_selection_rejects_unreachable_precision_target():
    with pytest.raises(ValueError, match="No validation threshold"):
        select_threshold_for_precision([0, 1], [0.9, 0.8], target_precision=1.0)
