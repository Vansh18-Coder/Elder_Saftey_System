from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_report


def select_threshold_for_precision(y_true, fall_probabilities, target_precision=0.9):
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(fall_probabilities, dtype=np.float64)

    if not 0.0 < target_precision <= 1.0:
        raise ValueError("Target precision must be greater than 0 and at most 1.")
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and fall probabilities must be one-dimensional.")
    if labels.size == 0 or labels.size != probabilities.size:
        raise ValueError("Labels and fall probabilities must be non-empty and have equal lengths.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 (normal) and 1 (fall).")
    if not np.isfinite(probabilities).all() or ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Fall probabilities must be finite values between 0 and 1.")
    if not np.isin(labels, [0]).any() or not np.isin(labels, [1]).any():
        raise ValueError("Validation data must contain both normal and fall examples.")

    candidates = []
    for threshold in np.unique(probabilities):
        report = classification_report(labels, probabilities >= threshold, num_classes=2)
        true_positives = int(report["confusion_matrix"][1, 1])
        if true_positives and report["precision"] >= target_precision:
            candidates.append((report["recall"], report["precision"], -threshold, threshold, report))

    if not candidates:
        raise ValueError(
            f"No validation threshold achieved the target precision of {target_precision:.3f} "
            "while detecting at least one fall."
        )

    _, _, _, threshold, report = max(candidates, key=lambda candidate: candidate[:3])
    return float(threshold), report
