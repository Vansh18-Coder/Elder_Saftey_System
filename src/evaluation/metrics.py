from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=2):
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for actual, predicted in zip(true, pred):
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            matrix[actual, predicted] += 1

    return matrix


def classification_report(y_true, y_pred, num_classes=2):
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    results = {}

    total = matrix.sum()
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "specificity": 0.0}

    tp = matrix[1, 1]
    fp = matrix[0, 1]
    fn = matrix[1, 0]
    tn = matrix[0, 0]

    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)

    results["accuracy"] = float(accuracy)
    results["precision"] = float(precision)
    results["recall"] = float(recall)
    results["f1"] = float(f1)
    results["specificity"] = float(specificity)
    results["confusion_matrix"] = matrix
    return results
