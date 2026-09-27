from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()), "precision": precision.tolist(),
        "recall": recall.tolist(), "f1": f1.tolist(), "support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
