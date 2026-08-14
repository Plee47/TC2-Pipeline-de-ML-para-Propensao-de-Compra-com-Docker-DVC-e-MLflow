
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series | None = None
) -> dict[str, float]:
    """Evaluate a binary classifier.

    The target is heavily imbalanced (~15% positives), so accuracy alone is
    misleading: ``average_precision`` (PR-AUC) and ``recall`` are the metrics
    that actually tell whether the model finds buyers.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Probability of the positive class (optional).

    Returns:
        Dictionary with metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_pred_proba)
        )

    return metrics


def log_metrics_to_mlflow(metrics: dict[str, float]) -> None:
    """Log metrics to the active MLflow run."""
    mlflow.log_metrics(metrics)
