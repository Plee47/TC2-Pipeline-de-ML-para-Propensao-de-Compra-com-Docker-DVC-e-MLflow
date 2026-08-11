import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, Any


def evaluate_model(
    y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame | None = None
) -> Dict[str, float]:
    """Evaluate model and return metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC).

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
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_pred_proba.iloc[:, 1])
            )
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def log_metrics_to_mlflow(metrics: Dict[str, float]) -> None:
    """Log metrics to MLflow run."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
