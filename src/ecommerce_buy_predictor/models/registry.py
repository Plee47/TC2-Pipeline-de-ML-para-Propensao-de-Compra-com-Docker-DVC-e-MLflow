from typing import Any

import mlflow

from ecommerce_buy_predictor.config import settings


def promote_best_model_to_registry(
    model_name: str | None = None,
    experiment_name: str | None = None,
    metric_name: str = "average_precision",
    alias: str | None = None,
) -> dict[str, Any]:
    """Register the best run of an experiment and point an alias at it.

    Args:
        model_name: Name to register the model under.
        experiment_name: Experiment to search. Passing it explicitly matters:
            ``mlflow.search_runs()`` without it looks at the *active*
            experiment (``Default``), which never holds the training runs.
        metric_name: Metric used to rank runs (higher is better).
        alias: Registry alias moved to the winning version. Aliases replace
            the stages API, removed in MLflow 3.

    Returns:
        Dict with ``run_id``, ``version``, ``model_uri`` and ``metric``.

    Raises:
        ValueError: If the experiment has no finished run with that metric.
    """
    model_name = model_name or settings.model_name
    experiment_name = experiment_name or settings.experiment_name
    alias = alias or settings.model_alias
    metric_column = f"metrics.{metric_name}"

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"{metric_column} DESC"],
    )

    if runs.empty or metric_column not in runs.columns:
        raise ValueError(
            f"No finished run with metric '{metric_name}' in experiment "
            f"'{experiment_name}'. Run the train stage first."
        )

    runs = runs.dropna(subset=[metric_column])
    if runs.empty:
        raise ValueError(
            f"No run in experiment '{experiment_name}' logged '{metric_name}'."
        )

    best_run = runs.iloc[0]
    model_version = mlflow.register_model(
        f"runs:/{best_run.run_id}/model", model_name
    )

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=model_name, alias=alias, version=model_version.version
    )

    return {
        "run_id": best_run.run_id,
        "version": int(model_version.version),
        "model_uri": f"models:/{model_name}@{alias}",
        "metric": {metric_name: float(best_run[metric_column])},
    }
