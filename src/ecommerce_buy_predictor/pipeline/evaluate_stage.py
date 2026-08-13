import json
from pathlib import Path

import mlflow

from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.models.registry import promote_best_model_to_registry
from ecommerce_buy_predictor.pipeline.params import load_params
from ecommerce_buy_predictor.pipeline.train_stage import TRAIN_REPORT_FILE

METRICS_FILE = Path("metrics.json")


def evaluate_stage() -> None:
    """DVC stage: promote the best run to the Model Registry and write metrics.

    ``metrics.json`` is the file DVC reads, so it carries the actual model
    metrics (that is what ``dvc metrics show/diff`` is for), plus the
    promotion outcome.
    """
    params = load_params()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    metrics_data = json.loads(TRAIN_REPORT_FILE.read_text(encoding="utf-8"))

    promotion = promote_best_model_to_registry(
        model_name=settings.model_name,
        experiment_name=settings.experiment_name,
        metric_name=params["selection_metric"],
        alias=settings.model_alias,
    )

    metrics_data.update(
        {
            "promoted": True,
            "model_uri": promotion["model_uri"],
            "registered_version": promotion["version"],
            "run_id": promotion["run_id"],
        }
    )

    METRICS_FILE.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"Promoted {promotion['model_uri']} (version {promotion['version']})")


if __name__ == "__main__":
    evaluate_stage()
