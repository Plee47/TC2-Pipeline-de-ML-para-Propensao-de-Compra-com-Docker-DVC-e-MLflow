from pathlib import Path
import json
import mlflow
from tech_challenge.config import settings
from tech_challenge.models.registry import promote_best_model_to_registry


def evaluate_stage() -> None:
    """DVC pipeline stage: promote best model to registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    metrics_data = {"promoted": False, "model_uri": None}

    try:
        model_uri = promote_best_model_to_registry(
            "online_shoppers_intention", metric_name="f1"
        )
        print(f"Best model promoted to registry: {model_uri}")
        metrics_data["promoted"] = True
        metrics_data["model_uri"] = model_uri

    except ValueError as e:
        print(f"Warning: {e}")

    metrics_file = Path("metrics.json")
    metrics_file.write_text(json.dumps(metrics_data))


if __name__ == "__main__":
    evaluate_stage()
