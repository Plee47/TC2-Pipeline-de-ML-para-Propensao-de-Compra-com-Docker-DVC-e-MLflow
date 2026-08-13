import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.models.evaluate import evaluate_model, log_metrics_to_mlflow
from ecommerce_buy_predictor.models.train import ModelTrainer
from ecommerce_buy_predictor.pipeline.params import load_params

TRAIN_REPORT_FILE = Path("reports/train_metrics.json")


def _load_split(processed_dir: Path):
    """Read the train/test split produced by the preprocess stage."""
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found: {processed_dir}. Run the preprocess stage first."
        )

    return (
        pd.read_csv(processed_dir / "X_train.csv"),
        pd.read_csv(processed_dir / "X_test.csv"),
        pd.read_csv(processed_dir / "y_train.csv").iloc[:, 0],
        pd.read_csv(processed_dir / "y_test.csv").iloc[:, 0],
    )


def train_stage() -> None:
    """DVC stage: train every model in ``params.yaml`` and log them to MLflow.

    The best model by the configured selection metric is persisted to
    ``models/model.pkl`` and its metrics to ``reports/train_metrics.json``.
    """
    params = load_params()
    random_seed = params["random_seed"]
    selection_metric = params["selection_metric"]

    X_train, X_test, y_train, y_test = _load_split(Path(settings.data_processed_path))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    best = {"score": float("-inf"), "name": None, "trainer": None, "metrics": {}}

    for estimator_name, estimator_params in params["models"].items():
        with mlflow.start_run(run_name=estimator_name):
            mlflow.log_params({"model": estimator_name, **estimator_params})
            mlflow.log_param("random_seed", random_seed)

            trainer = ModelTrainer(estimator_name, estimator_params, random_seed).fit(
                X_train, y_train
            )

            y_pred = trainer.predict(X_test)
            y_pred_proba = trainer.predict_proba(X_test)
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            log_metrics_to_mlflow(metrics)

            mlflow.sklearn.log_model(
                trainer.model,
                artifact_path="model",
                signature=infer_signature(X_test, y_pred.to_numpy()),
                input_example=X_test.head(),
            )

            print(f"{estimator_name}: {metrics}")

            if metrics[selection_metric] > best["score"]:
                best = {
                    "score": metrics[selection_metric],
                    "name": estimator_name,
                    "trainer": trainer,
                    "metrics": metrics,
                }

    if best["trainer"] is None:
        raise ValueError("No model was trained. Check the 'models' key in params.yaml.")

    models_dir = Path(settings.model_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    best["trainer"].save_model(models_dir / "model.pkl")

    TRAIN_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORT_FILE.write_text(
        json.dumps(
            {"best_model": best["name"], "selection_metric": selection_metric,
             **best["metrics"]},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Best model: {best['name']} "
        f"({selection_metric}={best['score']:.4f}) saved to {models_dir / 'model.pkl'}"
    )


if __name__ == "__main__":
    train_stage()
