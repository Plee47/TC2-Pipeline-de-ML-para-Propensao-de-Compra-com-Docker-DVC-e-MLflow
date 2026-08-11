import sys
from pathlib import Path
import pickle
import mlflow
import pandas as pd
from tech_challenge.config import settings
from tech_challenge.models.train import ModelTrainer
from tech_challenge.models.evaluate import evaluate_model, log_metrics_to_mlflow


def train_stage() -> None:
    """DVC pipeline stage: train model and log to MLflow."""
    processed_dir = Path(settings.data_processed_path)
    models_dir = Path(settings.model_registry_uri)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_dir}")

    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_test = pd.read_parquet(processed_dir / "X_test.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("online_shoppers_intention")

    models_to_train = [
        ("LogisticRegression", {"max_iter": 1000}),
        ("RandomForest", {"n_estimators": 100, "max_depth": 10}),
    ]

    for model_name, params in models_to_train:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({"model": model_name, **params})

            trainer = ModelTrainer(random_seed=settings.random_seed)

            if model_name == "LogisticRegression":
                trainer.train_logistic_regression(X_train, y_train, **params)
            else:
                trainer.train_random_forest(X_train, y_train, **params)

            y_pred = trainer.predict(X_test)
            y_pred_proba = trainer.predict_proba(X_test)

            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            log_metrics_to_mlflow(metrics)

            mlflow.sklearn.log_model(
                trainer.model, "model", input_example=X_test.iloc[:1]
            )

            print(f"{model_name} trained. Metrics: {metrics}")

    print("Training complete.")


if __name__ == "__main__":
    train_stage()
