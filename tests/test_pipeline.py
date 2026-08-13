"""End-to-end tests for the DVC stages, against a temporary MLflow store."""
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from ecommerce_buy_predictor.models.registry import promote_best_model_to_registry
from ecommerce_buy_predictor.pipeline.params import load_params

PARAMS = """
random_seed: 42
split:
  test_size: 0.2
selection_metric: average_precision
models:
  LogisticRegression:
    max_iter: 200
  RandomForest:
    n_estimators: 15
    max_depth: 4
"""


@pytest.fixture
def project(tmp_path, raw_dataset, monkeypatch):
    """Isolated working directory with raw data, params and MLflow store."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    raw_dataset.to_csv(tmp_path / "data" / "raw" / "online_shoppers_intention.csv", index=False)
    (tmp_path / "params.yaml").write_text(PARAMS, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr("ecommerce_buy_predictor.config.settings.mlflow_tracking_uri", tracking_uri)
    monkeypatch.setattr(
        "ecommerce_buy_predictor.config.settings.experiment_name", "test_experiment"
    )
    mlflow.set_tracking_uri(tracking_uri)
    return tmp_path


def test_load_params_reads_the_file(project):
    params = load_params()

    assert params["models"]["RandomForest"]["n_estimators"] == 15


def test_load_params_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "nope.yaml")


def test_full_pipeline(project):
    from ecommerce_buy_predictor.pipeline.evaluate_stage import evaluate_stage
    from ecommerce_buy_predictor.pipeline.preprocess_stage import preprocess_stage
    from ecommerce_buy_predictor.pipeline.train_stage import train_stage

    preprocess_stage()
    assert (project / "data" / "processed" / "X_train.csv").exists()

    train_stage()
    assert (project / "models" / "model.pkl").exists()

    report = json.loads((project / "reports" / "train_metrics.json").read_text())
    assert report["best_model"] in {"LogisticRegression", "RandomForest"}

    evaluate_stage()
    metrics = json.loads((project / "metrics.json").read_text())

    # The regression this whole exercise is about: the file DVC tracks must
    # carry real metrics, and the model must actually reach the registry.
    assert metrics["promoted"] is True
    assert metrics["model_uri"].endswith("@champion")
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert metrics["recall"] > 0.0


def test_preprocess_requires_raw_data(project):
    from ecommerce_buy_predictor.pipeline.preprocess_stage import preprocess_stage

    os.remove(project / "data" / "raw" / "online_shoppers_intention.csv")

    with pytest.raises(FileNotFoundError, match="Raw data not found"):
        preprocess_stage()


def test_promotion_fails_loudly_without_runs(project):
    with pytest.raises(ValueError, match="No finished run"):
        promote_best_model_to_registry(
            model_name="empty", experiment_name="test_experiment"
        )


def test_promoted_model_is_servable(project):
    """The URI written to metrics.json must be loadable by the API."""
    from ecommerce_buy_predictor.pipeline.preprocess_stage import preprocess_stage
    from ecommerce_buy_predictor.pipeline.train_stage import train_stage

    preprocess_stage()
    train_stage()
    promotion = promote_best_model_to_registry(
        model_name="servable", experiment_name="test_experiment"
    )

    model = mlflow.sklearn.load_model(promotion["model_uri"])
    raw = pd.read_csv(Path("data") / "processed" / "X_test.csv")

    assert len(model.predict_proba(raw)) == len(raw)
