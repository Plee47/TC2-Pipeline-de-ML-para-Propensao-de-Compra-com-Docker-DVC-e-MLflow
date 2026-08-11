import pandas as pd
import pytest
import tempfile
import os
from tech_challenge.models.train import ModelTrainer


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [2.0, 3.0, 4.0, 5.0, 6.0],
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_logistic_regression_training(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(random_seed=42)
    trainer.train_logistic_regression(X, y)

    assert trainer.model is not None
    predictions = trainer.predict(X)
    assert len(predictions) == len(X)


def test_random_forest_training(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(random_seed=42)
    trainer.train_random_forest(X, y, n_estimators=10)

    assert trainer.model is not None
    predictions = trainer.predict(X)
    assert len(predictions) == len(X)


def test_predict_proba(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(random_seed=42)
    trainer.train_logistic_regression(X, y)

    proba = trainer.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert proba.shape[1] == 2


def test_save_and_load_model(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(random_seed=42)
    trainer.train_logistic_regression(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        trainer.save_model(model_path)

        trainer2 = ModelTrainer()
        trainer2.load_model(model_path)

        pred1 = trainer.predict(X)
        pred2 = trainer2.predict(X)

        assert (pred1 == pred2).all()
