import pandas as pd
import pytest

from ecommerce_buy_predictor.models.evaluate import evaluate_model
from ecommerce_buy_predictor.models.train import ESTIMATORS, ModelTrainer, build_model


@pytest.mark.parametrize("estimator_name", sorted(ESTIMATORS))
def test_pipeline_trains_on_raw_features(estimator_name, raw_features, raw_target):
    trainer = ModelTrainer(estimator_name, {}, random_seed=42).fit(raw_features, raw_target)

    predictions = trainer.predict(raw_features)
    probabilities = trainer.predict_proba(raw_features)

    assert len(predictions) == len(raw_features)
    assert probabilities.between(0, 1).all()


def test_build_model_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="Unknown estimator"):
        build_model("XGBoost", {}, random_seed=42)


def test_hyperparameters_reach_the_estimator():
    model = build_model("RandomForest", {"n_estimators": 7, "max_depth": 3}, 42)
    estimator = model.named_steps["estimator"]

    assert estimator.n_estimators == 7
    assert estimator.max_depth == 3
    assert estimator.class_weight == "balanced"


def test_model_accepts_raw_categorical_values(raw_features, raw_target):
    """The pipeline must swallow strings and booleans, not pre-encoded numbers."""
    trainer = ModelTrainer("LogisticRegression", {"max_iter": 200}, 42).fit(
        raw_features, raw_target
    )

    single_row = raw_features.head(1)

    assert single_row["Month"].dtype == object
    assert trainer.predict(single_row).iloc[0] in (0, 1)


def test_training_is_deterministic(raw_features, raw_target):
    first = ModelTrainer("RandomForest", {"n_estimators": 10}, 42).fit(raw_features, raw_target)
    second = ModelTrainer("RandomForest", {"n_estimators": 10}, 42).fit(raw_features, raw_target)

    pd.testing.assert_series_equal(
        first.predict_proba(raw_features), second.predict_proba(raw_features)
    )


def test_save_and_load_roundtrip(tmp_path, raw_features, raw_target):
    trainer = ModelTrainer("LogisticRegression", {"max_iter": 200}, 42).fit(
        raw_features, raw_target
    )
    model_path = tmp_path / "model.pkl"
    trainer.save_model(str(model_path))

    reloaded = ModelTrainer.load_model(str(model_path))

    assert (reloaded.predict(raw_features) == trainer.predict(raw_features)).all()


def test_metrics_cover_imbalance(raw_features, raw_target):
    trainer = ModelTrainer("RandomForest", {"n_estimators": 25}, 42).fit(
        raw_features, raw_target
    )
    metrics = evaluate_model(
        raw_target, trainer.predict(raw_features), trainer.predict_proba(raw_features)
    )

    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"} <= set(
        metrics
    )
    assert metrics["roc_auc"] > 0.5


def test_balanced_class_weight_avoids_all_zero_predictions(raw_features, raw_target):
    """The original code always predicted the majority class."""
    trainer = ModelTrainer("LogisticRegression", {"max_iter": 500}, 42).fit(
        raw_features, raw_target
    )

    assert trainer.predict(raw_features).sum() > 0
