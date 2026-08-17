from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.data.preprocess import build_preprocessor
from ecommerce_buy_predictor.features.build_features import add_engineered_features

#: Estimators available to the training stage, keyed by the name used in
#: ``params.yaml`` and in the MLflow run name.
ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
}


def build_model(estimator_name: str, params: dict[str, Any], random_seed: int) -> Pipeline:
    """Build a full pipeline: feature engineering, preprocessing and estimator.

    The returned object takes **raw** features as input, which is what the API
    receives, so there is no risk of training/serving skew.

    Args:
        estimator_name: Key of :data:`ESTIMATORS`.
        params: Hyperparameters forwarded to the estimator.
        random_seed: Seed forwarded to the estimator.

    Returns:
        Unfitted sklearn Pipeline.
    """
    if estimator_name not in ESTIMATORS:
        raise ValueError(
            f"Unknown estimator '{estimator_name}'. Available: {sorted(ESTIMATORS)}"
        )

    estimator = ESTIMATORS[estimator_name](
        random_state=random_seed, class_weight="balanced", **params
    )

    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_engineered_features)),
            ("preprocessor", build_preprocessor()),
            ("estimator", estimator),
        ]
    )


class ModelTrainer:
    """Thin wrapper around a model pipeline (fit, predict, persist)."""

    def __init__(
        self,
        estimator_name: str,
        params: dict[str, Any],
        random_seed: int = settings.random_seed,
    ):
        self.estimator_name = estimator_name
        self.random_seed = random_seed
        self.model: Pipeline = build_model(estimator_name, params, random_seed)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelTrainer":
        """Fit the pipeline on raw features."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict class labels."""
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Predict the probability of the positive class."""
        return pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)

    def save_model(self, filepath: str) -> None:
        """Persist the fitted pipeline to disk."""
        joblib.dump(self.model, filepath)

    @staticmethod
    def load_model(filepath: str) -> Pipeline:
        """Load a pipeline previously saved by :meth:`save_model`."""
        return joblib.load(filepath)
