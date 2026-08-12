import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Any
import joblib


class ModelTrainer:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.model: Any = None

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> None:
        """Train Logistic Regression model."""
        self.model = LogisticRegression(random_state=self.random_seed, **kwargs)
        self.model.fit(X_train, y_train)

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> None:
        """Train Random Forest model."""
        self.model = RandomForestClassifier(
            random_state=self.random_seed, **kwargs
        )
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return pd.Series(self.model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return pd.DataFrame(self.model.predict_proba(X))

    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
