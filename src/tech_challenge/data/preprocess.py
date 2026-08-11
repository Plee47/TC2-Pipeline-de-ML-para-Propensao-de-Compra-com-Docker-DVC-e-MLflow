import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
from tech_challenge.config import settings


class Preprocessor:
    def __init__(self, random_seed: int = settings.random_seed):
        self.random_seed = random_seed
        self.scaler: StandardScaler | None = None
        self.encoders: dict[str, LabelEncoder] = {}

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "Revenue"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit preprocessor and return processed data.

        Args:
            df: Raw DataFrame.
            target_col: Name of target column.

        Returns:
            X_train, X_test, y_train, y_test
        """
        df = df.copy()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )

        X_train = self._preprocess_features(X_train, fit=True)
        X_test = self._preprocess_features(X_test, fit=False)

        return X_train, X_test, y_train, y_test

    def _preprocess_features(
        self, X: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """Preprocess feature columns (encode, scale)."""
        X = X.copy()

        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            if fit:
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))

        if fit:
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            if self.scaler is None:
                raise ValueError("Preprocessor not fitted yet")
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X
