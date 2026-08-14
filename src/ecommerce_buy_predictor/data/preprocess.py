
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.features.schema import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_ID_COLUMNS,
    ENGINEERED_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the feature transformer used inside the model pipeline.

    Nominal columns (including the integer *codes* such as ``Browser``) are
    one-hot encoded with ``handle_unknown="ignore"``, so a category unseen at
    training time degrades gracefully instead of raising at predict time.

    Returns:
        Unfitted ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS + ENGINEERED_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS + CATEGORICAL_ID_COLUMNS,
            ),
            ("boolean", "passthrough", BOOLEAN_COLUMNS),
        ],
        remainder="drop",
    )


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_seed: int = settings.random_seed,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw data into stratified train/test sets.

    No scaling or encoding happens here: those live in the model pipeline so
    that training and serving share exactly the same transformations.

    Args:
        df: Raw DataFrame.
        target_col: Name of the target column.
        test_size: Fraction held out for testing.
        random_seed: Seed for the split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
