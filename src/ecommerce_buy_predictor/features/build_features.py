import pandas as pd

from ecommerce_buy_predictor.features.schema import ENGINEERED_COLUMNS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add session-level aggregates derived from the raw page counters.

    Runs as the first step of the model pipeline, so training and serving
    always see the same derived columns.

    Args:
        df: DataFrame with the raw feature columns.

    Returns:
        Copy of the input with the columns in ``ENGINEERED_COLUMNS`` added.
    """
    df = df.copy()

    df["TotalPages"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )
    df["TotalDuration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )
    df["AvgDurationPerPage"] = df["TotalDuration"] / df["TotalPages"].where(
        df["TotalPages"] > 0, 1
    )

    return df[[c for c in df.columns if c not in ENGINEERED_COLUMNS] + ENGINEERED_COLUMNS]
