"""Single source of truth for the dataset schema.

Column names follow the original *Online Shoppers Purchasing Intention*
dataset (UCI / Kaggle), so the same names are used by the data generator,
the model pipeline and the API request payload.
"""

import pandas as pd

TARGET_COLUMN = "Revenue"

#: Continuous behavioural counters and rates.
NUMERIC_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

#: Nominal text columns.
CATEGORICAL_COLUMNS = [
    "Month",
    "VisitorType",
]

#: Integer codes that identify a category, not a magnitude.
CATEGORICAL_ID_COLUMNS = [
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

#: Boolean flags.
BOOLEAN_COLUMNS = [
    "Weekend",
]

#: Features derived by :func:`ecommerce_buy_predictor.features.build_features`.
ENGINEERED_COLUMNS = [
    "TotalPages",
    "TotalDuration",
    "AvgDurationPerPage",
]

#: Every feature the model expects as input, in dataset order.
FEATURE_COLUMNS = (
    NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + CATEGORICAL_ID_COLUMNS + BOOLEAN_COLUMNS
)

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

#: Dtypes the model signature was inferred from. Requests are cast to these
#: before predicting, otherwise MLflow rejects the payload.
FEATURE_DTYPES = {
    **{column: "float64" for column in NUMERIC_COLUMNS},
    **{column: "object" for column in CATEGORICAL_COLUMNS},
    **{column: "int64" for column in CATEGORICAL_ID_COLUMNS},
    **{column: "bool" for column in BOOLEAN_COLUMNS},
}


def to_model_frame(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the exact column order and dtypes of training.

    Args:
        records: One dict per row, keyed by feature name.

    Returns:
        DataFrame ready to be passed to the model.
    """
    return pd.DataFrame(records, columns=FEATURE_COLUMNS).astype(FEATURE_DTYPES)
