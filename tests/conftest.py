import pandas as pd
import pytest

from ecommerce_buy_predictor.features.schema import FEATURE_COLUMNS

SESSION = {
    "Administrative": 2.0,
    "Administrative_Duration": 80.0,
    "Informational": 0.0,
    "Informational_Duration": 0.0,
    "ProductRelated": 31.0,
    "ProductRelated_Duration": 1200.5,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "PageValues": 12.4,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 3,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


@pytest.fixture
def session_payload() -> dict:
    """A single valid API payload."""
    return dict(SESSION)


@pytest.fixture
def raw_dataset() -> pd.DataFrame:
    """Small dataset with the real schema and a learnable target."""
    from scripts.generate_sample_data import generate_sample_data

    return generate_sample_data(n_samples=400, seed=7)


@pytest.fixture
def raw_features(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    return raw_dataset[FEATURE_COLUMNS]


@pytest.fixture
def raw_target(raw_dataset: pd.DataFrame) -> pd.Series:
    return raw_dataset["Revenue"].astype(int)
