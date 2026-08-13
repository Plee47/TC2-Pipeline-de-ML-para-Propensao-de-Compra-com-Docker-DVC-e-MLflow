import pandas as pd
import pytest

from ecommerce_buy_predictor.data.preprocess import build_preprocessor, split_data
from ecommerce_buy_predictor.features.build_features import add_engineered_features
from ecommerce_buy_predictor.features.schema import ENGINEERED_COLUMNS


def test_split_is_stratified(raw_dataset: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(raw_dataset, random_seed=42)

    assert len(X_train) + len(X_test) == len(raw_dataset)
    assert len(y_train) == len(X_train)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.03)


def test_split_is_deterministic(raw_dataset: pd.DataFrame):
    first, _, _, _ = split_data(raw_dataset, random_seed=42)
    second, _, _, _ = split_data(raw_dataset, random_seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_split_requires_target(raw_dataset: pd.DataFrame):
    with pytest.raises(ValueError, match="not found in data"):
        split_data(raw_dataset.drop(columns=["Revenue"]))


def test_target_is_not_a_feature(raw_dataset: pd.DataFrame):
    X_train, _, _, _ = split_data(raw_dataset)

    assert "Revenue" not in X_train.columns


def test_engineered_features_are_added(raw_dataset: pd.DataFrame):
    enriched = add_engineered_features(raw_dataset)

    assert all(column in enriched.columns for column in ENGINEERED_COLUMNS)
    assert (enriched["TotalPages"] >= raw_dataset["ProductRelated"]).all()
    assert enriched["AvgDurationPerPage"].notna().all()


def test_engineered_features_survive_zero_pages():
    empty_session = pd.DataFrame(
        {
            "Administrative": [0.0],
            "Administrative_Duration": [0.0],
            "Informational": [0.0],
            "Informational_Duration": [0.0],
            "ProductRelated": [0.0],
            "ProductRelated_Duration": [0.0],
        }
    )

    result = add_engineered_features(empty_session)

    assert result["AvgDurationPerPage"].iloc[0] == 0.0


def test_unknown_category_does_not_raise(raw_dataset: pd.DataFrame):
    """LabelEncoder used to blow up here; OneHotEncoder must not."""
    X_train, X_test, y_train, _ = split_data(raw_dataset)
    preprocessor = build_preprocessor()
    preprocessor.fit(add_engineered_features(X_train))

    unseen = add_engineered_features(X_test).copy()
    unseen.loc[unseen.index[0], "Month"] = "Jan"
    unseen.loc[unseen.index[0], "VisitorType"] = "Astronaut"

    transformed = preprocessor.transform(unseen)

    assert len(transformed) == len(unseen)


def test_numeric_columns_are_scaled(raw_dataset: pd.DataFrame):
    X_train, _, _, _ = split_data(raw_dataset)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(add_engineered_features(X_train))
    scaled = pd.DataFrame(transformed).iloc[:, :10]

    assert scaled.mean().abs().max() < 1e-9
    assert scaled.std(ddof=0).round(3).eq(1.0).all()
