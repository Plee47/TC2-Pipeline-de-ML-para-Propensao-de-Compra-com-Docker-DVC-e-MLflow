import pandas as pd
import pytest
from tech_challenge.data.preprocess import Preprocessor


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": ["A", "B", "A", "B", "A", "B"],
        "Revenue": [0, 1, 0, 1, 0, 1],
    })


def test_preprocessor_fit_transform(sample_data):
    preprocessor = Preprocessor(random_seed=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_preprocessor_stratified_split(sample_data):
    preprocessor = Preprocessor(random_seed=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)

    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)

    assert 0.2 <= train_ratio <= 0.8
    assert 0.2 <= test_ratio <= 0.8
