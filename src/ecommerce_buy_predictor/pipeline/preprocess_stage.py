from pathlib import Path
import pickle
from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.data.loader import load_raw_data
from ecommerce_buy_predictor.data.preprocess import Preprocessor


def preprocess_stage() -> None:
    """DVC pipeline stage: load and preprocess data."""
    raw_path = Path(settings.data_raw_path)
    processed_dir = Path(settings.data_processed_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    preprocessor = Preprocessor(random_seed=settings.random_seed)

    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    with open(processed_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("Preprocessing complete. Data saved to", processed_dir)


if __name__ == "__main__":
    preprocess_stage()
