from pathlib import Path

from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.data.loader import load_raw_data
from ecommerce_buy_predictor.data.preprocess import split_data
from ecommerce_buy_predictor.pipeline.params import load_params


def preprocess_stage() -> None:
    """DVC stage: load the raw dataset and write a stratified train/test split.

    Scaling and encoding are *not* applied here. They belong to the model
    pipeline, so the artefact that gets served already knows how to transform
    raw input.
    """
    params = load_params()
    raw_path = Path(settings.data_raw_path)
    processed_dir = Path(settings.data_processed_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}. Run 'dvc pull', "
            "'python scripts/download_dataset.py' or "
            "'python scripts/generate_sample_data.py' first."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=params["split"]["test_size"],
        random_seed=params["random_seed"],
    )

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    print(
        f"Preprocessing complete: {len(X_train)} train / {len(X_test)} test rows "
        f"saved to {processed_dir}"
    )


if __name__ == "__main__":
    preprocess_stage()
