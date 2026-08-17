"""Generate a synthetic dataset with the schema of *Online Shoppers Intention*.

This is a **smoke-test dataset**, not a substitute for the real one: it exists
so the pipeline can be exercised end to end without network access. Unlike a
purely random sample, the target depends on the features, so the metrics it
produces are non-degenerate and regressions in the pipeline are visible.

For anything you intend to report, use the real dataset:
``python scripts/download_dataset.py``.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_buy_predictor.config import settings

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HIGH_SEASON = {"Sep", "Oct", "Nov", "Dec"}
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_sample_data(n_samples: int = 5000, seed: int = settings.random_seed) -> pd.DataFrame:
    """Generate a synthetic dataset whose target depends on the features.

    Args:
        n_samples: Number of sessions to generate.
        seed: Seed for reproducibility. Defaults to the same
            ``random_seed`` used by the real training pipeline
            (``params.yaml`` / :class:`~ecommerce_buy_predictor.config.Settings`).

    Returns:
        DataFrame with the same columns and dtypes as the real dataset.
    """
    rng = np.random.default_rng(seed)

    administrative = rng.poisson(2.3, n_samples)
    informational = rng.poisson(0.5, n_samples)
    product_related = rng.poisson(32, n_samples)

    df = pd.DataFrame(
        {
            "Administrative": administrative,
            "Administrative_Duration": administrative * rng.uniform(0, 70, n_samples),
            "Informational": informational,
            "Informational_Duration": informational * rng.uniform(0, 90, n_samples),
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related * rng.uniform(0, 75, n_samples),
            "BounceRates": rng.beta(1.2, 40, n_samples),
            "ExitRates": rng.beta(2.0, 40, n_samples),
            "PageValues": rng.exponential(6, n_samples) * rng.binomial(1, 0.25, n_samples),
            "SpecialDay": rng.choice(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                n_samples,
                p=[0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
            ),
            "Month": rng.choice(MONTHS, n_samples),
            "OperatingSystems": rng.integers(1, 9, n_samples),
            "Browser": rng.integers(1, 14, n_samples),
            "Region": rng.integers(1, 10, n_samples),
            "TrafficType": rng.integers(1, 21, n_samples),
            "VisitorType": rng.choice(VISITOR_TYPES, n_samples, p=[0.85, 0.14, 0.01]),
            "Weekend": rng.random(n_samples) < 0.23,
        }
    )

    # Purchase intention driven by the features, roughly mirroring the signal
    # of the real dataset (PageValues dominates, high exit rates hurt).
    logit = (
        -3.45
        + 0.85 * np.log1p(df["PageValues"])
        + 0.35 * np.log1p(df["ProductRelated"])
        - 12.0 * df["ExitRates"]
        - 0.60 * df["SpecialDay"]
        + 0.45 * df["Month"].isin(HIGH_SEASON)
        + 0.30 * (df["VisitorType"] == "Returning_Visitor")
        + 0.20 * df["Weekend"]
    )
    df["Revenue"] = rng.random(n_samples) < _sigmoid(logit.to_numpy())

    return df


if __name__ == "__main__":
    output_path = Path("data/raw/online_shoppers_intention.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_sample_data()
    data.to_csv(output_path, index=False)

    print(f"Synthetic sample data saved to {output_path}")
    print(f"Shape: {data.shape}")
    print(f"Positive rate: {data['Revenue'].mean():.1%}")
