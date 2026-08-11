"""Generate sample online shoppers dataset for testing pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic online shoppers dataset.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic data.
    """
    np.random.seed(seed)

    data = {
        "Administrative": np.random.exponential(2, n_samples),
        "AdministrativeDuration": np.random.exponential(100, n_samples),
        "Informational": np.random.exponential(1, n_samples),
        "InformationalDuration": np.random.exponential(50, n_samples),
        "ProductRelated": np.random.exponential(5, n_samples),
        "ProductRelatedDuration": np.random.exponential(500, n_samples),
        "BounceRate": np.random.uniform(0, 100, n_samples),
        "ExitRate": np.random.uniform(0, 100, n_samples),
        "PageValues": np.random.exponential(1, n_samples),
        "SpecialDay": np.random.choice([0, 1, 2, 3, 4], n_samples),
        "Month": np.random.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], n_samples),
        "OperatingSystems": np.random.randint(1, 10, n_samples),
        "Browser": np.random.randint(1, 10, n_samples),
        "Region": np.random.randint(1, 10, n_samples),
        "TrafficType": np.random.randint(1, 30, n_samples),
        "VisitorType": np.random.choice(["New_Visitor", "Returning_Visitor", "Other"], n_samples),
        "Weekend": np.random.choice([True, False], n_samples),
    }

    df = pd.DataFrame(data)

    df["Revenue"] = np.random.choice(
        [False, True],
        n_samples,
        p=[0.85, 0.15]
    )

    return df


if __name__ == "__main__":
    output_path = Path("data/raw/online_shoppers_intention.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sample_data(n_samples=5000)
    df.to_csv(output_path, index=False)

    print(f"Sample data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Revenue distribution:\n{df['Revenue'].value_counts()}")
