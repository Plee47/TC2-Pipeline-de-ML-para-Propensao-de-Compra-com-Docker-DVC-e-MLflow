from pathlib import Path

import pandas as pd


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw CSV dataset.

    Args:
        filepath: Path to the raw data CSV file.

    Returns:
        DataFrame with raw data.
    """
    return pd.read_csv(filepath)
