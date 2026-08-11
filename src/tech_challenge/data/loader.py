import pandas as pd
from pathlib import Path
from typing import Tuple


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw CSV dataset.

    Args:
        filepath: Path to the raw data CSV file.

    Returns:
        DataFrame with raw data.
    """
    return pd.read_csv(filepath)
