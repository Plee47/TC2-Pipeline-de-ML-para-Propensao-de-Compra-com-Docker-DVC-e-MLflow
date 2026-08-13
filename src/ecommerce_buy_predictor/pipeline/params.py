from pathlib import Path
from typing import Any

import yaml

PARAMS_FILE = Path("params.yaml")


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    """Load the pipeline hyperparameters.

    Args:
        path: Path to ``params.yaml``.

    Returns:
        Parsed parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
