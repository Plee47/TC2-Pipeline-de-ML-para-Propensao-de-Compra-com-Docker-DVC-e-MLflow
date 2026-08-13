"""Download the real *Online Shoppers Purchasing Intention* dataset.

Source: UCI Machine Learning Repository, dataset 468 (the same data published
on Kaggle). 12,330 sessions, ~15.5% positives.

    python scripts/download_dataset.py

After downloading, version it and rerun the pipeline:

    dvc add data/raw/online_shoppers_intention.csv
    dvc repro
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
CSV_NAME = "online_shoppers_intention.csv"
DEFAULT_OUTPUT = Path("data/raw") / CSV_NAME
EXPECTED_ROWS = 12330


def download_dataset(output_path: Path = DEFAULT_OUTPUT, url: str = DATASET_URL) -> Path:
    """Download the dataset zip and extract the CSV.

    Args:
        output_path: Where to write the CSV.
        url: Source archive.

    Returns:
        Path to the extracted CSV.

    Raises:
        RuntimeError: If the archive does not contain the expected CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV inside the archive: {archive.namelist()}")
        output_path.write_bytes(archive.read(members[0]))

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    path = download_dataset(args.output)

    rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    print(f"Saved {path} ({path.stat().st_size:,} bytes, {rows} rows)")
    if rows != EXPECTED_ROWS:
        print(
            f"Warning: expected {EXPECTED_ROWS} rows, got {rows}. "
            "The upstream file may have changed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
