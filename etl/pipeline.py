"""
ETL pipeline orchestrator.

Run this file directly to execute the full pipeline:
    python -m etl.pipeline

Output: 4 Parquet files written to data/processed/
Falls back to CSV if pyarrow is not installed.
"""

import time
from pathlib import Path

from etl.loader import load_all
from etl.cleaner import clean_all
from etl.transformer import build_all

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

try:
    import pyarrow  # noqa: F401
    USE_PARQUET = True
except ImportError:
    USE_PARQUET = False


def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("Loading raw CSVs...")
    raw = load_all()

    print("Cleaning data...")
    cleaned = clean_all(raw)

    print("Building analytical tables...")
    tables = build_all(cleaned)

    fmt = "parquet" if USE_PARQUET else "csv"
    print(f"Writing processed files ({fmt})...")
    for name, df in tables.items():
        if USE_PARQUET:
            path = PROCESSED_DIR / f"{name}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = PROCESSED_DIR / f"{name}.csv"
            df.to_csv(path, index=False)
        print(f"  {name}: {len(df):,} rows -> {path.name}")

    elapsed = round(time.time() - start, 1)
    print(f"\nPipeline complete in {elapsed}s")
    return tables


if __name__ == "__main__":
    run()
