"""
src/load_data.py
Robustly loads the LendingClub CSV into PostgreSQL using chunked pandas ingestion with string dtypes.
"""

import os
import pandas as pd
from sqlalchemy import create_engine

DB_URI = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/lending"
)
CSV_PATH = "data/raw/loan.csv"


def load_csv_to_postgres():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV file not found at {
                CSV_PATH}. Place your CSV file there first."
        )

    print("Initializing database engine...")
    engine = create_engine(DB_URI)

    print(f"Reading and loading CSV in chunks from {CSV_PATH}...")
    chunk_size = 50_000
    first = True

    # Force all columns to read as string to prevent type-mismatch bugs across chunks
    for i, chunk in enumerate(
        pd.read_csv(
            CSV_PATH,
            chunksize=chunk_size,
            low_memory=False,
            dtype=str,
            on_bad_lines="skip",
        )
    ):
        mode = "replace" if first else "append"
        chunk.to_sql("loans_raw", engine, if_exists=mode, index=False)
        first = False
        print(f"Loaded chunk {
              i + 1} (~{(i + 1) * chunk_size:,} rows processed)...")

    print("Raw data successfully loaded into PostgreSQL table `loans_raw`.")


if __name__ == "__main__":
    load_csv_to_postgres()
