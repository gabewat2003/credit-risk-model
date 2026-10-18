"""Load the LendingClub CSV into Postgres via COPY FROM STDIN.

psycopg3's ``cursor.copy()`` context manager streams the file through the
server-side COPY protocol — ~10x faster than ``pandas.to_sql`` on this dataset
and matches the pattern the README advertises.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import psycopg

from src.config import DATA_RAW, DATABASE_URL, RAW_TABLE

logger = logging.getLogger(__name__)


def _psycopg_url(sqlalchemy_url: str) -> str:
    """Strip the SQLAlchemy ``+psycopg`` dialect suffix for raw psycopg3 use."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _create_raw_table(conn: psycopg.Connection, columns: list[str]) -> None:
    """Drop and recreate ``loans_raw`` with every column typed as TEXT.

    All typing happens later in ``sql/01_build_target.sql`` where column names
    and casts live together — keeps load fast and tolerant of messy source data.
    """
    quoted = ", ".join(f'"{c}" TEXT' for c in columns)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {RAW_TABLE} CASCADE;")
        cur.execute(f"CREATE TABLE {RAW_TABLE} ({quoted});")
    conn.commit()


def load_csv_to_postgres(csv_path: Path = DATA_RAW) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. See data/README.md for the Kaggle download."
        )

    columns = _read_header(csv_path)
    logger.info("CSV has %d columns", len(columns))

    conn_url = _psycopg_url(DATABASE_URL)
    with psycopg.connect(conn_url) as conn:
        _create_raw_table(conn, columns)

        logger.info("Streaming CSV into %s via COPY...", RAW_TABLE)
        copy_sql = (
            f"COPY {RAW_TABLE} FROM STDIN "
            "WITH (FORMAT CSV, HEADER TRUE, QUOTE '\"', ESCAPE '\"')"
        )
        with conn.cursor() as cur, cur.copy(copy_sql) as cp:
            with csv_path.open("rb") as f:
                while chunk := f.read(1024 * 1024):  # 1 MiB per iteration
                    cp.write(chunk)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {RAW_TABLE};")
            row_count = cur.fetchone()[0]

    logger.info("Loaded %s rows into %s.", f"{row_count:,}", RAW_TABLE)
    return row_count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_csv_to_postgres()


if __name__ == "__main__":
    main()
