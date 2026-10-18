"""Run the SQL feature pipeline against Postgres."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, SQL_DIR

logger = logging.getLogger(__name__)

SQL_FILES = ("01_build_target.sql", "02_features.sql")


def build_features() -> None:
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        for name in SQL_FILES:
            path = SQL_DIR / name
            logger.info("Executing %s", path.name)
            conn.execute(text(path.read_text(encoding="utf-8")))
    logger.info("Feature pipeline complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_features()


if __name__ == "__main__":
    main()
