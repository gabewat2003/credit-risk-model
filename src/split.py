"""Time-based train/valid/test split from the `features` table."""

from __future__ import annotations

import pandas as pd
import sqlalchemy

from src.config import DATABASE_URL, TRAIN_END, VALID_END


def load_and_split_data(
    db_uri: str = DATABASE_URL,
    train_end: str = TRAIN_END,
    valid_end: str = VALID_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``features`` ordered by issue date and split chronologically.

    Returns ``(train, valid, test)`` with ``issue_d`` parsed as datetime.
    No row appears in more than one split; split boundaries are half-open:
    ``train.issue_d < train_end <= valid.issue_d < valid_end <= test.issue_d``.
    """
    engine = sqlalchemy.create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM features ORDER BY issue_d;", engine)
    df["issue_d"] = pd.to_datetime(df["issue_d"])

    train = df[df["issue_d"] < train_end].copy()
    valid = df[(df["issue_d"] >= train_end) & (df["issue_d"] < valid_end)].copy()
    test = df[df["issue_d"] >= valid_end].copy()
    return train, valid, test
