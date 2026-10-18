"""
src/split.py
Handles time-based splitting of the feature dataset into train, validation, and test splits.
"""

import pandas as pd
import sqlalchemy


def load_and_split_data(db_uri: str):
    engine = sqlalchemy.create_engine(db_uri)
    query = "SELECT * FROM features ORDER BY issue_d;"
    df = pd.read_sql(query, engine)

    # Ensure issue_d is parsed as datetime
    df["issue_d"] = pd.to_datetime(df["issue_d"])

    # Time-based split
    train = df[df["issue_d"] < "2015-01-01"].copy()
    valid = df[(df["issue_d"] >= "2015-01-01") & (df["issue_d"] < "2016-07-01")].copy()
    test = df[df["issue_d"] >= "2016-07-01"].copy()

    return train, valid, test
