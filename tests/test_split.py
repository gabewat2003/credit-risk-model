"""Time-split sanity: no row overlap, chronological boundaries honored."""

from __future__ import annotations

import pandas as pd


def _fake_features(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2013-01-01", "2018-01-01", periods=n)
    return pd.DataFrame({"issue_d": dates, "defaulted": [0, 1] * (n // 2)})


def _split(df: pd.DataFrame, train_end: str, valid_end: str):
    df = df.copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    train = df[df["issue_d"] < train_end]
    valid = df[(df["issue_d"] >= train_end) & (df["issue_d"] < valid_end)]
    test = df[df["issue_d"] >= valid_end]
    return train, valid, test


def test_split_counts_match_total():
    df = _fake_features(100)
    train, valid, test = _split(df, "2015-01-01", "2016-07-01")
    assert len(train) + len(valid) + len(test) == len(df)


def test_split_no_overlap():
    df = _fake_features(100)
    train, valid, test = _split(df, "2015-01-01", "2016-07-01")
    assert train["issue_d"].max() < pd.Timestamp("2015-01-01")
    assert valid["issue_d"].min() >= pd.Timestamp("2015-01-01")
    assert valid["issue_d"].max() < pd.Timestamp("2016-07-01")
    assert test["issue_d"].min() >= pd.Timestamp("2016-07-01")


def test_split_is_chronological():
    df = _fake_features(100)
    train, valid, test = _split(df, "2015-01-01", "2016-07-01")
    if len(train) and len(valid):
        assert train["issue_d"].max() < valid["issue_d"].min()
    if len(valid) and len(test):
        assert valid["issue_d"].max() < test["issue_d"].min()
