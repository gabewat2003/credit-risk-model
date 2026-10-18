"""KS statistic + decile table shape checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import compute_ks_statistic, generate_decile_table


def test_ks_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    assert compute_ks_statistic(y, p) == 1.0


def test_ks_random_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=5_000)
    p = rng.random(5_000)
    assert compute_ks_statistic(y, p) < 0.1


def test_decile_table_shape_and_counts():
    rng = np.random.default_rng(42)
    n = 1_000
    df = pd.DataFrame({"id": range(n)})
    y = pd.Series(rng.integers(0, 2, size=n))
    p = rng.random(n)
    table = generate_decile_table(y, p, df)
    assert list(table["decile"]) == list(range(1, 11))
    assert table["loan_count"].sum() == n


def test_decile_table_monotone_on_perfect_signal():
    n = 1_000
    rng = np.random.default_rng(7)
    p = np.linspace(0.01, 0.99, n)
    # Default probability equals pd_hat → actual rate must rise across deciles.
    y = pd.Series((rng.random(n) < p).astype(int))
    df = pd.DataFrame({"id": range(n)})
    table = generate_decile_table(y, p, df)
    rates = table["actual_default_rate"].to_numpy()
    assert (rates[1:] >= rates[:-1] - 0.05).all()  # allow small noise
