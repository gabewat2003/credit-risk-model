"""Elasticity: volume multiplier math + sensitivity table shape."""

from __future__ import annotations

import pandas as pd

from src.elasticity import (
    ELASTICITY_GRID,
    HEADLINE_BETA,
    apply_elasticity,
    sensitivity_table,
    volume_multiplier,
)


def _toy_pricing() -> pd.DataFrame:
    # Each decile set so the proposed APR produces positive profit per loan,
    # i.e. losing volume costs profit (otherwise churn can look like a free lunch).
    return pd.DataFrame(
        {
            "decile": range(1, 4),
            "loan_count": [1000, 1000, 1000],
            "actual_default_rate": [0.05, 0.15, 0.30],
            "mean_loan_amnt": [10_000.0] * 3,
            "mean_term_months": [36] * 3,
            "actual_apr": [8.0, 12.0, 15.0],
            "break_even_apr": [3.0, 10.0, 21.5],
            "proposed_apr": [6.0, 13.0, 25.0],
        }
    )


def _toy_predictions() -> pd.DataFrame:
    """Row-level synthetic predictions matching the toy pricing deciles."""
    rows = []
    for d, pd_rate in zip([1, 2, 3], [0.05, 0.15, 0.30], strict=True):
        for i in range(1000):
            rows.append(
                {
                    "decile": d,
                    "loan_amnt": 10_000.0,
                    "term_months": 36,
                    "int_rate": [8.0, 12.0, 15.0][d - 1],
                    "defaulted": 1 if i < int(pd_rate * 1000) else 0,
                    "pd_hat": pd_rate,
                    "installment": 332.14,
                }
            )
    return pd.DataFrame(rows)


def test_zero_elasticity_keeps_volume():
    # β = 0 means fully inelastic: everyone accepts regardless of price.
    assert volume_multiplier(20.0, 10.0, 0.0) == 1.0
    assert volume_multiplier(5.0, 10.0, 0.0) == 1.0


def test_rate_increase_sheds_volume():
    # β = -4, raise APR 10 -> 20 (double). Expect (20/10)^-4 = 0.0625.
    assert volume_multiplier(20.0, 10.0, -4.0) < 1.0
    assert abs(volume_multiplier(20.0, 10.0, -4.0) - 0.0625) < 1e-6


def test_rate_decrease_capped_at_one():
    # Lower APR can't grow observed pool.
    assert volume_multiplier(5.0, 10.0, -4.0) == 1.0


def test_steeper_beta_loses_more_volume_at_rate_hike():
    rates = [volume_multiplier(20.0, 10.0, b) for b in (-2.0, -4.0, -6.0, -8.0)]
    assert all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))


def test_apply_elasticity_elastic_count_bounded():
    out = apply_elasticity(_toy_pricing(), -4.0)
    assert (out["elastic_count"] >= 0).all()
    assert (out["elastic_count"] <= out["loan_count"]).all()


def test_sensitivity_table_has_headline_and_endpoints():
    table = sensitivity_table(_toy_predictions(), _toy_pricing(), betas=ELASTICITY_GRID)
    assert 0.0 in table["beta"].to_numpy()
    assert HEADLINE_BETA in table["beta"].to_numpy()
    assert "nobody accepts" in table["note"].to_numpy()


def test_sensitivity_profit_monotone_in_elasticity():
    # More-negative β (more rate-sensitive) → less elastic profit.
    table = sensitivity_table(_toy_predictions(), _toy_pricing(), betas=(-2.0, -4.0, -6.0, -8.0))
    finite = table[table["beta"].between(-8.1, 0.1) & (table["beta"] != 0.0)]
    profits = finite.sort_values("beta", ascending=False)["elastic_profit_M"].to_numpy()
    assert all(profits[i] >= profits[i + 1] for i in range(len(profits) - 1))
