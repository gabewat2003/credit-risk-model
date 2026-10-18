"""Risk-based pricing: amortization + break-even APR solver."""

from __future__ import annotations

import pandas as pd

from src.config import LGD
from src.pricing import amortized_installment, break_even_apr, build_pricing_table


def test_amortized_installment_matches_textbook():
    # $10,000 @ 12% / 36mo -> installment $332.14 (verified against standard amortization).
    assert abs(amortized_installment(10_000, 12.0, 36) - 332.14) < 0.01


def test_amortized_zero_apr_is_straight_line():
    assert amortized_installment(1200, 0.0, 12) == 100.0


def test_break_even_apr_zero_pd_is_zero():
    assert break_even_apr(0.0, 10_000, 36) == 0.0


def test_break_even_apr_rises_with_pd():
    rates = [break_even_apr(p, 10_000, 36) for p in [0.05, 0.15, 0.30, 0.45]]
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


def test_break_even_apr_zero_profit_at_be():
    # At the break-even APR, expected profit per loan should be ~0.
    pd_rate, loan, term = 0.15, 10_000, 36
    be = break_even_apr(pd_rate, loan, term)
    installment = amortized_installment(loan, be, term)
    total_interest = installment * term - loan
    expected_profit = (1 - pd_rate) * total_interest - pd_rate * LGD * loan
    assert abs(expected_profit) < 1.0  # within $1 numerical tolerance


def test_build_pricing_table_shape():
    # Synthetic 1000 loans across 10 PD buckets.
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1000
    pd_hat = np.linspace(0.02, 0.5, n)
    df = pd.DataFrame(
        {
            "pd_hat": pd_hat,
            "defaulted": (rng.random(n) < pd_hat).astype(int),
            "loan_amnt": [10_000.0] * n,
            "installment": [332.14] * n,
            "term_months": [36] * n,
            "int_rate": [13.0] * n,
        }
    )
    table = build_pricing_table(df, n_deciles=10)
    assert list(table["decile"]) == list(range(1, 11))
    # Overall trend should rise: last decile APR well above first, allowing noise.
    assert table["break_even_apr"].iloc[-1] > table["break_even_apr"].iloc[0]
