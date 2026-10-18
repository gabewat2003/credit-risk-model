"""LGD sensitivity: break-even APRs rise monotonically with LGD."""

from __future__ import annotations

from src.pricing import break_even_apr


def test_be_apr_monotone_in_lgd():
    rates = [break_even_apr(0.20, 10_000, 36, lgd=lgd) for lgd in (0.5, 0.7, 0.85, 1.0)]
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


def test_be_apr_zero_lgd_floor():
    # LGD = 0 means no loss on default; break-even APR should be 0.
    assert break_even_apr(0.20, 10_000, 36, lgd=0.0) == 0.0
