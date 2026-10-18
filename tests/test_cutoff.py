"""Cutoff sweep math: expected profit on a tiny hand-built frame."""

from __future__ import annotations

import pandas as pd

from src.config import LGD
from src.cutoff import _augment, sweep_cutoffs


def _toy_predictions() -> pd.DataFrame:
    # 10 loans, equal size, mix of good/bad and low/high PD.
    return pd.DataFrame(
        {
            "loan_amnt": [10_000] * 10,
            "int_rate": [12.0] * 10,
            "term_months": [36] * 10,
            "defaulted": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "pd_hat": [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.80],
        }
    )


def test_augment_interest_formula():
    aug = _augment(_toy_predictions())
    expected = 10_000 * (12.0 / 100.0) * (36 / 12.0)
    assert aug["interest_earned"].iloc[0] == expected


def test_realized_profit_default_pays_lgd():
    aug = _augment(_toy_predictions())
    defaults = aug[aug["defaulted"] == 1]
    assert (defaults["realized_profit"] == -LGD * 10_000).all()


def test_sweep_monotone_approval_rate():
    aug = _augment(_toy_predictions())
    sweep = sweep_cutoffs(aug)
    rates = sweep["approval_rate"].to_numpy()
    assert (rates[1:] >= rates[:-1]).all()


def test_sweep_bounded_approval():
    aug = _augment(_toy_predictions())
    sweep = sweep_cutoffs(aug)
    assert sweep["approval_rate"].min() >= 0.0
    assert sweep["approval_rate"].max() <= 1.0
