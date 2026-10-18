"""Fair-lending audit: disparate-impact ratios compute correctly."""

from __future__ import annotations

import pandas as pd

from src.fairness import DISPARATE_IMPACT_FLOOR, MIN_GROUP_SIZE, audit_group


def _synthetic_frame(n_per_group: int = 1000) -> pd.DataFrame:
    """Two groups: 'ref' always below cutoff, 'disparate' mostly above."""
    rng_pd_ref = [0.1] * n_per_group  # all approved at cutoff 0.354
    rng_pd_disp = [0.5] * n_per_group  # all declined
    return pd.DataFrame(
        {
            "home_ownership": ["ref"] * n_per_group + ["disparate"] * n_per_group,
            "pd_hat": rng_pd_ref + rng_pd_disp,
            "defaulted": [0, 1] * n_per_group,
            "loan_amnt": [10_000] * (2 * n_per_group),
        }
    )


def test_audit_sorts_worst_first():
    audit = audit_group(_synthetic_frame(), "home_ownership", cutoff_pd=0.354)
    assert audit.iloc[0]["home_ownership"] == "disparate"
    assert audit.iloc[0]["disparate_impact_ratio"] == 0.0


def test_four_fifths_flag():
    audit = audit_group(_synthetic_frame(), "home_ownership", cutoff_pd=0.354)
    assert bool(audit.iloc[0]["flagged"]) is True
    assert bool(audit.iloc[-1]["flagged"]) is False
    assert audit.iloc[0]["disparate_impact_ratio"] < DISPARATE_IMPACT_FLOOR


def test_small_groups_dropped():
    df = _synthetic_frame(n_per_group=100)  # below MIN_GROUP_SIZE
    audit = audit_group(df, "home_ownership", cutoff_pd=0.354)
    assert len(audit) == 0
    assert MIN_GROUP_SIZE > 100  # sanity
