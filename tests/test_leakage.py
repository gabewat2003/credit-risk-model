"""Guard the leakage column list against regressions."""

from __future__ import annotations

import pytest

from src.config import LEAKAGE_COLS
from src.train import _assert_no_leakage


def test_assert_no_leakage_passes_on_clean_feature_list():
    _assert_no_leakage(["loan_amnt", "dti", "credit_history_years"])


@pytest.mark.parametrize("col", LEAKAGE_COLS)
def test_assert_no_leakage_catches_each_forbidden_col(col):
    with pytest.raises(AssertionError):
        _assert_no_leakage(["loan_amnt", col])


def test_critical_leakage_cols_listed():
    # Changes to this list should be deliberate — if you need to add one,
    # update this test too so a reviewer sees the intent.
    must_include = {
        "recoveries",
        "total_rec_prncp",
        "total_rec_int",
        "last_pymnt_amnt",
        "last_pymnt_d",
        "collection_recovery_fee",
    }
    assert must_include.issubset(set(LEAKAGE_COLS))
