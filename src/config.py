"""Project-wide constants. Pulled out so magic numbers don't scatter across scripts."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw" / "loan.csv"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
SQL_DIR: Path = PROJECT_ROOT / "sql"

# --- Database ---
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/lending",
)

# --- Load ---
RAW_TABLE: str = "loans_raw"

# --- Time split (train < 2015-01-01 <= valid < 2016-07-01 <= test) ---
TRAIN_END: str = "2015-01-01"
VALID_END: str = "2016-07-01"

# --- Business ---
LGD: float = 0.85
CUTOFF_GRID_MIN: float = 0.02
CUTOFF_GRID_MAX: float = 0.60
CUTOFF_GRID_STEPS: int = 100

# --- Models ---
RANDOM_SEED: int = 42
LGBM_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}
LGBM_EARLY_STOPPING_ROUNDS: int = 50
LOGREG_MAX_ITER: int = 1000
# n_jobs removed from LogisticRegression — deprecated in sklearn 1.8.

# --- Leakage: columns derived from loan outcome, must never enter features ---
LEAKAGE_COLS: tuple[str, ...] = (
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "last_pymnt_d",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "settlement_status",
    "settlement_amount",
    "debt_settlement_flag",
)

# --- Non-feature columns (identifiers, target, or dropped by design) ---
# grade/sub_grade/int_rate dropped: README's "build your own score" framing —
# otherwise LightGBM essentially re-learns LendingClub's own score.
NON_FEATURE_COLS: tuple[str, ...] = (
    "id",
    "issue_d",
    "defaulted",
    "recoveries",
    "grade",
    "sub_grade",
    "int_rate",
)
