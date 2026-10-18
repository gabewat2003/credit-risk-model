"""Decision layer: sweep probability cutoffs, pick the one that maximises expected profit.

Profit per loan (fully amortizing):

    E[profit] = (1 - PD) * interest_earned - PD * (LGD * loan_amnt)
    interest_earned = installment * term_months - loan_amnt

Using the real amortization schedule (via the servicer-published ``installment``
column) rather than a simple ``loan_amnt * rate * years`` approximation —
the simple formula roughly doubles total interest because principal declines
each month on an amortizing loan.

The chart overlays an "approve-everyone" baseline so the lift from the model is visible.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CUTOFF_GRID_MAX, CUTOFF_GRID_MIN, CUTOFF_GRID_STEPS, LGD, REPORTS_DIR

logger = logging.getLogger(__name__)


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Exact scheduled total interest from the amortization schedule.
    df["interest_earned"] = df["installment"] * df["term_months"] - df["loan_amnt"]
    df["exp_profit"] = (
        (1.0 - df["pd_hat"]) * df["interest_earned"]
        - df["pd_hat"] * (LGD * df["loan_amnt"])
    )
    df["realized_profit"] = np.where(
        df["defaulted"] == 0,
        df["interest_earned"],
        -LGD * df["loan_amnt"],
    )
    return df


def sweep_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    """Expected- and realized-profit totals across the configured cutoff grid."""
    rows = []
    thresholds = np.linspace(CUTOFF_GRID_MIN, CUTOFF_GRID_MAX, CUTOFF_GRID_STEPS)
    for t in thresholds:
        approved = df[df["pd_hat"] <= t]
        if approved.empty:
            continue
        rows.append(
            {
                "cutoff": t,
                "approval_rate": len(approved) / len(df),
                "realized_default_rate": approved["defaulted"].mean(),
                "expected_profit": approved["exp_profit"].sum(),
                "realized_profit": approved["realized_profit"].sum(),
            }
        )
    return pd.DataFrame(rows)


def run_decision_layer() -> None:
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} missing — run `make train` first.")

    df = _augment(pd.read_parquet(preds_path))

    sweep = sweep_cutoffs(df)
    optimal = sweep.loc[sweep["expected_profit"].idxmax()]
    approve_all_profit = df["realized_profit"].sum() / 1e6

    logger.info("--- Decision Layer Results ---")
    logger.info("Optimal cutoff PD    : %.2f%%", optimal["cutoff"] * 100)
    logger.info("Approval rate        : %.1f%%", optimal["approval_rate"] * 100)
    logger.info("Realized default rate: %.2f%%", optimal["realized_default_rate"] * 100)
    logger.info("Total expected profit: $%.2fM", optimal["expected_profit"] / 1e6)
    logger.info("Total realized profit: $%.2fM", optimal["realized_profit"] / 1e6)
    logger.info("Approve-all realized : $%.2fM", approve_all_profit)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        sweep["approval_rate"] * 100,
        sweep["expected_profit"] / 1e6,
        label="Expected profit (model cutoff)",
        color="teal",
        lw=2,
    )
    ax.plot(
        sweep["approval_rate"] * 100,
        sweep["realized_profit"] / 1e6,
        label="Realized profit (test set)",
        color="orange",
        lw=1.5,
        alpha=0.8,
    )
    ax.axhline(
        approve_all_profit,
        color="gray",
        linestyle=":",
        label=f"Approve everyone (${approve_all_profit:.1f}M)",
    )
    ax.axvline(
        optimal["approval_rate"] * 100,
        color="crimson",
        linestyle="--",
        label=f"Optimal cutoff ({optimal['cutoff']:.1%} PD)",
    )
    ax.set_title("Approval rate vs. portfolio profit")
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Total profit ($ millions)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = REPORTS_DIR / "cutoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)

    sweep_path = REPORTS_DIR / "cutoff_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    logger.info("Wrote %s", sweep_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_decision_layer()


if __name__ == "__main__":
    main()
