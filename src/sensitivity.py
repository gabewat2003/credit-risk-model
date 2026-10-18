"""LGD sensitivity: how the approve/decline cutoff and break-even APR curve
shift under different loss-given-default assumptions.

LGD is the single biggest unknown in the profit math — the LendingClub
``recoveries`` column suggests a realized LGD closer to 0.95, academic
work on unsecured consumer credit uses 0.70–0.90, and 1.0 is the
worst-case assumption. Rerunning the decision layer at three LGDs makes
the sensitivity explicit instead of hiding it behind a point estimate.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CUTOFF_GRID_MAX, CUTOFF_GRID_MIN, CUTOFF_GRID_STEPS, REPORTS_DIR
from src.pricing import TARGET_MARGIN_PCT, break_even_apr

logger = logging.getLogger(__name__)

LGD_GRID: tuple[float, ...] = (0.70, 0.85, 1.00)


def _cutoff_sweep_at_lgd(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    interest = df["installment"] * df["term_months"] - df["loan_amnt"]
    exp_profit = (1.0 - df["pd_hat"]) * interest - df["pd_hat"] * (lgd * df["loan_amnt"])
    realized = np.where(df["defaulted"] == 0, interest, -lgd * df["loan_amnt"])
    rows = []
    for t in np.linspace(CUTOFF_GRID_MIN, CUTOFF_GRID_MAX, CUTOFF_GRID_STEPS):
        mask = df["pd_hat"] <= t
        if not mask.any():
            continue
        rows.append(
            {
                "lgd": lgd,
                "cutoff": t,
                "approval_rate": float(mask.mean()),
                "expected_profit": float(exp_profit[mask].sum()),
                "realized_profit": float(realized[mask].sum()),
            }
        )
    return pd.DataFrame(rows)


def _pricing_at_lgd(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    tmp = df.copy()
    tmp["decile"] = pd.qcut(tmp["pd_hat"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for d, group in tmp.groupby("decile"):
        pd_rate = float(group["defaulted"].mean())
        mean_loan = float(group["loan_amnt"].mean())
        mean_term = int(round(group["term_months"].mean()))
        rows.append(
            {
                "lgd": lgd,
                "decile": int(d),
                "actual_default_rate": pd_rate,
                "break_even_apr": break_even_apr(pd_rate, mean_loan, mean_term, lgd=lgd),
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity() -> None:
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} missing — run `make train` first.")
    df = pd.read_parquet(preds_path)

    cutoff_rows = []
    pricing_rows = []
    for lgd in LGD_GRID:
        sweep = _cutoff_sweep_at_lgd(df, lgd)
        optimal = sweep.loc[sweep["expected_profit"].idxmax()]
        cutoff_rows.append(
            {
                "lgd": lgd,
                "optimal_cutoff_pd": float(optimal["cutoff"]),
                "approval_rate": float(optimal["approval_rate"]),
                "realized_profit_M": float(optimal["realized_profit"]) / 1e6,
            }
        )
        pricing_rows.append(_pricing_at_lgd(df, lgd))

    cutoff_summary = pd.DataFrame(cutoff_rows)
    cutoff_summary.to_csv(REPORTS_DIR / "sensitivity_cutoff.csv", index=False)
    logger.info("LGD sensitivity on approve/decline:\n%s", cutoff_summary.to_string(index=False))

    pricing_wide = pd.concat(pricing_rows, ignore_index=True)
    pricing_wide.to_csv(REPORTS_DIR / "sensitivity_pricing.csv", index=False)

    # Chart: break-even APR curves at each LGD, with the proposed policy curve for 0.85.
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {0.70: "steelblue", 0.85: "teal", 1.00: "crimson"}
    for lgd in LGD_GRID:
        sub = pricing_wide[pricing_wide["lgd"] == lgd]
        ax.plot(
            sub["decile"],
            sub["break_even_apr"],
            "o-",
            color=palette[lgd],
            label=f"Break-even APR @ LGD = {lgd:.2f}",
        )
    ax.axhline(
        TARGET_MARGIN_PCT,
        linestyle=":",
        color="gray",
        alpha=0.5,
        label=f"Target margin ({TARGET_MARGIN_PCT}%)",
    )
    ax.set_xlabel("PD decile")
    ax.set_ylabel("Break-even APR (%)")
    ax.set_title("Break-even APR by decile under alternative LGD assumptions")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = REPORTS_DIR / "sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_sensitivity()


if __name__ == "__main__":
    main()
