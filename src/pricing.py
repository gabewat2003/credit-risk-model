"""Risk-based pricing: compute the break-even APR per PD decile and compare to
the rate LendingClub actually charged.

The decision layer in ``cutoff.py`` answers *who to approve*. This module
answers *what rate to charge them*. A real consumer lender uses PD estimates
primarily to price risk, not to gate approval — low-decile borrowers get the
headline rate, high-decile borrowers get a markup that covers their expected
loss, and only genuinely unprofitable borrowers are declined.

Break-even derivation (fully amortizing loan, flat LGD):

    E[profit per loan] = (1 - PD) · total_interest − PD · LGD · L  = 0
    total_interest*    = PD · LGD · L / (1 - PD)
    installment*       = (L + total_interest*) / term_months

Standard amortization gives the APR r that produces that installment:

    installment = L · (r/12) / (1 − (1 + r/12)^(-term_months))

We solve for r numerically with Brent's method for each decile.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.config import LGD, REPORTS_DIR

logger = logging.getLogger(__name__)

# Add this much APR on top of break-even as the lender's margin + funding cost.
TARGET_MARGIN_PCT: float = 3.0

# Floor and ceiling for the solver — units are APR percent (0.1% to 100%).
APR_SEARCH_LOW_PCT: float = 0.1
APR_SEARCH_HIGH_PCT: float = 100.0


def amortized_installment(loan_amnt: float, apr_pct: float, term_months: int) -> float:
    """Standard amortizing monthly payment at the given annual APR (percent)."""
    r = apr_pct / 100.0 / 12.0
    if r == 0:
        return loan_amnt / term_months
    return loan_amnt * r / (1 - (1 + r) ** -term_months)


def break_even_apr(pd_rate: float, loan_amnt: float, term_months: int, lgd: float = LGD) -> float:
    """APR (percent) at which E[profit per loan] = 0 for a borrower with this PD."""
    if pd_rate <= 0:
        return 0.0
    if pd_rate >= 1:
        return float("inf")

    target_installment = (loan_amnt + pd_rate * lgd * loan_amnt / (1 - pd_rate)) / term_months

    def residual(apr_pct: float) -> float:
        return amortized_installment(loan_amnt, apr_pct, term_months) - target_installment

    # Guard: if target_installment is below the near-zero-APR payment, no positive
    # APR solves it (borrower riskless enough that 0% APR already covers losses).
    if residual(APR_SEARCH_LOW_PCT) >= 0:
        return 0.0
    if residual(APR_SEARCH_HIGH_PCT) <= 0:
        return APR_SEARCH_HIGH_PCT
    return brentq(residual, APR_SEARCH_LOW_PCT, APR_SEARCH_HIGH_PCT)


def build_pricing_table(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """One row per PD decile with actual rate charged vs. break-even and proposed.

    ``df`` must carry ``pd_hat``, ``int_rate`` (percent), ``loan_amnt``,
    ``term_months``, and ``defaulted`` columns from the trained-model output.
    """
    out = df.copy()
    out["decile"] = pd.qcut(out["pd_hat"], n_deciles, labels=False, duplicates="drop") + 1

    rows = []
    for d, group in out.groupby("decile"):
        pd_rate = float(group["defaulted"].mean())
        mean_loan = float(group["loan_amnt"].mean())
        mean_term = int(round(group["term_months"].mean()))
        be_apr = break_even_apr(pd_rate, mean_loan, mean_term)
        rows.append(
            {
                "decile": int(d),
                "loan_count": int(len(group)),
                "mean_pd_hat": float(group["pd_hat"].mean()),
                "actual_default_rate": pd_rate,
                "mean_loan_amnt": mean_loan,
                "mean_term_months": mean_term,
                "actual_apr": float(group["int_rate"].mean()),
                "break_even_apr": be_apr,
                "proposed_apr": be_apr + TARGET_MARGIN_PCT,
            }
        )
    return pd.DataFrame(rows)


def apply_pricing_policy(df: pd.DataFrame, pricing: pd.DataFrame) -> pd.DataFrame:
    """Repay each test loan at the proposed APR for its decile, keeping actual outcome."""
    joined = df.merge(pricing[["decile", "proposed_apr"]], on="decile", how="left")
    joined["proposed_installment"] = [
        amortized_installment(L, apr, T)
        for L, apr, T in zip(
            joined["loan_amnt"], joined["proposed_apr"], joined["term_months"], strict=True
        )
    ]
    joined["proposed_interest"] = (
        joined["proposed_installment"] * joined["term_months"] - joined["loan_amnt"]
    )
    joined["proposed_realized"] = np.where(
        joined["defaulted"] == 0,
        joined["proposed_interest"],
        -LGD * joined["loan_amnt"],
    )
    joined["actual_interest"] = joined["installment"] * joined["term_months"] - joined["loan_amnt"]
    joined["actual_realized"] = np.where(
        joined["defaulted"] == 0,
        joined["actual_interest"],
        -LGD * joined["loan_amnt"],
    )
    return joined


def _plot_pricing(pricing: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = pricing["decile"]
    ax.plot(x, pricing["actual_apr"], "o-", color="gray", label="LendingClub actual APR")
    ax.plot(x, pricing["break_even_apr"], "s--", color="crimson", label="Model break-even APR")
    ax.plot(
        x,
        pricing["proposed_apr"],
        "D-",
        color="teal",
        label=f"Proposed APR (break-even + {TARGET_MARGIN_PCT}%)",
    )
    ax.set_xlabel("PD decile (1 = safest, 10 = riskiest)")
    ax.set_ylabel("APR (%)")
    ax.set_title("Risk-based pricing: break-even vs charged APR by PD decile")
    ax.set_xticks(range(1, len(pricing) + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pricing() -> None:
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} missing — run `make train` first.")

    df = pd.read_parquet(preds_path)
    pricing = build_pricing_table(df)
    pricing_path = REPORTS_DIR / "pricing_table.csv"
    pricing.to_csv(pricing_path, index=False)
    logger.info("Wrote %s", pricing_path)

    for _, row in pricing.iterrows():
        logger.info(
            "D%02d | PD %.1f%% | actual %.1f%% | break-even %.1f%% | proposed %.1f%%",
            row["decile"],
            row["actual_default_rate"] * 100,
            row["actual_apr"],
            row["break_even_apr"],
            row["proposed_apr"],
        )

    # Simulate repricing on the test set to show the headline portfolio number.
    df_dec = df.copy()
    df_dec["decile"] = pd.qcut(df_dec["pd_hat"], 10, labels=False, duplicates="drop") + 1
    policy = apply_pricing_policy(df_dec, pricing)
    total_actual = policy["actual_realized"].sum() / 1e6
    total_proposed = policy["proposed_realized"].sum() / 1e6
    delta = total_proposed - total_actual
    logger.info("Portfolio realized at LendingClub actual APR: $%.2fM", total_actual)
    logger.info("Portfolio realized at proposed APR:            $%.2fM", total_proposed)
    logger.info("Pricing policy lift:                           $%.2fM (%+.1f%%)", delta, 100 * delta / total_actual)

    chart_path = REPORTS_DIR / "pricing.png"
    _plot_pricing(pricing, chart_path)
    logger.info("Wrote %s", chart_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pricing()


if __name__ == "__main__":
    main()
