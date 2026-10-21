"""Demand elasticity: realistic revenue uplift under rate-sensitive borrower churn.

The $264M uplift reported by ``src/pricing.py`` holds applicant behavior
constant under the proposed APR. That is an upper bound — in reality, some
borrowers walk when offered a higher rate. This module applies a
constant-elasticity demand model to the pricing table and reports the uplift
across the published range of consumer-credit elasticity estimates.

Model (log-linear / constant elasticity):

    volume_mult(d, β) = min(1.0, (proposed_apr_d / actual_apr_d) ** β)
    new_count(d, β)   = original_count_d · volume_mult(d, β)
    revenue(d, β)     = new_count(d, β) · E[profit per loan at proposed_apr_d]

The cap at 1.0 enforces that lowering a rate can never add borrowers we did
not observe — we only lose borrowers who walk when the rate rises. Default
rate per decile is held constant (no adverse-selection modelling).

There is no clean published β for US unsecured consumer-instalment lending
that transfers directly to this population — the closest field evidence
(Karlan-Zinman 2008, AER 98(3), on a South African consumer lender)
reports steep demand responses without a point estimate the reader can
port over, and most other cited work (e.g. DeFusco-Paciorek 2017) studies
mortgages. β = -4 is used as an explicit midpoint assumption, with the
grid β ∈ [-2, -8] bracketing the plausible range (prime/inelastic to
deep-subprime/elastic).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import LGD, REPORTS_DIR
from src.pricing import TARGET_MARGIN_PCT, amortized_installment, build_pricing_table

logger = logging.getLogger(__name__)

ELASTICITY_GRID: tuple[float, ...] = (-2.0, -3.0, -4.0, -5.0, -6.0, -8.0)
HEADLINE_BETA: float = -4.0


def volume_multiplier(proposed_apr: float, actual_apr: float, beta: float) -> float:
    """Fraction of the original decile population that still accepts the proposed APR.

    Capped at 1.0 — borrowers in the dataset already accepted at the actual
    rate, so a lower proposed rate cannot grow the observed pool.
    """
    if actual_apr <= 0 or proposed_apr <= 0:
        return 1.0
    return min(1.0, (proposed_apr / actual_apr) ** beta)


def _mean_profit_per_loan_at_apr(
    mean_loan: float,
    mean_term: int,
    default_rate: float,
    apr_pct: float,
    lgd: float = LGD,
) -> float:
    """Expected realized profit per approved loan at the given APR."""
    installment = amortized_installment(mean_loan, apr_pct, mean_term)
    interest = installment * mean_term - mean_loan
    return (1.0 - default_rate) * interest - default_rate * lgd * mean_loan


def _row_level_realized(preds: pd.DataFrame, apr_col: str, lgd: float = LGD) -> pd.Series:
    """Per-row realized profit using actual loan outcomes and the APR in ``apr_col``."""
    installment = [
        amortized_installment(L, apr, T)
        for L, apr, T in zip(preds["loan_amnt"], preds[apr_col], preds["term_months"], strict=True)
    ]
    installment = np.asarray(installment)
    interest = installment * preds["term_months"].to_numpy() - preds["loan_amnt"].to_numpy()
    paid = preds["defaulted"].to_numpy() == 0
    return pd.Series(
        np.where(paid, interest, -lgd * preds["loan_amnt"].to_numpy()),
        index=preds.index,
    )


def apply_elasticity(pricing: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Decile-level volume multiplier + retained-count summary."""
    out = pricing.copy()
    out["volume_mult"] = [
        volume_multiplier(p, a, beta)
        for p, a in zip(out["proposed_apr"], out["actual_apr"], strict=True)
    ]
    out["elastic_count"] = out["loan_count"] * out["volume_mult"]
    out["profit_per_loan_proposed"] = [
        _mean_profit_per_loan_at_apr(L, T, pd_rate, apr)
        for L, T, pd_rate, apr in zip(
            out["mean_loan_amnt"],
            out["mean_term_months"],
            out["actual_default_rate"],
            out["proposed_apr"],
            strict=True,
        )
    ]
    return out


def sensitivity_table(
    preds: pd.DataFrame,
    pricing: pd.DataFrame,
    betas: tuple[float, ...] = ELASTICITY_GRID,
) -> pd.DataFrame:
    """Row-level portfolio profit under each β, matching cutoff.py's baseline conventions."""
    # Attach proposed APR per row via the decile mapping.
    df = preds.merge(pricing[["decile", "proposed_apr", "actual_apr"]], on="decile", how="left")
    # Each row's elasticity-driven retention probability.
    df["volume_mult"] = [
        volume_multiplier(p, a, 0.0)  # placeholder, replaced per β below
        for p, a in zip(df["proposed_apr"], df["actual_apr"], strict=True)
    ]
    actual_realized = _row_level_realized(df, apr_col="int_rate")
    proposed_realized = _row_level_realized(df, apr_col="proposed_apr")
    baseline_total = float(actual_realized.sum())
    no_churn_total = float(proposed_realized.sum())

    rows = [
        {
            "beta": 0.0,
            "retained_volume_pct": 100.0,
            "elastic_profit_M": no_churn_total / 1e6,
            "uplift_vs_actual_M": (no_churn_total - baseline_total) / 1e6,
            "note": "upper bound (no churn)",
        }
    ]
    for beta in betas:
        mult = np.array(
            [volume_multiplier(p, a, beta) for p, a in zip(df["proposed_apr"], df["actual_apr"], strict=True)]
        )
        elastic_total = float((proposed_realized.to_numpy() * mult).sum())
        retained = float(mult.mean() * 100)
        rows.append(
            {
                "beta": beta,
                "retained_volume_pct": retained,
                "elastic_profit_M": elastic_total / 1e6,
                "uplift_vs_actual_M": (elastic_total - baseline_total) / 1e6,
                "note": "headline" if beta == HEADLINE_BETA else "",
            }
        )
    rows.append(
        {
            "beta": float("-inf"),
            "retained_volume_pct": 0.0,
            "elastic_profit_M": 0.0,
            "uplift_vs_actual_M": -baseline_total / 1e6,
            "note": "nobody accepts",
        }
    )
    return pd.DataFrame(rows)


def _plot_sensitivity(summary: pd.DataFrame, baseline_M: float, no_churn_M: float, out_path) -> None:
    finite = summary[np.isfinite(summary["beta"])].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(finite["beta"], finite["elastic_profit_M"], "o-", color="teal", lw=2,
            label="Elasticity-adjusted portfolio profit")
    ax.axhline(baseline_M, linestyle="--", color="gray",
               label=f"LendingClub actual APRs (${baseline_M:.1f}M)")
    ax.axhline(no_churn_M, linestyle=":", color="crimson",
               label=f"Proposed APRs, no churn (${no_churn_M:.1f}M)")
    headline = finite[finite["beta"] == HEADLINE_BETA]
    if not headline.empty:
        hp = headline["elastic_profit_M"].iloc[0]
        ax.annotate(f"β = {HEADLINE_BETA}\n${hp:.1f}M",
                    xy=(HEADLINE_BETA, hp), xytext=(HEADLINE_BETA + 0.3, hp + 15),
                    fontsize=10, color="teal",
                    arrowprops=dict(arrowstyle="->", color="teal", lw=1))
    ax.set_xlabel("Price elasticity of demand (β)")
    ax.set_ylabel("Realized portfolio profit ($M)")
    ax.set_title(f"Pricing uplift under demand elasticity (margin = {TARGET_MARGIN_PCT}% over break-even)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.invert_xaxis()  # more-elastic (more negative β) on the right
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_elasticity() -> None:
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} missing — run `make train` first.")

    df = pd.read_parquet(preds_path)
    df["decile"] = pd.qcut(df["pd_hat"], 10, labels=False, duplicates="drop") + 1
    pricing = build_pricing_table(df)

    summary = sensitivity_table(df, pricing)
    out_csv = REPORTS_DIR / "elasticity.csv"
    summary.to_csv(out_csv, index=False)
    logger.info("Elasticity sensitivity:\n%s", summary.to_string(index=False))
    logger.info("Wrote %s", out_csv)

    zero_row = summary.loc[summary["beta"] == 0.0].iloc[0]
    no_churn_M = float(zero_row["elastic_profit_M"])
    baseline_M = no_churn_M - float(zero_row["uplift_vs_actual_M"])
    chart = REPORTS_DIR / "elasticity.png"
    _plot_sensitivity(summary, baseline_M, no_churn_M, chart)
    logger.info("Wrote %s", chart)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_elasticity()


if __name__ == "__main__":
    main()
