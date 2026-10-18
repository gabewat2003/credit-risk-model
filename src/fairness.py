"""Fair-lending audit: approval, risk, and pricing disparities across groups.

Protected characteristics (race, sex, age) are not in the LendingClub data —
nor should they ever enter the model. The audit instead inspects three
proxies routinely tested by compliance teams for disparate impact:

- ``addr_state`` — geographic concentration can proxy for race and income.
- ``home_ownership`` — correlated with wealth and race in US data.
- ``purpose`` — self-reported, but ``small_business`` and ``educational`` loans
  concentrate differently than ``debt_consolidation``.

For each group this reports:

1. Approval rate under the model cutoff (PD ≤ 35.4%).
2. Realized default rate (how the group actually performed).
3. **Disparate-impact ratio** = group approval rate / reference-group approval rate.
   Under the "four-fifths rule" a ratio below 0.80 flags for review.

A disparate-impact ratio below 0.80 is not proof of discrimination — the
population may genuinely be higher risk — but it is the metric a compliance
team starts with and a model owner has to be able to defend.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

CUTOFF_PD: float = 0.354  # the operating cutoff from src/cutoff.py (PD <= 35.4%)
MIN_GROUP_SIZE: int = 500  # ignore sparse groups so ratios are not noise
DISPARATE_IMPACT_FLOOR: float = 0.80  # four-fifths rule


def audit_group(
    df: pd.DataFrame, column: str, cutoff_pd: float = CUTOFF_PD
) -> pd.DataFrame:
    """Group-level approval, risk, and disparate-impact ratios on one column."""
    df = df.copy()
    df["approved"] = (df["pd_hat"] <= cutoff_pd).astype(int)

    grouped = (
        df.groupby(column)
        .agg(
            loan_count=("defaulted", "size"),
            approval_rate=("approved", "mean"),
            default_rate=("defaulted", "mean"),
            mean_pd=("pd_hat", "mean"),
            mean_loan=("loan_amnt", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["loan_count"] >= MIN_GROUP_SIZE].copy()

    # Reference = highest approval rate group (strongest proxy for the "majority" bar).
    ref_rate = grouped["approval_rate"].max()
    grouped["disparate_impact_ratio"] = grouped["approval_rate"] / ref_rate
    grouped["flagged"] = grouped["disparate_impact_ratio"] < DISPARATE_IMPACT_FLOOR
    return grouped.sort_values("disparate_impact_ratio").reset_index(drop=True)


def _plot_disparate_impact(audit: pd.DataFrame, column: str, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(audit))))
    colors = ["crimson" if f else "steelblue" for f in audit["flagged"]]
    ax.barh(audit[column].astype(str), audit["disparate_impact_ratio"], color=colors)
    ax.axvline(DISPARATE_IMPACT_FLOOR, linestyle="--", color="black", alpha=0.5, label="4/5ths floor")
    ax.axvline(1.0, linestyle=":", color="gray", alpha=0.5)
    ax.set_xlabel("Disparate-impact ratio (group approval / reference approval)")
    ax.set_title(f"Disparate-impact audit: approval by {column}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_fairness_audit() -> None:
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} missing — run `make train` first.")

    df = pd.read_parquet(preds_path)
    missing = [c for c in ("addr_state", "home_ownership", "purpose") if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Predictions parquet lacks protected-proxy columns {missing}. "
            "Rerun `make train` after updating src/train.py to carry them through."
        )

    summary_rows = []
    for col in ("home_ownership", "purpose", "addr_state"):
        audit = audit_group(df, col)
        out_csv = REPORTS_DIR / f"fairness_{col}.csv"
        audit.to_csv(out_csv, index=False)
        logger.info("Wrote %s", out_csv)
        flagged = audit[audit["flagged"]]
        logger.info(
            "%s: %d groups >=%d loans, %d flagged under 4/5ths rule",
            col,
            len(audit),
            MIN_GROUP_SIZE,
            len(flagged),
        )
        if len(flagged):
            logger.info("Flagged %s groups:\n%s", col, flagged.to_string(index=False))

        if col in ("home_ownership", "purpose"):
            _plot_disparate_impact(audit, col, REPORTS_DIR / f"fairness_{col}.png")

        summary_rows.append(
            {
                "column": col,
                "groups_audited": len(audit),
                "groups_flagged": int(len(flagged)),
                "min_ratio": float(audit["disparate_impact_ratio"].min()),
                "worst_group": str(audit.iloc[0][col]) if len(audit) else "",
            }
        )

    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "fairness_summary.csv", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_fairness_audit()


if __name__ == "__main__":
    main()
