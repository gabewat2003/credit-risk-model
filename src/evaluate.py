"""Load test predictions and produce evaluation artifacts.

Writes:

- ``reports/evaluation.md`` — AUC/KS/Brier per model plus the chosen cutoff.
- ``reports/decile_table.csv`` — risk-decile summary for the calibrated model.
- ``reports/calibration.png`` — reliability diagram.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def compute_ks_statistic(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Kolmogorov–Smirnov separation = max(TPR − FPR) across thresholds."""
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    return float((tpr - fpr).max())


def generate_decile_table(y_true: pd.Series, p_pred: np.ndarray, df_source: pd.DataFrame) -> pd.DataFrame:
    """Risk-decile summary: ordered 1 (lowest PD) to 10 (highest).

    A well-ranked model shows a monotonic rise in ``actual_default_rate``
    across deciles; the span between decile 1 and 10 is a lift measure.
    """
    out = df_source[[]].copy()
    out["defaulted"] = y_true.to_numpy()
    out["pd_hat"] = p_pred
    out["decile"] = pd.qcut(out["pd_hat"], 10, labels=False, duplicates="drop") + 1
    return (
        out.groupby("decile")
        .agg(
            min_pd=("pd_hat", "min"),
            max_pd=("pd_hat", "max"),
            mean_pd=("pd_hat", "mean"),
            actual_default_rate=("defaulted", "mean"),
            loan_count=("defaulted", "count"),
        )
        .reset_index()
    )


def _plot_calibration(y_true: np.ndarray, p_pred: np.ndarray, out_path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, p_pred, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color="teal", label="LightGBM (isotonic)")
    ax.set_xlabel("Predicted PD (bin mean)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration — LightGBM (isotonic)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_evaluation() -> None:
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} missing — run `make train` first.")

    df = pd.read_parquet(preds_path)
    y = df["defaulted"].to_numpy()

    lines = ["# Evaluation", ""]
    lines.append("| model | AUC | KS | Brier |")
    lines.append("| --- | ---: | ---: | ---: |")
    for col, label in [
        ("pd_logreg", "Logistic Regression"),
        ("pd_lgbm", "LightGBM (uncalibrated)"),
        ("pd_hat", "LightGBM (isotonic)"),
    ]:
        p = df[col].to_numpy()
        auc = roc_auc_score(y, p)
        ks = compute_ks_statistic(y, p)
        brier = brier_score_loss(y, p)
        lines.append(f"| {label} | {auc:.4f} | {ks:.4f} | {brier:.4f} |")
        logger.info("%s | AUC %.4f | KS %.4f | Brier %.4f", label, auc, ks, brier)

    decile = generate_decile_table(df["defaulted"], df["pd_hat"].to_numpy(), df)
    decile_path = REPORTS_DIR / "decile_table.csv"
    decile.to_csv(decile_path, index=False)
    logger.info("Wrote %s", decile_path)

    lift = decile["actual_default_rate"].iloc[-1] / decile["actual_default_rate"].iloc[0]
    lines += ["", f"Decile lift (D10 / D1 default rate): **{lift:.1f}x**", ""]

    cal_path = REPORTS_DIR / "calibration.png"
    _plot_calibration(y, df["pd_hat"].to_numpy(), cal_path)
    logger.info("Wrote %s", cal_path)
    lines += ["![calibration](calibration.png)", ""]

    eval_md = REPORTS_DIR / "evaluation.md"
    eval_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", eval_md)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_evaluation()


if __name__ == "__main__":
    main()
