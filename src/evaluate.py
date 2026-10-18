"""
src/evaluate.py
Computes credit risk evaluation metrics including AUC, KS statistic, Brier score,
and generates decile calibration tables.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve


def compute_ks_statistic(y_true, p_pred):
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    return (tpr - fpr).max()


def generate_decile_table(y_true, p_pred, df_source):
    """
    Generates a decile summary table to check monotonicity of default rates across score deciles.
    """
    eval_df = df_source.copy()
    eval_df["defaulted"] = y_true.values
    eval_df["pd_hat"] = p_pred

    # Create 10 risk deciles (1 = lowest risk, 10 = highest risk)
    eval_df["decile"] = pd.qcut(
        eval_df["pd_hat"], 10, labels=False, duplicates="drop")
    eval_df["decile"] = eval_df["decile"] + 1

    decile_summary = (
        eval_df.groupby("decile")
        .agg(
            min_pd=("pd_hat", "min"),
            max_pd=("pd_hat", "max"),
            mean_pd=("pd_hat", "mean"),
            actual_default_rate=("defaulted", "mean"),
            loan_count=("defaulted", "count"),
        )
        .reset_index()
    )

    return decile_summary


if __name__ == "__main__":
    print(
        "Evaluation module loaded successfully. Import compute_ks_statistic or generate_decile_table as needed."
    )
