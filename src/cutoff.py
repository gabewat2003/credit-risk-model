"""
src/cutoff.py
Implements the decision layer: computes expected profit per loan across probability cutoffs,
evaluates optimal thresholds, and generates summary visualizations and outputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_decision_layer():
    print("Loading test predictions...")
    if not os.path.exists("reports/test_predictions.parquet"):
        print("Predictions not found. Run train.py first.")
        return

    test = pd.read_parquet("reports/test_predictions.parquet")

    # Financial assumptions
    # E[profit] = (1 - PD) * interest_earned - PD * (LGD * principal)
    # interest_earned approx loan_amnt * (int_rate / 100) * (term_months / 12)
    lgd = 0.85
    test["interest_earned"] = (
        test["loan_amnt"] * (test["int_rate"] / 100.0) *
        (test["term_months"] / 12.0)
    )
    test["exp_profit"] = (1.0 - test["pd_hat"]) * test["interest_earned"] - test[
        "pd_hat"
    ] * (lgd * test["loan_amnt"])
    test["realized_profit"] = np.where(
        test["defaulted"] == 0, test["interest_earned"], -
        lgd * test["loan_amnt"]
    )

    rows = []
    thresholds = np.linspace(0.02, 1.00, 100)
    for t in thresholds:
        approved = test[test["pd_hat"] <= t]
        if len(approved) == 0:
            continue
        rows.append(
            {
                "cutoff": t,
                "approval_rate": len(approved) / len(test),
                "realized_default_rate": approved["defaulted"].mean(),
                "expected_profit": approved["exp_profit"].sum(),
                "realized_profit": approved["realized_profit"].sum(),
            }
        )

    df_sweep = pd.DataFrame(rows)
    optimal_row = df_sweep.loc[df_sweep["expected_profit"].idxmax()]

    print("\n--- Decision Layer Results ---")
    print(f"Optimal Cutoff PD       : {optimal_row['cutoff']:.2%}")
    print(f"Approval Rate           : {optimal_row['approval_rate']:.2%}")
    print(f"Realized Default Rate   : {
          optimal_row['realized_default_rate']:.2%}")
    print(f"Total Expected Profit   : ${optimal_row['expected_profit']:,.2f}")
    print(f"Total Realized Profit   : ${optimal_row['realized_profit']:,.2f}")

    # Plotting approval rate vs expected profit
    plt.figure(figsize=(9, 5))
    plt.plot(
        df_sweep["approval_rate"] * 100,
        df_sweep["expected_profit"] / 1e6,
        label="Expected Profit",
        color="teal",
        lw=2,
    )
    plt.axvline(
        optimal_row["approval_rate"] * 100,
        color="crimson",
        linestyle="--",
        label=f"Optimal Cutoff ({optimal_row['cutoff']:.1%})",
    )
    plt.title("Approval Rate vs. Expected Portfolio Profit")
    plt.xlabel("Approval Rate (%)")
    plt.ylabel("Total Expected Profit ($ Millions)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/approval_vs_profit.png", dpi=300)
    print("\nSaved chart to reports/approval_vs_profit.png")


if __name__ == "__main__":
    run_decision_layer()
