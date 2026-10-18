"""
src/train.py
Trains baseline Logistic Regression and LightGBM models, applies calibration,
and evaluates performance metrics (AUC, KS, Brier, Decile tables).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import lightgbm as lgb
from split import load_and_split_data

DB_URI = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/lending"
)


def run_training():
    print("Loading data from database...")
    train, valid, test = load_and_split_data(DB_URI)

    # Assert leakage columns are absent
    forbidden_cols = ["total_pymnt",
                      "recoveries_actual", "out_prncp", "last_pymnt_d"]
    for col in forbidden_cols:
        assert col not in train.columns, f"Leakage column found: {col}"

    features_to_drop = ["id", "issue_d", "defaulted", "recoveries"]
    num_cols = [
        c
        for c in train.select_dtypes(include=[np.number]).columns
        if c not in features_to_drop
    ]
    cat_cols = [
        c
        for c in train.select_dtypes(include=[object, "category"]).columns
        if c not in features_to_drop
    ]

    X_train, y_train = train.drop(columns=features_to_drop), train["defaulted"]
    X_valid, y_valid = valid.drop(columns=features_to_drop), valid["defaulted"]
    X_test, y_test = test.drop(columns=features_to_drop), test["defaulted"]

    # Preprocessor
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=100),
                cat_cols,
            ),
        ]
    )

    print("Training Logistic Regression baseline...")
    logit_pipeline = Pipeline(
        [("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))]
    )
    logit_pipeline.fit(X_train, y_train)
    p_logit = logit_pipeline.predict_proba(X_test)[:, 1]

    print("Training LightGBM model...")
    # For LightGBM, use a simpler pipeline or transform data first
    X_train_trans = preprocessor.fit_transform(X_train)
    X_valid_trans = preprocessor.transform(X_valid)
    X_test_trans = preprocessor.transform(X_test)

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1
    )
    lgb_clf.fit(
        X_train_trans,
        y_train,
        eval_set=[(X_valid_trans, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    p_lgb = lgb_clf.predict_proba(X_test_trans)[:, 1]

    print("Calibrating LightGBM probabilities...")
    calibrated_lgb = CalibratedClassifierCV(
        estimator=lgb_clf, method="sigmoid", cv=3)
    calibrated_lgb.fit(X_valid_trans, y_valid)
    p_cal_lgb = calibrated_lgb.predict_proba(X_test_trans)[:, 1]

    # Evaluation metrics
    def evaluate_model(y_true, p_pred, name):
        auc = roc_auc_score(y_true, p_pred)
        brier = brier_score_loss(y_true, p_pred)
        fpr, tpr, _ = roc_curve(y_true, p_pred)
        ks = (tpr - fpr).max()
        print(f"--- {name} ---")
        print(f"AUC   : {auc:.4f}")
        print(f"KS    : {ks:.4f}")
        print(f"Brier : {brier:.4f}")
        return {"Model": name, "AUC": auc, "KS": ks, "Brier": brier}

    results = [
        evaluate_model(y_test, p_logit, "Logistic Regression"),
        evaluate_model(y_test, p_lgb, "LightGBM (Uncalibrated)"),
        evaluate_model(y_test, p_cal_lgb, "LightGBM (Calibrated)"),
    ]

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(results).to_csv("reports/metrics.csv", index=False)

    # Save predictions for cutoff decision layer
    test_out = test[
        ["id", "loan_amnt", "funded_amnt", "int_rate", "term_months", "defaulted"]
    ].copy()
    test_out["pd_hat"] = p_cal_lgb
    test_out.to_parquet("reports/test_predictions.parquet")
    print("Training complete. Results and predictions saved.")


if __name__ == "__main__":
    run_training()
