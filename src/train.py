"""Train Logistic Regression + LightGBM, calibrate LGBM, write metrics and predictions.

Pipeline outline:

1. Load the time-split frames from the ``features`` table.
2. Assert no leakage column survived into the feature frame.
3. Fit a logistic-regression baseline on numeric + one-hot categorical features.
4. Fit LightGBM on the same preprocessed matrix with early stopping on validation.
5. Isotonic-calibrate the LightGBM probabilities on the validation set
   (``FrozenEstimator`` keeps the base model untouched).
6. Score the test set, write ``reports/metrics.csv``,
   ``reports/test_predictions.parquet``, and ``reports/model.joblib``.
"""

from __future__ import annotations

import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATABASE_URL,
    LEAKAGE_COLS,
    LGBM_EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    LOGREG_MAX_ITER,
    NON_FEATURE_COLS,
    REPORTS_DIR,
)
from src.split import load_and_split_data

logger = logging.getLogger(__name__)


def _assert_no_leakage(feature_cols: list[str]) -> None:
    leaked = sorted(set(feature_cols) & set(LEAKAGE_COLS))
    assert not leaked, f"Leakage columns present in feature set: {leaked}"


def _ks_statistic(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    return float((tpr - fpr).max())


def _score(y_true: np.ndarray, p_pred: np.ndarray, name: str) -> dict:
    auc = roc_auc_score(y_true, p_pred)
    ks = _ks_statistic(y_true, p_pred)
    brier = brier_score_loss(y_true, p_pred)
    logger.info("%s | AUC %.4f | KS %.4f | Brier %.4f", name, auc, ks, brier)
    return {"model": name, "auc": float(auc), "ks": ks, "brier": float(brier)}


def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
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


def run_training() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from database...")
    train, valid, test = load_and_split_data(DATABASE_URL)
    logger.info("Rows — train %d | valid %d | test %d", len(train), len(valid), len(test))

    drop_cols = list(NON_FEATURE_COLS)
    num_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns if c not in drop_cols
    ]
    cat_cols = [
        c
        for c in train.select_dtypes(exclude=[np.number]).columns
        if c not in drop_cols and not pd.api.types.is_datetime64_any_dtype(train[c])
    ]
    feature_cols = num_cols + cat_cols
    _assert_no_leakage(feature_cols)
    logger.info("Features: %d numeric, %d categorical", len(num_cols), len(cat_cols))

    y_train, y_valid, y_test = train["defaulted"], valid["defaulted"], test["defaulted"]
    X_train = train[feature_cols]
    X_valid = valid[feature_cols]
    X_test = test[feature_cols]

    logger.info("Training Logistic Regression baseline...")
    logreg = Pipeline(
        [
            ("pre", _build_preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER)),
        ]
    )
    logreg.fit(X_train, y_train)
    p_logreg_test = logreg.predict_proba(X_test)[:, 1]

    logger.info("Training LightGBM...")
    lgbm_preprocessor = _build_preprocessor(num_cols, cat_cols)
    X_train_t = lgbm_preprocessor.fit_transform(X_train)
    X_valid_t = lgbm_preprocessor.transform(X_valid)
    X_test_t = lgbm_preprocessor.transform(X_test)

    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    # Using legacy eval_set signature; eval_X/eval_y in lightgbm 4.7 doesn't
    # accept sparse matrices from ColumnTransformer. Deprecation warning only.
    lgbm.fit(
        X_train_t,
        y_train,
        eval_set=[(X_valid_t, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    p_lgbm_test = lgbm.predict_proba(X_test_t)[:, 1]

    logger.info("Isotonic-calibrating LightGBM on validation set...")
    calibrated = CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic")
    calibrated.fit(X_valid_t, y_valid)
    p_cal_test = calibrated.predict_proba(X_test_t)[:, 1]

    metrics = pd.DataFrame(
        [
            _score(y_test, p_logreg_test, "logreg"),
            _score(y_test, p_lgbm_test, "lgbm_uncalibrated"),
            _score(y_test, p_cal_test, "lgbm_isotonic"),
        ]
    )
    metrics_path = REPORTS_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("Wrote %s", metrics_path)

    # Predictions used by the cutoff decision layer. int_rate and term_months
    # are the only non-feature loan attributes needed to compute revenue.
    carry_cols = [
        "id",
        "loan_amnt",
        "funded_amnt",
        "int_rate",
        "installment",
        "term_months",
        "defaulted",
        "addr_state",
        "home_ownership",
        "purpose",
    ]
    predictions = test[carry_cols].copy()
    predictions["pd_logreg"] = p_logreg_test
    predictions["pd_lgbm"] = p_lgbm_test
    predictions["pd_hat"] = p_cal_test  # the production score
    preds_path = REPORTS_DIR / "test_predictions.parquet"
    predictions.to_parquet(preds_path)
    logger.info("Wrote %s", preds_path)

    model_path = REPORTS_DIR / "model.joblib"
    joblib.dump(
        {
            "logreg": logreg,
            "lgbm_preprocessor": lgbm_preprocessor,
            "lgbm": lgbm,
            "lgbm_calibrated": calibrated,
            "feature_cols": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        },
        model_path,
    )
    logger.info("Wrote %s", model_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_training()


if __name__ == "__main__":
    main()
