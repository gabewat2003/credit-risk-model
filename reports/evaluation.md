# Evaluation

| model | AUC | KS | Brier |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.6650 | 0.2407 | 0.1674 |
| LightGBM (uncalibrated) | 0.6703 | 0.2456 | 0.1651 |
| LightGBM (isotonic) | 0.6702 | 0.2448 | 0.1641 |

Decile lift (D10 / D1 default rate): **6.3x**

![calibration](calibration.png)
