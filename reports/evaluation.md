# Evaluation

| model | AUC | KS | Brier |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.6649 | 0.2405 | 0.1673 |
| LightGBM (uncalibrated) | 0.6703 | 0.2464 | 0.1653 |
| LightGBM (isotonic) | 0.6703 | 0.2462 | 0.1641 |

Decile lift (D10 / D1 default rate): **6.4x**

![calibration](calibration.png)
