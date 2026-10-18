# Portfolio Memo — LendingClub PD, Approval Cutoff, Risk-Based Pricing

**Question.** Build a probability-of-default model on LendingClub borrower attributes alone — no grade, no sub-grade, no int_rate — and use it to answer two decisions a lender actually makes: who to approve, and what rate to charge.

**Approach.** 2.26M loans 2007–2018 → filter to resolved outcomes (1.30M rows) → engineer 13 numeric + 3 categorical features in SQL, including a state-level prior default-rate window function → time-based split (train < 2015-01; valid < 2016-07; test ≥ 2016-07) → logistic regression baseline + LightGBM with early stopping → isotonic calibration via `FrozenEstimator` → amortized-interest profit math driving an approve/decline sweep and a per-decile break-even APR solver (Brent's method on the amortization equation).

## Model quality

Calibrated LightGBM on 319k test loans: AUC 0.669, KS 0.244, Brier 0.164, **6.4× decile lift** (D1 default 6.8% vs D10 43.5%). Modest AUC is the price of dropping LendingClub's own grade/int_rate from the feature set; the lift metric is more informative for a rank-ordering decision.

## Decision 1: Approve/decline

| Policy | Approval rate | Default rate | Realized profit |
| --- | ---: | ---: | ---: |
| Approve everyone | 100% | 20.1% | $52.9M |
| **Model cutoff (PD ≤ 35.4%)** | **86.4%** | **19.6%** | **$73.0M (+38%)** |

Declining the riskiest 14% of applicants adds $20M to realized portfolio profit on the test window.

## Decision 2: Risk-based pricing

Per-decile break-even APR vs what LendingClub actually charged:

| Decile | PD | LendingClub APR | Break-even APR |
| ---: | ---: | ---: | ---: |
| 1 | 7.1% | 9.4% | 4.2% |
| 5 | 20.6% | 13.2% | 13.1% |
| 9 | 33.5% | 17.3% | 17.0% |
| 10 | 42.8% | 20.0% | 22.1% |

**Structural finding:** LendingClub's own APR curve tracks default rate roughly 1-for-1, leaving ~zero margin before LGD. Once a flat 85% LGD is applied, their actual pricing sits below break-even from decile 9 onward. Repricing each decile at `break_even_apr + 3%` lifts simulated realized profit from $52.9M to $264.0M on the same accepted population — directional, not a forecast, since it holds applicant behavior constant.

## Caveats

- Expected > realized profit in the approve/decline layer: flat LGD = 0.85 and no prepayment modelling bias expectations high. Directional ranking of policies is reliable, absolute dollar levels are not.
- The 400% pricing uplift assumes no borrower elasticity (nobody walks when the APR rises) and no competitor response.
- Reject inference is unmodeled — LendingClub only publishes funded loans.
- The Kaggle snapshot is privacy-scrubbed (no FICO), so a production model with full bureau access would have a higher AUC ceiling.

## LGD sensitivity

| LGD | Optimal cutoff | Approval | Realized profit |
| ---: | ---: | ---: | ---: |
| 0.70 | 43.6% | 93.7% | $220.2M |
| 0.85 | 36.6% | 88.9% | $77.3M |
| 1.00 | 20.2% | 58.6% | $5.0M |

Model is sharply LGD-sensitive. At zero recoveries the optimal rule approves ~59% of applicants (deciles 1–5 only); a production deployment needs its own LGD estimate.

## Fair-lending audit

Four-fifths-rule disparate-impact ratios across three proxies at the operating cutoff (PD ≤ 35.4%). No flags on `addr_state` (49 states) or `home_ownership` (3 groups). One flag on `purpose`: `small_business` loans approve at 64.2% of the reference group's rate (ratio 0.68). That group's actual default rate is 35.9% — the model is responding to real risk, not inventing it. Compliance-grade mitigation is to re-price small-business loans through the pricing layer rather than decline them.

## Follow-ups

Model borrower APR elasticity to translate the pricing policy into a realistic revenue forecast. Replace the flat LGD with a conditional model fit on the recoveries column. Add a reject-inference study using LendingClub's declined-applicants file (if obtainable).
