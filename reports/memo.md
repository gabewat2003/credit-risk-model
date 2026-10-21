# Portfolio Memo — LendingClub PD, Approval Cutoff, Risk-Based Pricing

**Question.** Build a probability-of-default model on LendingClub borrower attributes alone — no grade, no sub-grade, no int_rate — and use it to answer two decisions a lender actually makes: who to approve, and what rate to charge.

**Approach.** 2.26M loans 2007–2018 → filter to resolved outcomes (1.30M rows) → engineer 13 numeric + 3 categorical features in SQL, including a state-level prior default-rate window function → time-based split (train < 2015-01; valid < 2016-07; test ≥ 2016-07) → logistic regression baseline + LightGBM with early stopping → isotonic calibration via `FrozenEstimator` → amortized-interest profit math driving an approve/decline sweep and a per-decile break-even APR solver (Brent's method on the amortization equation).

## Model quality

Calibrated LightGBM on 319k test loans: AUC 0.670, KS 0.245, Brier 0.164, **6.3× decile lift** (D1 default 6.9% vs D10 43.2%). Modest AUC is the price of dropping LendingClub's own grade/int_rate from the feature set; the lift metric is more informative for a rank-ordering decision.

## Decision 1: Approve/decline

| Policy | Approval rate | Default rate | Realized profit |
| --- | ---: | ---: | ---: |
| Approve everyone | 100.0% | 22.5% | $52.9M |
| **Model cutoff (PD ≤ 36.0%)** | **87.0%** | **19.7%** | **$76.9M (+45%)** |

Declining the riskiest ~13% of applicants adds $24M to realized portfolio profit on the test window.

## Decision 2: Risk-based pricing with demand elasticity

Per-decile break-even APR vs what LendingClub actually charged:

| Decile | PD | LendingClub APR | Break-even APR |
| ---: | ---: | ---: | ---: |
| 1 | 6.9% | 9.4% | 4.0% |
| 5 | 20.9% | 13.3% | 13.3% |
| 9 | 33.5% | 17.4% | 17.0% |
| 10 | 43.2% | 20.0% | 22.5% |

**Structural finding:** LendingClub's own APR curve tracks default rate roughly 1-for-1, leaving ~zero margin before LGD. Once a flat 85% LGD is applied, their actual pricing sits below break-even across deciles 6, 7, 8 and the top tier (decile 10 is 2.4pp short of break-even).

Applying a `break_even_apr + 3%` policy, with a constant-elasticity demand curve capped at the observed pool (so lowering a rate can't recruit new borrowers):

| β | Volume retained | Elastic profit | Uplift vs LC actual | Multiple |
| ---: | ---: | ---: | ---: | ---: |
| 0 (no churn upper bound) | 100.0% | $266.1M | +$213.1M | 5.0× |
| −2 | 77.5% | $192.0M | +$139.0M | 3.6× |
| **−4 (headline, explicit assumption)** | **62.6%** | **$143.5M** | **+$90.6M** | **2.7×** |
| −6 | 52.6% | $111.5M | +$58.5M | 2.1× |
| −8 | 45.8% | $90.1M | +$37.2M | 1.7× |

Policy stays **1.7× to 3.6× over the LC baseline** across the full elasticity range.

## Caveats

- Expected > realized profit in the approve/decline layer: flat LGD = 0.85 and no prepayment modelling bias expectations high. Directional ranking of policies is reliable, absolute dollar levels are not.
- The 5× no-churn pricing uplift assumes nobody walks when the APR rises and no competitor response; the headline uses β = −4 as an explicit assumption, since the published consumer-credit elasticity literature does not give a clean citation for unsecured instalment loans.
- Realized profit is `installment × term − principal` for every non-default; it does not use the `total_pymnt`/`recoveries` columns, so it also ignores prepayment and partial recoveries. The resolved-only cohort filter right-censors late-vintage loans.
- Reject inference is unmodeled — LendingClub only publishes funded loans.
- The Kaggle snapshot is privacy-scrubbed (no FICO), so a production model with full bureau access would have a higher AUC ceiling.

## LGD sensitivity

| LGD | Optimal cutoff | Approval | Realized profit |
| ---: | ---: | ---: | ---: |
| 0.70 | 44.2% | 94.2% | $221.3M |
| 0.85 | 36.0% | 87.0% | $76.9M |
| 1.00 | 23.7% | 66.9% | **−$9.3M** |

Model is sharply LGD-sensitive. At LGD = 1.0 the expected-profit sweep picks a policy that realizes a ~$9M loss, which is worse than the trivial decline-everyone policy ($0) — because `E[profit]` is calibrated with model PD and a flat LGD while realized profit uses actual outcomes, so under extreme LGD the expected-profit optimum diverges from the realized optimum. The row tells us LGD = 1.0 is the breaking point for a fixed-threshold approve/decline rule; a production deployment needs its own LGD estimate.

## Fair-lending audit

Four-fifths-rule disparate-impact ratios across three proxies at the operating cutoff (PD ≤ 36.0%). No flags on `addr_state` (49 states) or `home_ownership` (3 groups). One flag on `purpose`: `small_business` loans approve at 65% of the reference group's rate (ratio 0.69). That group's actual default rate is 35.9% — the model is responding to real risk, not inventing it. Compliance-grade mitigation is to re-price small-business loans through the pricing layer rather than decline them.

## Follow-ups

Model **adverse selection** inside the retained-after-churn pool (borrowers who walk at a higher APR are likely the lowest-risk ones; the ones who stay default more). Replace the flat LGD with a conditional model fit on the recoveries column. Add an actual-cash P&L using `total_pymnt + recoveries − funded_amnt`, and a default-rate-by-issue-quarter plot to visualise the right-censoring in the test window. Add a reject-inference study using LendingClub's declined-applicants file (if obtainable).
