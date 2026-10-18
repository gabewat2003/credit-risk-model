# Portfolio Memo — LendingClub PD + Approval-Cutoff

**Question.** Can we build a probability-of-default model on LendingClub borrower attributes alone — without grade, sub-grade, or int_rate — and turn it into a cutoff rule a credit officer could defend?

**Approach.** 2.26M loans 2007–2018 → filter to resolved outcomes (1.30M rows) → engineer 13 numeric + 3 categorical features in SQL, including a state-level prior default-rate window function → time-based split (train < 2015-01; valid < 2016-07; test ≥ 2016-07) → logistic regression baseline + LightGBM with early stopping on validation → isotonic calibration via `FrozenEstimator` → expected-profit sweep across PD cutoffs on the test set, using exact amortized interest from the published `installment` column.

**Headline numbers (319k test loans).**

| Policy | Approval rate | Default rate | Realized profit |
| --- | ---: | ---: | ---: |
| Approve everyone | 100% | 20.1% | $52.9M |
| **Model cutoff (PD ≤ 35.4%)** | **86.4%** | **19.6%** | **$73.0M** |

The model-chosen cutoff adds **$20M of realized profit** (+38%) on top of a blanket-approve baseline by declining the riskiest ~14% of applicants.

**Model quality.** Calibrated LightGBM: AUC 0.669, KS 0.244, Brier 0.164. **Decile 1 default rate 6.8% vs decile 10 at 43.5%** — a **6.4× rank-ordering lift**. The modest AUC is a function of deliberately dropping LendingClub's own grade/int_rate; what remains is the lender's view of a borrower built from self-reported attributes and public credit-bureau telemetry.

**Honest caveats.**
- Expected profit ($177.6M at the optimal cutoff) sits well above realized ($73.0M). The gap is driven by the flat LGD = 0.85 assumption and the absence of prepayment modelling; the directional ranking of policies is reliable, the absolute dollar level is not.
- Reject inference is unmodeled — LendingClub only publishes funded loans.
- The Kaggle snapshot is privacy-scrubbed (no FICO), so this is a lower-bound of what the model would do with full bureau data.

**Follow-ups.** Risk-based pricing per decile (deciles 6–10 need higher APR to break even — natural project 02). Fair-lending audit across `addr_state` and protected proxies. LGD sensitivity across `[0.7, 0.85, 1.0]`.
