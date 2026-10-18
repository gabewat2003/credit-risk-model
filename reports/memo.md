# Portfolio Memo — LendingClub PD + Approval-Cutoff

**Question.** Can we build a probability-of-default model on LendingClub borrower attributes alone — without grade, sub-grade, or int_rate — and turn it into a cutoff rule a credit officer could defend?

**Approach.** 2.26M loans 2007–2018 → filter to resolved outcomes (1.30M rows) → engineer 13 numeric + 3 categorical features in SQL, including a state-level prior default-rate window function → time-based split (train < 2015-01; valid < 2016-07; test ≥ 2016-07) → logistic regression baseline + LightGBM with early stopping on validation → isotonic calibration via `FrozenEstimator` → expected-profit sweep across PD cutoffs on the test set.

**Headline numbers.**
- Calibrated LightGBM: AUC 0.670, KS 0.246, Brier 0.164 on 319k test loans.
- Decile 1 default rate 6.8% vs decile 10 at 43.5% — a **6.4× lift**.
- Within a 0.02–0.60 cutoff grid, the expected-profit-maximising rule approves 98.7% of applicants at a PD ≤ 55.9%.

**Honest finding.** On realized test-set profit, the model-chosen cutoff ($818.1M) is marginally *worse* than approving everyone ($823.7M). The interest-weighted profit math is forgiving (36-month 13% loan recovers interest income across the entire term), and the dropped-by-design LendingClub-native features would have carried most of the signal. The practical value of this model is **loan tiering / pricing**, not a hard approve/decline gate.

**Chosen cutoff caveat.** The optimum sits at the upper edge of the configured grid; a strict business policy would set a lower, more conservative cutoff (e.g. PD ≤ 25% approving ~78% of applicants) and accept some foregone revenue in exchange for a tighter default rate.

**Follow-ups.** Price-tier the deciles. Add FICO (requires a non-scrubbed dataset or a bureau pull). Model LGD separately. Fair-lending audit across `addr_state` and protected proxies.
