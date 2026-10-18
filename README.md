# Credit Risk Model with an Approval-Cutoff Decision Layer

> **One-line pitch (rewrite last):** Predicts probability of default on unsecured consumer loans, then converts that prediction into an approval cutoff by modeling expected profit, so the output is a business decision, not just an AUC.

**Stack:** PostgreSQL (feature engineering) · Python · scikit-learn · LightGBM · Altair · Docker

---

## 1. Business question

> If we approve every applicant whose predicted default probability is below a threshold *t*, what *t* maximizes expected profit, and what does that cost us in approval rate?

This is the core tradeoff a Risk & Analytics team manages: **approval rate vs. losses vs. profitability**. A model with a good AUC is only useful once someone picks a cutoff and defends it.

**WRITE (3-4 sentences):** In your own words, why a lender cares about this. Mention that approving more people grows revenue but increases charge-offs, and that the "right" cutoff depends on margins, not on the model alone.

---

## 2. Data

| Item | Detail |
|---|---|
| Source | LendingClub accepted loans, 2007-2018 (Kaggle: "Lending Club Loan Data", file `accepted_2007_through_2018Q4.csv`) |
| Grain | One row per funded loan |
| Size | ~2.2M rows, ~150 columns (most are sparse) |
| Download | `data/README.md` documents the exact Kaggle command. **Do not commit the raw CSV** (add `data/raw/` to `.gitignore`) |

**Alternative:** Home Credit Default Risk (Kaggle) if you want multi-table joins. Joins are better SQL practice, but LendingClub is simpler to explain.

### Define the target carefully (this is where most portfolio projects go wrong)

`loan_status` has many values: `Fully Paid`, `Charged Off`, `Current`, `Late (16-30 days)`, `In Grace Period`, `Default`, ...

- Keep only **`Fully Paid` (0)** and **`Charged Off` (1)**. These loans have a known outcome.
- Drop `Current` / `Late` / etc. Their outcomes aren't resolved yet; including them biases the model.
- Document how many rows this removes and why. That paragraph shows judgment.

```sql
-- sql/01_build_target.sql
CREATE TABLE loans_labeled AS
SELECT
    id,
    issue_d,
    CASE loan_status
        WHEN 'Fully Paid'  THEN 0
        WHEN 'Charged Off' THEN 1
    END AS defaulted,
    *
FROM loans_raw
WHERE loan_status IN ('Fully Paid', 'Charged Off');
```

---

## 3. How the pipeline works (read this to re-learn the flow)

```
raw CSV ──► Postgres (loans_raw)
              │  sql/01_build_target.sql      label + filter
              │  sql/02_features.sql          feature engineering
              ▼
        features table
              │  src/split.py                 TIME-based train/valid/test
              ▼
   baseline logistic regression ──┐
   LightGBM                       ├─► src/evaluate.py  (AUC, KS, calibration)
                                  ▼
                    calibrated probabilities
                                  │  src/cutoff.py     expected profit per loan
                                  ▼
              approval-rate / loss / profit tradeoff chart + memo
```

### Step A: Load into Postgres

Use `COPY` (fast) rather than `pandas.to_sql` (slow on 2M rows). Docker Compose runs Postgres and mounts `data/raw`.

```yaml
# docker-compose.yml (sketch)
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: ${PGPASSWORD}
      POSTGRES_DB: lending
    volumes:
      - ./data/raw:/data
      - pgdata:/var/lib/postgresql/data
    ports: ["5432:5432"]
volumes:
  pgdata:
```

**Gotcha:** the CSV has ~150 columns. Create the table with only the columns you need (~25) and use `COPY ... (col list)` via a staging step, or load to a wide text table first. Write down which approach you picked and why.

### Step B: Feature engineering in SQL

The point is to show SQL depth. Put real logic here, not `SELECT *`.

Features to build (pick ~15-20, justify each):

| Feature | SQL idea | Why it matters |
|---|---|---|
| `dti` | already present, cap outliers | Debt burden |
| `loan_to_income` | `loan_amnt / NULLIF(annual_inc, 0)` | Affordability |
| `credit_history_years` | `EXTRACT(YEAR FROM age(issue_d, earliest_cr_line))` | Thin-file risk |
| `revol_util` | clean `'45.2%'` strings → numeric | Utilization is a strong signal |
| `emp_length_num` | map `'10+ years'` → 10, `'< 1 year'` → 0 | Stability |
| `recent_inquiries` | `inq_last_6mths` | Credit-seeking behavior |
| `fico_mid` | `(fico_range_low + fico_range_high) / 2.0` | Bureau score |
| `term_months` | strip `' months'` | Longer term → more risk |

```sql
-- sql/02_features.sql (illustrative excerpt)
CREATE TABLE features AS
SELECT
    id,
    issue_d,
    defaulted,
    loan_amnt,
    CAST(REPLACE(term, ' months', '') AS INT)                  AS term_months,
    CAST(REPLACE(revol_util, '%', '') AS NUMERIC)              AS revol_util,
    loan_amnt / NULLIF(annual_inc, 0)                          AS loan_to_income,
    (fico_range_low + fico_range_high) / 2.0                   AS fico_mid,
    EXTRACT(YEAR FROM age(issue_d, earliest_cr_line))          AS credit_history_years,
    CASE
        WHEN emp_length = '10+ years' THEN 10
        WHEN emp_length = '< 1 year'  THEN 0
        ELSE CAST(REGEXP_REPLACE(emp_length, '[^0-9]', '', 'g') AS INT)
    END                                                        AS emp_length_num,
    purpose,
    home_ownership,
    addr_state
FROM loans_labeled
WHERE annual_inc > 0;
```

**Add one window-function feature** to show range, e.g., state-level default rate computed *only from loans issued before this one* (avoid leakage):

```sql
AVG(defaulted) OVER (
    PARTITION BY addr_state
    ORDER BY issue_d
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
) AS state_prior_default_rate
```

### Step C: Prevent leakage (document this explicitly)

Leakage = using information that wouldn't exist at application time. Drop all of these:

- `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `out_prncp`, `last_pymnt_d`, `last_pymnt_amnt`, `collection_recovery_fee`, any `hardship_*` / `settlement_*` columns.

**Decision to make and write up:** `grade`, `sub_grade`, and `int_rate` are LendingClub's *own* underwriting output. They're available at issuance but encode LendingClub's own model. Two clean options:
1. Exclude them and build your own score from raw borrower attributes (recommended, since it's the honest "build a risk model" framing).
2. Include them as a benchmark and show how much your model adds beyond them.

### Step D: Split by time, not randomly

Random splits leak macro conditions across train/test and overstate performance. Use issue date:

```python
# src/split.py
train = df[df.issue_d <  "2015-01-01"]
valid = df[(df.issue_d >= "2015-01-01") & (df.issue_d < "2016-07-01")]
test  = df[df.issue_d >= "2016-07-01"]
```

**Gotcha:** 60-month loans issued late in the dataset won't have resolved yet. Your cutoff date for test must leave room for outcomes. Check default-rate-by-issue-quarter and look for the drop-off; that drop is censoring, not improvement.

### Step E: Models

1. **Baseline: logistic regression** with standardized numerics, one-hot categoricals, median imputation. It's interpretable and the benchmark any lender would use.
2. **LightGBM** with early stopping on the validation set. Keep hyperparameters modest; don't spend a week tuning.

```python
# src/train.py (sketch)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat_cols),
])
logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
```

### Step F: Evaluate like a risk analyst, not a Kaggle competitor

| Metric | What it tells you | How to compute |
|---|---|---|
| **AUC** | Overall rank-ordering | `roc_auc_score` |
| **KS statistic** | Max separation between good/bad score distributions. Credit-risk standard | `max(tpr - fpr)` from `roc_curve` |
| **Calibration** | Does predicted 12% default actually default ~12%? | `calibration_curve`, decile table |
| **Decile table** | Default rate by score decile; should be monotonic | `pd.qcut` on predicted PD |

```python
fpr, tpr, _ = roc_curve(y_test, p_test)
ks = (tpr - fpr).max()
```

**Why calibration matters here:** the cutoff step multiplies probability × dollars. A model with great AUC but miscalibrated probabilities gives the wrong expected loss. If LightGBM is miscalibrated, fit `CalibratedClassifierCV(method="isotonic")` on the validation set and show before/after.

### Step G: The decision layer (the part that makes this project stand out)

Per-loan expected profit if approved:

```
E[profit] = (1 - PD) * interest_earned  -  PD * (LGD * principal)
```

- `interest_earned ≈ loan_amnt * int_rate * term_years` (simplified; state it as an assumption)
- `LGD` (loss given default): estimate from the data as `1 - recoveries / funded_amnt` on charged-off loans, or assume a flat value (e.g., 0.85) and run sensitivity at 0.7 / 0.85 / 1.0
- Ignore cost of funds / servicing at first, then add a flat cost-of-capital assumption as an extension

Then sweep thresholds on the **test set**:

```python
rows = []
for t in np.linspace(0.02, 0.60, 59):
    approved = test[test.pd_hat <= t]
    rows.append({
        "cutoff": t,
        "approval_rate": len(approved) / len(test),
        "realized_default_rate": approved.defaulted.mean(),
        "expected_profit": approved.exp_profit.sum(),
        "realized_profit": approved.realized_profit.sum(),  # uses actual outcomes
    })
```

Compare **expected** vs **realized** profit. If they diverge, say why (calibration drift, macro change between train and test). That honesty is a strength.

**Chart to produce:** approval rate (x) vs. expected profit (y), with the optimal cutoff marked and the current "approve everyone" baseline for comparison.

---

## 4. Results (fill with real numbers)

| Model | AUC | KS | Brier |
|---|---|---|---|
| Logistic regression | TODO | TODO | TODO |
| LightGBM | TODO | TODO | TODO |
| LightGBM (calibrated) | TODO | TODO | TODO |

**Decision result (template):**
> At a cutoff of **X%**, approving **Y%** of applicants yields **$Z** expected profit on the test set vs. **$W** approving everyone, with default rate falling from **A%** to **B%**.

**WRITE:** 1 paragraph on what surprised you (a feature that mattered more/less than expected, where the model was weak).

---

## 5. Limitations (be specific; this reads as maturity)

- **Reject inference:** LendingClub only shows funded loans, so the model never sees applicants who were rejected. Population drift between funded and all applicants is unmeasured.
- Simplified profit model (no prepayment, servicing cost, or funding cost).
- Single lender, 2007-2018; macro regime differences (post-GFC vs. late-cycle).
- Fairness not assessed here. See project 03 for fair-lending analysis.

---

## 6. AI-assisted development note

> **WRITE honestly.** Example: "I used Claude Code to scaffold the SQL and plotting code. I verified outputs by (a) reconciling row counts at each SQL stage against the raw table, (b) checking that no leakage columns reached the model by asserting on the feature list, (c) recomputing KS by hand on a 1,000-row sample, and (d) confirming the decile table is monotonic."

This is exactly what the job posting asks for ("carefully verifying outputs").

---

## 7. Reproduce

```bash
git clone <repo> && cd credit-risk-model
cp .env.example .env
docker compose up -d db
make data      # downloads data (requires Kaggle API token) and loads Postgres
make features  # runs sql/*.sql
make train     # trains + evaluates, writes reports/
make cutoff    # runs decision analysis, writes reports/cutoff.html
```

## 8. Repo layout

```
.
├── README.md
├── docker-compose.yml
├── Makefile
├── data/            # gitignored raw; README explains download
├── sql/             # 01_build_target.sql, 02_features.sql
├── src/             # split.py, train.py, evaluate.py, cutoff.py
├── notebooks/       # exploration only; not required to reproduce
└── reports/         # charts (png/html) + memo.md committed
```

## 9. Checklist before publishing

- [ ] Leakage columns asserted-out in code
- [ ] Time-based split documented
- [ ] Decile table monotonic (or explained)
- [ ] Calibration plot included
- [ ] Cutoff chart embedded in this README
- [ ] Numbers in Results table are real and reproducible
- [ ] Limitations section written
