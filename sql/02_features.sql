-- sql/02_features.sql
-- Engineer model-ready features. All outputs here are safe to use at scoring
-- time — nothing derived from the loan's post-origination performance.
DROP TABLE IF EXISTS features CASCADE;

CREATE TABLE features AS
SELECT
    id,
    issue_d,
    defaulted,
    loan_amnt,
    funded_amnt,
    recoveries,               -- carried for the realized-profit calc in cutoff.py only
    int_rate,                 -- carried for display; dropped from features in train.py
    installment,              -- monthly payment; drives exact amortized interest in cutoff.py
    CAST(REPLACE(term, ' months', '') AS INT) AS term_months,
    CAST(REPLACE(revol_util, '%', '') AS NUMERIC) AS revol_util,
    loan_amnt / NULLIF(annual_inc, 0) AS loan_to_income,
    CASE
        WHEN emp_length = '10+ years' THEN 10
        WHEN emp_length = '< 1 year' THEN 0
        WHEN emp_length IS NULL THEN NULL
        ELSE CAST(NULLIF(REGEXP_REPLACE(emp_length, '[^0-9]', '', 'g'), '') AS INT)
    END AS emp_length_num,
    purpose,
    home_ownership,
    addr_state,
    -- Winsorize DTI: a handful of rows carry extreme values (e.g. 9999) from bad source rows.
    LEAST(GREATEST(dti, 0), 60) AS dti,
    open_acc,
    pub_rec,
    total_acc,
    inq_last_6mths AS recent_inquiries,
    -- Credit history length at loan origination (years).
    EXTRACT(YEAR FROM AGE(issue_d, earliest_cr_line))
        + EXTRACT(MONTH FROM AGE(issue_d, earliest_cr_line)) / 12.0 AS credit_history_years,
    -- Window function: historical default rate by state, strictly prior to this loan.
    AVG(defaulted::NUMERIC) OVER (
        PARTITION BY addr_state
        ORDER BY issue_d
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS state_prior_default_rate
FROM loans_labeled
WHERE annual_inc > 0;
