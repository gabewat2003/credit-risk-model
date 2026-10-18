-- sql/02_features.sql
DROP TABLE IF EXISTS features CASCADE;

CREATE TABLE features AS
SELECT
    id,
    issue_d,
    defaulted,
    loan_amnt,
    funded_amnt,
    recoveries,
    int_rate,
    CAST(REPLACE(term, ' months', '') AS INT) AS term_months,
    CAST(REPLACE(revol_util, '%', '') AS NUMERIC) AS revol_util,
    loan_amnt / NULLIF(annual_inc, 0) AS loan_to_income,
    CASE 
        WHEN emp_length = '10+ years' THEN 10
        WHEN emp_length = '< 1 year' THEN 0
        ELSE CAST(REGEXP_REPLACE(emp_length, '[^0-9]', '', 'g') AS INT)
    END AS emp_length_num,
    purpose,
    home_ownership,
    addr_state,
    dti,
    open_acc,
    pub_rec,
    total_acc,
    inq_last_6mths,
    -- Window function: Historical default rate by state prior to the current loan issue date
    AVG(defaulted::NUMERIC) OVER (
        PARTITION BY addr_state
        ORDER BY issue_d
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS state_prior_default_rate
FROM loans_labeled
WHERE annual_inc > 0;
