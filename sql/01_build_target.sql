-- sql/01_build_target.sql
-- Filter loans to resolved outcomes (Fully Paid / Charged Off), build the
-- binary `defaulted` target, and cast raw TEXT columns to proper types.
--
-- LendingClub date columns ship as 'Mon-YYYY' strings (e.g. "Dec-2015") so we
-- parse them with TO_DATE. The Kaggle snapshot used here is privacy-scrubbed
-- and does not carry fico_range_low / fico_range_high.
DROP TABLE IF EXISTS loans_labeled CASCADE;

CREATE TABLE loans_labeled AS
SELECT
    id,
    TO_DATE(issue_d, 'Mon-YYYY') AS issue_d,
    CASE loan_status
        WHEN 'Fully Paid'  THEN 0
        WHEN 'Charged Off' THEN 1
    END AS defaulted,
    CAST(loan_amnt AS NUMERIC) AS loan_amnt,
    CAST(funded_amnt AS NUMERIC) AS funded_amnt,
    term,
    CAST(REPLACE(int_rate, '%', '') AS NUMERIC) AS int_rate,
    CAST(installment AS NUMERIC) AS installment,
    grade,
    sub_grade,
    emp_length,
    home_ownership,
    CAST(annual_inc AS NUMERIC) AS annual_inc,
    verification_status,
    purpose,
    addr_state,
    CAST(dti AS NUMERIC) AS dti,
    CAST(delinq_2yrs AS INT) AS delinq_2yrs,
    TO_DATE(earliest_cr_line, 'Mon-YYYY') AS earliest_cr_line,
    CAST(inq_last_6mths AS INT) AS inq_last_6mths,
    CAST(open_acc AS INT) AS open_acc,
    CAST(pub_rec AS INT) AS pub_rec,
    CAST(revol_bal AS NUMERIC) AS revol_bal,
    revol_util,
    CAST(total_acc AS INT) AS total_acc,
    CAST(recoveries AS NUMERIC) AS recoveries
FROM loans_raw
WHERE loan_status IN ('Fully Paid', 'Charged Off')
  AND issue_d IS NOT NULL
  AND earliest_cr_line IS NOT NULL;
