# Project Methodology

This project follows a simple credit-scoring workflow. We first check whether the
data is clean enough to model, then we create features that describe repayment
behavior, fit several models, calibrate probabilities, and turn the final default
probability into a score.

## Data Checks

- Keep the raw Excel file unchanged in `data/raw`.
- Check the expected columns: credit limit, demographics, repayment status, bill
  amounts, payment amounts, and the next-month default target.
- Check row count, missing values, duplicate IDs, unexpected category codes, and
  accounting edge cases before modeling.

## Cleaning Choices

- Keep all rows unless there is a clear data-quality reason to remove them.
- Group undocumented demographic codes into explicit `other_unknown` buckets.
- Keep negative bill amounts as raw values, and add flags to identify them.
- Keep bills above the credit limit, and add utilization-stress indicators.
- Learn scaling, encoding, feature selection, and calibration only on training
  data to avoid leakage.

## Feature Engineering

- Repayment behavior: latest status, maximum delay, delinquency count, and severe
  delinquency count.
- Utilization: bill-to-limit ratios, maximum utilization, and average utilization.
- Payment capacity: payment-to-bill and payment-to-limit ratios.
- Accounting flags: negative bills, non-positive bills, bills above limit, and
  zero-payment months.
- Clean categories: demographic groups and repayment-status labels.

## Modeling

- Use stratified splits so each split has a similar default rate.
- Compare simple baselines with stronger tree-based models.
- Keep preprocessing inside pipelines when a model needs preprocessing.
- Compare ROC-AUC, PR-AUC, precision, recall, F1, Brier score, and calibration
  error.
- Choose the classification threshold on validation data, not on the test set.

## Scoring

- Convert default probabilities into a score where higher means lower risk.
- Fit score bands on development data and check default rates on test data.
- Keep the first `approve / manual_review / reject` rule simple. It can be tuned
  later depending on business targets and risk appetite.
