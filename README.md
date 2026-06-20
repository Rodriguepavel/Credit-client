# Credit Default Risk Scoring

This project builds a credit loan score from the UCI credit card default dataset.
We start from the raw Excel file, clean the data, create risk features, compare
several models, calibrate the default probabilities, and convert the final risk
estimate into a score.

The project is organized so each step can be checked and rerun. The notebooks
show the analysis, while the reusable code is kept in `src/credit_default`.

## Project Structure

```text
credit client/
|-- data/
|   |-- raw/          # source file, kept unchanged
|   |-- interim/      # cleaned analytical data
|   `-- processed/    # modeling and scoring outputs
|-- docs/             # methodology notes
|-- models/           # saved model artifacts
|-- notebooks/        # analysis notebooks
|-- reports/
|   |-- figures/      # plots
|   `-- tables/       # CSV summaries
|-- scripts/          # command-line scripts
|-- src/
|   `-- credit_default/
|       |-- config.py
|       |-- data.py
|       |-- eda.py
|       |-- modeling.py
|       |-- preprocessing.py
|       `-- scoring.py
`-- tests/            # basic regression tests
```

The raw file is stored as `data/raw/default_of_credit_card_clients.xls` and is
not modified.

## Workflow

1. Check the source data and understand the target.
2. Clean category codes and accounting edge cases without dropping rows.
3. Build repayment, utilization, payment-capacity, and accounting-flag features.
4. Compare models on a stratified train / validation / test split.
5. Calibrate probabilities so the score is based on reliable default estimates.
6. Convert default probabilities into a credit score and risk bands.

## Notebooks

```text
notebooks/01_exploration_credit_default.ipynb
notebooks/02_cleaning.ipynb
notebooks/03_modeling_baselines.ipynb
notebooks/04_credit_loan_score.ipynb
notebooks/05_probability_calibration.ipynb
```

## Current Model Choice

The best raw model in the current run is `catboost_plain`, selected in notebook
`03` by validation PR-AUC. Notebook `05` then checks probability calibration and
uses the best probability model to rebuild the score.

## Reproduce Outputs

```powershell
python -m pip install -r requirements-dev.txt
python scripts/01_run_initial_eda.py
```

Then run notebooks `01` to `05` in order.

## Main Outputs

```text
data/interim/credit_default_clean.csv
data/processed/credit_default_modeling_base.csv
data/processed/credit_default_scored.csv
data/processed/credit_default_scored_calibrated.csv
models/best_baseline_model.joblib
models/best_calibrated_score_model.joblib
reports/tables/model_validation_metrics.csv
reports/tables/calibration_test_metrics.csv
reports/tables/calibrated_credit_score_test_band_summary.csv
```

## Quality Checks

```powershell
python -m ruff check .
python -m pytest
```
