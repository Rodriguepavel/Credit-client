# Initial Data Assessment

## Dataset Contract

- Shape: 30,000 rows x 25 columns.
- Target default rate: 22.12%, confirming a moderately imbalanced binary target.
- Missing cells: 0.
- Duplicate IDs: 0.
- The schema matches the expected credit card default dataset structure.

## Data Quality Flags

- `EDUCATION` has 345 rows outside documented codes 1-4.
- `MARRIAGE` has 54 rows outside documented codes 1-3.
- Bill amounts can be negative, which should be treated as credits or adjustments rather than dropped blindly.
- Some bill amounts exceed `LIMIT_BAL`; this is useful as a potential utilization stress signal.

## Repayment Status Signal

Default rate rises clearly with the latest repayment status:

- PAY_0=-2: n=2759, default_rate=0.132
- PAY_0=-1: n=5686, default_rate=0.168
- PAY_0=0: n=14737, default_rate=0.128
- PAY_0=1: n=3688, default_rate=0.339
- PAY_0=2: n=2667, default_rate=0.691
- PAY_0=3: n=322, default_rate=0.758

This confirms that recent repayment behavior is a primary risk signal. Treat these variables
carefully: `-2`, `-1`, `0`, and positive delays mix qualitatively different states.

## Delinquency Frequency Signal

- 0 months: n=19931, default_rate=0.117
- 1 months: n=4426, default_rate=0.298
- 2 months: n=1899, default_rate=0.388
- 3 months: n=1154, default_rate=0.509
- 4 months: n=951, default_rate=0.573
- 5 months: n=298, default_rate=0.574
- 6 months: n=1341, default_rate=0.703

Counting delinquency months is more stable than relying only on a single raw status.

## Credit Limit Pattern

The credit-limit decile table is saved in `reports/tables/default_rate_by_limit_decile.csv`.
Lower credit-limit bands show higher default rates, which is in line with basic credit-risk
segmentation.

## Redundancy And Multicollinearity

The strongest raw correlation is `BILL_AMT1` vs `BILL_AMT2`
with correlation 0.951. High correlations among sequential bill
amounts indicate temporal redundancy that should be considered in modeling and interpretation.

## Figures

- `reports/figures/target_distribution.png`
- `reports/figures/default_rate_by_pay0.png`
- `reports/figures/default_rate_by_limit_decile.png`
- `reports/figures/default_rate_by_delinquency_count.png`
- `reports/figures/raw_feature_correlation_heatmap.png`
