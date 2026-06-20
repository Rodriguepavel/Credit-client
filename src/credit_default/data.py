"""Data loading and domain feature definitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import RAW_DATA_PATH

RAW_TARGET_COLUMN = "default payment next month"
TARGET_COLUMN = "default_payment_next_month"

ID_COLUMN = "ID"
LIMIT_COLUMN = "LIMIT_BAL"
DEMOGRAPHIC_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

FEATURE_COLUMNS = (
    [LIMIT_COLUMN]
    + DEMOGRAPHIC_COLUMNS
    + PAY_STATUS_COLUMNS
    + BILL_AMOUNT_COLUMNS
    + PAY_AMOUNT_COLUMNS
)
CANONICAL_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

PAY_STATUS_DOMAIN = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
EDUCATION_DOCUMENTED_DOMAIN = {1, 2, 3, 4}
MARRIAGE_DOCUMENTED_DOMAIN = {1, 2, 3}


def load_credit_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the canonical credit card default dataset.

    The Excel workbook contains a first metadata row with `X1`, `X2`, ... labels
    and a second row with the actual feature names. `header=1` keeps the business
    column names.
    """

    df = pd.read_excel(path, header=1)
    df = df.rename(columns={RAW_TARGET_COLUMN: TARGET_COLUMN})
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    return df.loc[:, CANONICAL_COLUMNS].copy()


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavior-based credit risk features."""

    enriched = df.copy()
    limit = enriched[LIMIT_COLUMN].replace(0, np.nan)

    bill_to_limit_columns = []
    payment_to_bill_columns = []
    payment_to_limit_columns = []

    for idx, (bill_col, payment_col) in enumerate(
        zip(BILL_AMOUNT_COLUMNS, PAY_AMOUNT_COLUMNS, strict=True),
        start=1,
    ):
        bill_to_limit = f"bill_to_limit_{idx}"
        payment_to_bill = f"payment_to_bill_{idx}"
        payment_to_limit = f"payment_to_limit_{idx}"

        bill = enriched[bill_col]
        payment = enriched[payment_col]

        enriched[bill_to_limit] = bill / limit
        enriched[payment_to_bill] = np.where(bill > 0, payment / bill, np.nan)
        enriched[payment_to_limit] = payment / limit

        bill_to_limit_columns.append(bill_to_limit)
        payment_to_bill_columns.append(payment_to_bill)
        payment_to_limit_columns.append(payment_to_limit)

    total_bill = enriched[BILL_AMOUNT_COLUMNS].sum(axis=1)
    total_payment = enriched[PAY_AMOUNT_COLUMNS].sum(axis=1)

    enriched["delinquency_months"] = (enriched[PAY_STATUS_COLUMNS] >= 1).sum(axis=1)
    enriched["severe_delinquency_months"] = (enriched[PAY_STATUS_COLUMNS] >= 2).sum(axis=1)
    enriched["max_delay_status"] = enriched[PAY_STATUS_COLUMNS].max(axis=1)
    enriched["recent_delay_status"] = enriched["PAY_0"]
    enriched["max_utilization"] = enriched[bill_to_limit_columns].max(axis=1)
    enriched["mean_utilization"] = enriched[bill_to_limit_columns].mean(axis=1)
    enriched["total_bill_amount"] = total_bill
    enriched["total_payment_amount"] = total_payment
    enriched["total_payment_to_bill"] = np.where(total_bill > 0, total_payment / total_bill, np.nan)
    enriched["mean_payment_to_bill"] = enriched[payment_to_bill_columns].mean(axis=1)
    enriched["mean_payment_to_limit"] = enriched[payment_to_limit_columns].mean(axis=1)

    return enriched
