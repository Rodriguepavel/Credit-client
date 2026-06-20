"""Cleaning and feature preparation for credit default modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_default.data import (
    BILL_AMOUNT_COLUMNS,
    CANONICAL_COLUMNS,
    LIMIT_COLUMN,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    TARGET_COLUMN,
)

SEX_GROUP_MAP = {
    1: "male",
    2: "female",
}

EDUCATION_GROUP_MAP = {
    0: "other_unknown",
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "other_unknown",
    5: "other_unknown",
    6: "other_unknown",
}

MARRIAGE_GROUP_MAP = {
    0: "other_unknown",
    1: "married",
    2: "single",
    3: "other_unknown",
}

PAY_STATUS_MAP = {
    -2: "no_consumption",
    -1: "paid_duly",
    0: "revolving_credit",
    1: "delay_1_month",
    2: "delay_2_months",
    3: "delay_3_months",
    4: "delay_4_months",
    5: "delay_5_months",
    6: "delay_6_months",
    7: "delay_7_months",
    8: "delay_8_months",
    9: "delay_9_plus_months",
}

CLEAN_DEMOGRAPHIC_COLUMNS = ["sex_group", "education_group", "marriage_group"]
PAY_STATUS_CATEGORY_COLUMNS = [f"{column}_category" for column in PAY_STATUS_COLUMNS]
BILL_TO_LIMIT_COLUMNS = [f"bill_to_limit_{idx}" for idx in range(1, 7)]
PAYMENT_TO_BILL_COLUMNS = [f"payment_to_bill_{idx}" for idx in range(1, 7)]
PAYMENT_TO_LIMIT_COLUMNS = [f"payment_to_limit_{idx}" for idx in range(1, 7)]

BEHAVIORAL_FEATURE_COLUMNS = [
    "delinquency_months",
    "severe_delinquency_months",
    "max_delay_status",
    "recent_delay_status",
    "mean_positive_delay",
    "months_no_consumption",
    "months_paid_duly",
    "months_revolving_credit",
    "negative_bill_months",
    "non_positive_bill_months",
    "bill_above_limit_months",
    "zero_payment_months",
    "max_utilization",
    "mean_utilization",
    "total_bill_amount",
    "total_positive_bill_amount",
    "total_payment_amount",
    "total_payment_to_bill",
    "mean_payment_to_bill",
    "mean_payment_to_limit",
]

MODEL_NUMERIC_COLUMNS = (
    [LIMIT_COLUMN, "AGE"]
    + BILL_AMOUNT_COLUMNS
    + PAY_AMOUNT_COLUMNS
    + BILL_TO_LIMIT_COLUMNS
    + PAYMENT_TO_BILL_COLUMNS
    + PAYMENT_TO_LIMIT_COLUMNS
    + BEHAVIORAL_FEATURE_COLUMNS
)

MODEL_CATEGORICAL_COLUMNS = CLEAN_DEMOGRAPHIC_COLUMNS + PAY_STATUS_CATEGORY_COLUMNS

MODELING_BASE_COLUMNS = (
    ["ID"]
    + MODEL_NUMERIC_COLUMNS
    + MODEL_CATEGORICAL_COLUMNS
    + [TARGET_COLUMN]
)


def _map_to_group(series: pd.Series, mapping: dict[int, str], unknown_value: str) -> pd.Series:
    """Map integer-coded categories to semantic labels with a stable unknown bucket."""

    return series.map(mapping).fillna(unknown_value).astype("string")


def add_clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add semantic demographic and payment-status categories.

    Raw columns are intentionally preserved. The model-ready dataset can decide
    whether to keep or drop the original numeric codes.
    """

    clean = df.copy()
    clean["sex_group"] = _map_to_group(clean["SEX"], SEX_GROUP_MAP, "unknown")
    clean["education_group"] = _map_to_group(
        clean["EDUCATION"],
        EDUCATION_GROUP_MAP,
        "other_unknown",
    )
    clean["marriage_group"] = _map_to_group(
        clean["MARRIAGE"],
        MARRIAGE_GROUP_MAP,
        "other_unknown",
    )

    for column in PAY_STATUS_COLUMNS:
        clean[f"{column}_category"] = _map_to_group(
            clean[column],
            PAY_STATUS_MAP,
            "unknown_status",
        )

    return clean


def add_clean_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready behavioral credit-risk features.

    Negative bill amounts are not dropped. They are flagged, while utilization and
    payment-to-bill ratios use the non-negative bill balance because negative bills
    usually reflect credits or accounting adjustments rather than current exposure.
    """

    clean = df.copy()
    limit = clean[LIMIT_COLUMN].replace(0, np.nan)

    positive_bill_columns = []

    for idx, (bill_col, payment_col) in enumerate(
        zip(BILL_AMOUNT_COLUMNS, PAY_AMOUNT_COLUMNS, strict=True),
        start=1,
    ):
        positive_bill = clean[bill_col].clip(lower=0)
        positive_bill_columns.append(f"positive_bill_amount_{idx}")
        clean[f"positive_bill_amount_{idx}"] = positive_bill

        clean[f"bill_to_limit_{idx}"] = (positive_bill / limit).fillna(0)
        clean[f"payment_to_bill_{idx}"] = np.where(
            positive_bill > 0,
            clean[payment_col] / positive_bill,
            0,
        )
        clean[f"payment_to_limit_{idx}"] = (clean[payment_col] / limit).fillna(0)

        clean[f"negative_bill_flag_{idx}"] = (clean[bill_col] < 0).astype(int)
        clean[f"non_positive_bill_flag_{idx}"] = (clean[bill_col] <= 0).astype(int)
        clean[f"bill_above_limit_flag_{idx}"] = (clean[bill_col] > clean[LIMIT_COLUMN]).astype(int)
        clean[f"zero_payment_flag_{idx}"] = (clean[payment_col] == 0).astype(int)

    positive_delay = clean[PAY_STATUS_COLUMNS].clip(lower=0)
    bill_to_limit = clean[BILL_TO_LIMIT_COLUMNS]
    payment_to_bill = clean[PAYMENT_TO_BILL_COLUMNS]
    payment_to_limit = clean[PAYMENT_TO_LIMIT_COLUMNS]

    total_positive_bill = clean[positive_bill_columns].sum(axis=1)
    total_payment = clean[PAY_AMOUNT_COLUMNS].sum(axis=1)

    clean["delinquency_months"] = (clean[PAY_STATUS_COLUMNS] >= 1).sum(axis=1)
    clean["severe_delinquency_months"] = (clean[PAY_STATUS_COLUMNS] >= 2).sum(axis=1)
    clean["max_delay_status"] = clean[PAY_STATUS_COLUMNS].max(axis=1)
    clean["recent_delay_status"] = clean["PAY_0"]
    clean["mean_positive_delay"] = positive_delay.mean(axis=1)
    clean["months_no_consumption"] = (clean[PAY_STATUS_COLUMNS] == -2).sum(axis=1)
    clean["months_paid_duly"] = (clean[PAY_STATUS_COLUMNS] == -1).sum(axis=1)
    clean["months_revolving_credit"] = (clean[PAY_STATUS_COLUMNS] == 0).sum(axis=1)

    clean["negative_bill_months"] = (clean[BILL_AMOUNT_COLUMNS] < 0).sum(axis=1)
    clean["non_positive_bill_months"] = (clean[BILL_AMOUNT_COLUMNS] <= 0).sum(axis=1)
    clean["bill_above_limit_months"] = clean[
        [f"bill_above_limit_flag_{idx}" for idx in range(1, 7)]
    ].sum(axis=1)
    clean["zero_payment_months"] = (clean[PAY_AMOUNT_COLUMNS] == 0).sum(axis=1)

    clean["max_utilization"] = bill_to_limit.max(axis=1)
    clean["mean_utilization"] = bill_to_limit.mean(axis=1)
    clean["total_bill_amount"] = clean[BILL_AMOUNT_COLUMNS].sum(axis=1)
    clean["total_positive_bill_amount"] = total_positive_bill
    clean["total_payment_amount"] = total_payment
    clean["total_payment_to_bill"] = np.where(
        total_positive_bill > 0,
        total_payment / total_positive_bill,
        0,
    )
    clean["mean_payment_to_bill"] = payment_to_bill.mean(axis=1)
    clean["mean_payment_to_limit"] = payment_to_limit.mean(axis=1)

    return clean


def clean_credit_default_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned analytical dataset while preserving raw source columns."""

    missing_columns = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing expected raw columns: {missing_columns}")

    clean = add_clean_categorical_features(df)
    clean = add_clean_behavioral_features(clean)
    return clean


def build_modeling_base(df: pd.DataFrame) -> pd.DataFrame:
    """Select a modeling-ready table from raw or already-cleaned data."""

    clean = clean_credit_default_data(df) if "education_group" not in df.columns else df.copy()
    missing_columns = sorted(set(MODELING_BASE_COLUMNS) - set(clean.columns))
    if missing_columns:
        raise ValueError(f"Missing expected modeling columns: {missing_columns}")

    modeling = clean.loc[:, MODELING_BASE_COLUMNS].copy()
    numeric_missing = modeling[MODEL_NUMERIC_COLUMNS].isna().sum().sum()
    categorical_missing = modeling[MODEL_CATEGORICAL_COLUMNS].isna().sum().sum()
    if numeric_missing or categorical_missing:
        raise ValueError(
            "Modeling base contains missing values: "
            f"{numeric_missing} numeric, {categorical_missing} categorical."
        )
    return modeling


def build_cleaning_summary(raw: pd.DataFrame, clean: pd.DataFrame, modeling: pd.DataFrame) -> pd.DataFrame:
    """Summarize the effects of cleaning decisions."""

    summary = [
        ("raw_rows", len(raw), "Rows in source data."),
        ("clean_rows", len(clean), "Rows after cleaning. No row is dropped."),
        ("modeling_rows", len(modeling), "Rows in modeling base."),
        ("raw_columns", raw.shape[1], "Columns in source data."),
        ("clean_columns", clean.shape[1], "Columns after semantic features and flags."),
        ("modeling_columns", modeling.shape[1], "Columns retained for baseline modeling."),
        (
            "target_rate_raw",
            round(float(raw[TARGET_COLUMN].mean()), 6),
            "Default rate before cleaning.",
        ),
        (
            "target_rate_modeling",
            round(float(modeling[TARGET_COLUMN].mean()), 6),
            "Default rate after cleaning.",
        ),
        (
            "education_other_unknown_rows",
            int((clean["education_group"] == "other_unknown").sum()),
            "Includes documented 'other' and undocumented 0/5/6.",
        ),
        (
            "marriage_other_unknown_rows",
            int((clean["marriage_group"] == "other_unknown").sum()),
            "Includes documented 'other' and undocumented 0.",
        ),
        (
            "negative_bill_rows",
            int((clean["negative_bill_months"] > 0).sum()),
            "Rows flagged, not removed.",
        ),
        (
            "bill_above_limit_rows",
            int((clean["bill_above_limit_months"] > 0).sum()),
            "Rows flagged, not removed.",
        ),
        (
            "modeling_missing_cells",
            int(modeling.isna().sum().sum()),
            "Should be zero for the exported modeling base.",
        ),
    ]
    return pd.DataFrame(summary, columns=["check", "value", "comment"])
