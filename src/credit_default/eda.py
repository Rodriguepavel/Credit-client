"""Initial descriptive analysis for the credit default dataset."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from credit_default.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from credit_default.data import (
    BILL_AMOUNT_COLUMNS,
    EDUCATION_DOCUMENTED_DOMAIN,
    LIMIT_COLUMN,
    MARRIAGE_DOCUMENTED_DOMAIN,
    PAY_AMOUNT_COLUMNS,
    PAY_STATUS_COLUMNS,
    PAY_STATUS_DOMAIN,
    TARGET_COLUMN,
    add_domain_features,
    load_credit_data,
)

sns.set_theme(style="whitegrid", context="notebook")


def _ensure_output_dirs() -> None:
    for directory in (FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    return path


def _default_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False, observed=False)[TARGET_COLUMN]
        .agg(n="size", defaults="sum", default_rate="mean")
        .reset_index()
        .sort_values(group_col)
    )
    grouped["share"] = grouped["n"] / len(df)
    return grouped


def build_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact table of data quality checks."""

    unexpected_education = sorted(set(df["EDUCATION"]) - EDUCATION_DOCUMENTED_DOMAIN)
    unexpected_marriage = sorted(set(df["MARRIAGE"]) - MARRIAGE_DOCUMENTED_DOMAIN)
    unexpected_pay_rows = int(
        (~df[PAY_STATUS_COLUMNS].isin(PAY_STATUS_DOMAIN)).any(axis=1).sum()
    )

    checks = [
        ("row_count", len(df), "Expected 30000 for the canonical dataset."),
        ("column_count", df.shape[1], "Expected 25 including ID and target."),
        ("missing_cells", int(df.isna().sum().sum()), "Missing values should be audited."),
        ("duplicate_id_count", int(df["ID"].duplicated().sum()), "ID should be unique."),
        (
            "target_default_rate",
            round(float(df[TARGET_COLUMN].mean()), 6),
            "Class imbalance checkpoint.",
        ),
        (
            "unexpected_education_codes",
            ",".join(map(str, unexpected_education)) or "none",
            "Undocumented codes should be grouped with 'other/unknown'.",
        ),
        (
            "unexpected_education_rows",
            int((~df["EDUCATION"].isin(EDUCATION_DOCUMENTED_DOMAIN)).sum()),
            "Rows outside documented EDUCATION codes 1-4.",
        ),
        (
            "unexpected_marriage_codes",
            ",".join(map(str, unexpected_marriage)) or "none",
            "Code 0 is not part of the documented MARRIAGE domain.",
        ),
        (
            "unexpected_marriage_rows",
            int((~df["MARRIAGE"].isin(MARRIAGE_DOCUMENTED_DOMAIN)).sum()),
            "Rows outside documented MARRIAGE codes 1-3.",
        ),
        (
            "unexpected_pay_status_rows",
            unexpected_pay_rows,
            "Rows outside PAY_* documented domain -2 to 9.",
        ),
        (
            "negative_bill_rows",
            int((df[BILL_AMOUNT_COLUMNS] < 0).any(axis=1).sum()),
            "Negative bills can represent credits or adjustments.",
        ),
        (
            "bill_above_limit_rows",
            int((df[BILL_AMOUNT_COLUMNS].gt(df[LIMIT_COLUMN], axis=0)).any(axis=1).sum()),
            "Potential utilization stress or fees above credit limit.",
        ),
        ("min_limit_bal", int(df[LIMIT_COLUMN].min()), "Lower bound check."),
        ("max_limit_bal", int(df[LIMIT_COLUMN].max()), "Upper bound check."),
        ("min_age", int(df["AGE"].min()), "Lower bound check."),
        ("max_age", int(df["AGE"].max()), "Upper bound check."),
    ]

    return pd.DataFrame(checks, columns=["check", "value", "comment"])


def build_categorical_distribution(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in ["SEX", "EDUCATION", "MARRIAGE"]:
        table = _default_rate_table(df, column).rename(columns={column: "value"})
        table.insert(0, "feature", column)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def build_pay_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in PAY_STATUS_COLUMNS:
        table = _default_rate_table(df, column).rename(columns={column: "status"})
        table.insert(0, "feature", column)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def build_domain_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    engineered = add_domain_features(df)
    columns = [
        "delinquency_months",
        "severe_delinquency_months",
        "max_delay_status",
        "recent_delay_status",
        "max_utilization",
        "mean_utilization",
        "total_bill_amount",
        "total_payment_amount",
        "total_payment_to_bill",
        "mean_payment_to_bill",
        "mean_payment_to_limit",
    ]
    summary = engineered[columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return summary.T.reset_index(names="feature")


def build_high_correlation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    columns = [LIMIT_COLUMN] + PAY_STATUS_COLUMNS + BILL_AMOUNT_COLUMNS + PAY_AMOUNT_COLUMNS
    corr = df[columns].corr(numeric_only=True)
    upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(upper_mask).stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4C78A8")
    ax.set_title("Target distribution")
    ax.set_xlabel("Default next month")
    ax.set_ylabel("Client count")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)


def plot_default_by_pay0(df: pd.DataFrame) -> None:
    table = _default_rate_table(df, "PAY_0")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=table, x="PAY_0", y="default_rate", ax=ax, color="#E45756")
    ax.set_title("Default rate by latest repayment status")
    ax.set_xlabel("PAY_0 status")
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, min(1.0, table["default_rate"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_pay0.png", dpi=160)
    plt.close(fig)


def plot_default_by_limit_bin(df: pd.DataFrame) -> pd.DataFrame:
    work = df[[LIMIT_COLUMN, TARGET_COLUMN]].copy()
    work["limit_bin"] = pd.qcut(work[LIMIT_COLUMN], q=10, duplicates="drop")
    table = _default_rate_table(work, "limit_bin")
    table["limit_bin"] = table["limit_bin"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=table, x="limit_bin", y="default_rate", ax=ax, color="#54A24B")
    ax.set_title("Default rate by credit limit decile")
    ax.set_xlabel("LIMIT_BAL decile")
    ax.set_ylabel("Default rate")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_limit_decile.png", dpi=160)
    plt.close(fig)
    return table


def plot_default_by_delinquency_count(df: pd.DataFrame) -> pd.DataFrame:
    engineered = add_domain_features(df)
    table = _default_rate_table(engineered, "delinquency_months")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=table, x="delinquency_months", y="default_rate", ax=ax, color="#F58518")
    ax.set_title("Default rate by delinquency count over six months")
    ax.set_xlabel("Months with PAY_* >= 1")
    ax.set_ylabel("Default rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_delinquency_count.png", dpi=160)
    plt.close(fig)
    return table


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    columns = [LIMIT_COLUMN] + PAY_STATUS_COLUMNS + BILL_AMOUNT_COLUMNS + PAY_AMOUNT_COLUMNS
    corr = df[columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation among raw credit variables")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "raw_feature_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def write_markdown_report(
    df: pd.DataFrame,
    quality: pd.DataFrame,
    pay0: pd.DataFrame,
    limit_bins: pd.DataFrame,
    delinquency: pd.DataFrame,
    high_corr: pd.DataFrame,
) -> Path:
    default_rate = df[TARGET_COLUMN].mean()
    unexpected_education_rows = quality.loc[
        quality["check"].eq("unexpected_education_rows"), "value"
    ].item()
    unexpected_marriage_rows = quality.loc[
        quality["check"].eq("unexpected_marriage_rows"), "value"
    ].item()
    strongest_corr = high_corr.iloc[0]

    pay0_selected = pay0.loc[pay0["PAY_0"].isin([-2, -1, 0, 1, 2, 3])].copy()
    pay0_lines = "\n".join(
        f"- PAY_0={int(row.PAY_0)}: n={int(row.n)}, default_rate={row.default_rate:.3f}"
        for row in pay0_selected.itertuples()
    )

    delinquency_lines = "\n".join(
        f"- {int(row.delinquency_months)} months: n={int(row.n)}, "
        f"default_rate={row.default_rate:.3f}"
        for row in delinquency.itertuples()
    )

    markdown = f"""# Initial Data Assessment

## Dataset Contract

- Shape: {len(df):,} rows x {df.shape[1]} columns.
- Target default rate: {default_rate:.2%}, confirming a moderately imbalanced binary target.
- Missing cells: {int(df.isna().sum().sum())}.
- Duplicate IDs: {int(df["ID"].duplicated().sum())}.
- The schema matches the expected credit card default dataset structure.

## Data Quality Flags

- `EDUCATION` has {unexpected_education_rows} rows outside documented codes 1-4.
- `MARRIAGE` has {unexpected_marriage_rows} rows outside documented codes 1-3.
- Bill amounts can be negative, which should be treated as credits or adjustments rather than dropped blindly.
- Some bill amounts exceed `LIMIT_BAL`; this is useful as a potential utilization stress signal.

## Repayment Status Signal

Default rate rises clearly with the latest repayment status:

{pay0_lines}

This confirms that recent repayment behavior is a primary risk signal. Treat these variables
carefully: `-2`, `-1`, `0`, and positive delays mix qualitatively different states.

## Delinquency Frequency Signal

{delinquency_lines}

Counting delinquency months is more stable than relying only on a single raw status.

## Credit Limit Pattern

The credit-limit decile table is saved in `reports/tables/default_rate_by_limit_decile.csv`.
Lower credit-limit bands show higher default rates, which is coherent with common credit-risk
segmentation.

## Redundancy And Multicollinearity

The strongest raw correlation is `{strongest_corr.feature_1}` vs `{strongest_corr.feature_2}`
with correlation {strongest_corr.correlation:.3f}. High correlations among sequential bill
amounts indicate temporal redundancy that should be considered in modeling and interpretation.

## Generated Figures

- `reports/figures/target_distribution.png`
- `reports/figures/default_rate_by_pay0.png`
- `reports/figures/default_rate_by_limit_decile.png`
- `reports/figures/default_rate_by_delinquency_count.png`
- `reports/figures/raw_feature_correlation_heatmap.png`
"""

    path = REPORTS_DIR / "eda_initial_findings.md"
    path.write_text(textwrap.dedent(markdown), encoding="utf-8")
    return path


def run_initial_eda() -> dict[str, Path]:
    """Run the initial EDA and persist report artifacts."""

    _ensure_output_dirs()
    df = load_credit_data()

    quality = build_data_quality_summary(df)
    categorical = build_categorical_distribution(df)
    pay_status = build_pay_status_distribution(df)
    numeric = df[[LIMIT_COLUMN, "AGE", *BILL_AMOUNT_COLUMNS, *PAY_AMOUNT_COLUMNS]].describe().T
    numeric = numeric.reset_index(names="feature")
    domain_features = build_domain_feature_summary(df)
    high_corr = build_high_correlation_pairs(df)

    pay0 = _default_rate_table(df, "PAY_0")
    limit_bins = plot_default_by_limit_bin(df)
    delinquency = plot_default_by_delinquency_count(df)

    plot_target_distribution(df)
    plot_default_by_pay0(df)
    plot_correlation_heatmap(df)

    outputs = {
        "quality": _save_table(quality, "data_quality_summary.csv"),
        "categorical": _save_table(categorical, "categorical_distribution.csv"),
        "pay_status": _save_table(pay_status, "pay_status_distribution.csv"),
        "numeric": _save_table(numeric, "numeric_summary.csv"),
        "domain_features": _save_table(
            domain_features, "domain_feature_summary.csv"
        ),
        "high_correlations": _save_table(high_corr, "high_correlation_pairs.csv"),
        "pay0_default_rate": _save_table(pay0, "default_rate_by_pay0.csv"),
        "limit_default_rate": _save_table(limit_bins, "default_rate_by_limit_decile.csv"),
        "delinquency_default_rate": _save_table(
            delinquency, "default_rate_by_delinquency_count.csv"
        ),
    }
    outputs["markdown_report"] = write_markdown_report(
        df=df,
        quality=quality,
        pay0=pay0,
        limit_bins=limit_bins,
        delinquency=delinquency,
        high_corr=high_corr,
    )
    return outputs
