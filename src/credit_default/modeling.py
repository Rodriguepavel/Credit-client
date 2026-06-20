"""Modeling utilities for credit default prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.config import PROCESSED_DATA_DIR
from credit_default.data import TARGET_COLUMN
from credit_default.preprocessing import MODEL_CATEGORICAL_COLUMNS, MODEL_NUMERIC_COLUMNS

RANDOM_STATE = 42
MODELING_BASE_PATH = PROCESSED_DATA_DIR / "credit_default_modeling_base.csv"


def load_modeling_base(path: str | Path = MODELING_BASE_PATH) -> pd.DataFrame:
    """Load the processed modeling table produced by the cleaning notebook."""

    return pd.read_csv(path)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical columns."""

    numeric_transformer: str | StandardScaler
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MODEL_NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                MODEL_CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, *, scale_numeric: bool) -> Pipeline:
    """Combine preprocessing and estimator in a single train-only pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ],
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target and identifier."""

    X = df.drop(columns=["ID", TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def positive_class_scores(estimator, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores from a fitted binary classifier."""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise TypeError("Estimator must expose predict_proba or decision_function.")


def binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute threshold-free and threshold-dependent binary metrics."""

    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_classifier(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float | str]:
    """Evaluate a fitted classifier and return a single metrics row."""

    y_score = positive_class_scores(estimator, X)
    metrics = binary_classification_metrics(y, y_score, threshold=threshold)
    return {"model": name, **metrics}


def build_threshold_table(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate precision/recall/F1 trade-offs over a threshold grid."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": sensitivity,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "predicted_positive_rate": float(y_pred.mean()),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold_by_f1(threshold_table: pd.DataFrame) -> float:
    """Choose the threshold with the best F1, breaking ties with higher recall."""

    ordered = threshold_table.sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False],
    )
    return float(ordered.iloc[0]["threshold"])


def confusion_matrix_frame(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Return a labeled 2x2 confusion matrix table."""

    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["actual_non_default", "actual_default"],
        columns=["predicted_non_default", "predicted_default"],
    )


def calibration_bin_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed default rates by score quantile bins."""

    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_score": np.asarray(y_score, dtype=float),
        }
    )
    frame["bin"] = pd.qcut(frame["y_score"], q=n_bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=False)
        .agg(
            n=("y_true", "size"),
            mean_predicted_default_probability=("y_score", "mean"),
            observed_default_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    table["abs_calibration_error"] = (
        table["observed_default_rate"] - table["mean_predicted_default_probability"]
    ).abs()
    table["weighted_abs_calibration_error"] = (
        table["abs_calibration_error"] * table["n"] / table["n"].sum()
    )
    table["bin"] = table["bin"].astype(str)
    return table


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute quantile-binned expected calibration error."""

    table = calibration_bin_table(y_true, y_score, n_bins=n_bins)
    return float(table["weighted_abs_calibration_error"].sum())
