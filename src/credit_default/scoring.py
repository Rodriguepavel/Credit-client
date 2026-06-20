"""Credit score scaling and risk-band utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BASE_SCORE = 600
DEFAULT_PDO = 50
DEFAULT_BAND_LABELS = [
    "E_very_high_risk",
    "D_high_risk",
    "C_medium_risk",
    "B_low_risk",
    "A_very_low_risk",
]


def build_score_scale(
    base_default_rate: float,
    *,
    base_score: int = DEFAULT_BASE_SCORE,
    pdo: int = DEFAULT_PDO,
) -> dict[str, float]:
    """Build a traditional scorecard scale based on good/bad odds.

    `pdo` means points to double the odds. Higher score means lower default risk.
    The `base_score` is assigned to the sample base odds.
    """

    if not 0 < base_default_rate < 1:
        raise ValueError("base_default_rate must be between 0 and 1.")

    base_good_bad_odds = (1 - base_default_rate) / base_default_rate
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_good_bad_odds)
    return {
        "base_score": float(base_score),
        "pdo": float(pdo),
        "base_default_rate": float(base_default_rate),
        "base_good_bad_odds": float(base_good_bad_odds),
        "factor": float(factor),
        "offset": float(offset),
    }


def probability_to_score(
    default_probability: pd.Series | np.ndarray,
    score_scale: dict[str, float],
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert default probabilities to credit scores."""

    probability = np.asarray(default_probability, dtype=float)
    probability = np.clip(probability, eps, 1 - eps)
    good_bad_odds = (1 - probability) / probability
    score = score_scale["offset"] + score_scale["factor"] * np.log(good_bad_odds)
    return score


def score_to_probability(score: pd.Series | np.ndarray, score_scale: dict[str, float]) -> np.ndarray:
    """Invert the score scale back to default probability."""

    score_array = np.asarray(score, dtype=float)
    log_good_bad_odds = (score_array - score_scale["offset"]) / score_scale["factor"]
    good_bad_odds = np.exp(log_good_bad_odds)
    return 1 / (1 + good_bad_odds)


def fit_score_band_edges(
    scores: pd.Series | np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> list[float]:
    """Fit score band cutoffs from a calibration sample."""

    score_array = np.asarray(scores, dtype=float)
    edges = np.quantile(score_array, quantiles).tolist()
    unique_edges = sorted(set(float(edge) for edge in edges))
    if len(unique_edges) != len(edges):
        raise ValueError("Score quantile edges are duplicated; reduce the number of bands.")
    return unique_edges


def assign_score_bands(
    scores: pd.Series | np.ndarray,
    edges: list[float],
    *,
    labels: list[str] | None = None,
) -> pd.Series:
    """Assign ordered credit-score bands. Higher bands mean lower risk."""

    band_labels = labels or DEFAULT_BAND_LABELS
    if len(band_labels) != len(edges) + 1:
        raise ValueError("Number of labels must equal number of edges plus one.")

    bins = [-np.inf, *edges, np.inf]
    return pd.cut(scores, bins=bins, labels=band_labels, include_lowest=True, ordered=True)


def assign_credit_decision(risk_band: pd.Series) -> pd.Series:
    """Map score bands to a simple initial credit policy."""

    decision_map = {
        "A_very_low_risk": "approve",
        "B_low_risk": "approve",
        "C_medium_risk": "approve",
        "D_high_risk": "manual_review",
        "E_very_high_risk": "reject",
    }
    return risk_band.astype("string").map(decision_map).astype("string")


def summarize_by_group(
    df: pd.DataFrame,
    group_col: str,
    *,
    target_col: str,
    score_col: str = "credit_score",
    probability_col: str = "predicted_default_probability",
) -> pd.DataFrame:
    """Summarize risk, score, and observed default rate by group."""

    summary = (
        df.groupby(group_col, observed=False)
        .agg(
            n=("ID", "size"),
            observed_defaults=(target_col, "sum"),
            observed_default_rate=(target_col, "mean"),
            mean_predicted_default_probability=(probability_col, "mean"),
            mean_credit_score=(score_col, "mean"),
            min_credit_score=(score_col, "min"),
            max_credit_score=(score_col, "max"),
        )
        .reset_index()
    )
    summary["share"] = summary["n"] / len(df)
    return summary
