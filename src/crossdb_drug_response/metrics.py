"""Metrics for cross-database drug-response predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def evaluate_predictions(predictions: pd.DataFrame, min_cells_per_drug: int = 5) -> pd.DataFrame:
    """Return global and equal-drug-weighted metrics for one prediction arm."""
    required = {"cell_id", "drug_id", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Prediction table is empty")
    values = predictions[["y_true", "y_pred"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions and observations must be finite")

    per_drug = []
    for _, group in predictions.groupby("drug_id", sort=True):
        if group["cell_id"].nunique() < min_cells_per_drug:
            continue
        rho = _safe_spearman(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        if np.isfinite(rho):
            per_drug.append(rho)

    residual = predictions["y_true"].to_numpy() - predictions["y_pred"].to_numpy()
    rows = [
        ("spearman_global", _safe_spearman(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy())),
        ("rmse", float(np.sqrt(np.mean(residual**2)))),
        ("mae", float(np.mean(np.abs(residual)))),
        ("spearman_drug_macro_mean", float(np.mean(per_drug)) if per_drug else float("nan")),
        ("spearman_drug_macro_median", float(np.median(per_drug)) if per_drug else float("nan")),
    ]
    return pd.DataFrame(
        {
            "metric": [row[0] for row in rows],
            "value": [row[1] for row in rows],
            "n_pairs": len(predictions),
            "n_cells": predictions["cell_id"].nunique(),
            "n_drugs": predictions["drug_id"].nunique(),
            "n_eligible_drugs": len(per_drug),
        }
    )
