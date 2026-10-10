"""Feature-free baselines that expose entity and split leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdditiveBiasRegressor:
    """Regularized global, cell, and drug effects fitted by backfitting."""

    def __init__(self, regularization: float = 10.0, iterations: int = 20):
        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.regularization = float(regularization)
        self.iterations = int(iterations)

    def fit(self, frame: pd.DataFrame) -> "AdditiveBiasRegressor":
        self.global_mean_ = float(frame["sensitivity"].mean())
        self.cell_bias_: dict[str, float] = {}
        self.drug_bias_: dict[str, float] = {}

        work = frame[["cell_id", "drug_id", "sensitivity"]].copy()
        for _ in range(self.iterations):
            drug_effect = work["drug_id"].map(self.drug_bias_).fillna(0.0)
            cell_residual = work["sensitivity"] - self.global_mean_ - drug_effect
            cell_sum = cell_residual.groupby(work["cell_id"]).sum()
            cell_count = work.groupby("cell_id").size()
            self.cell_bias_ = (
                cell_sum / (cell_count + self.regularization)
            ).to_dict()

            cell_effect = work["cell_id"].map(self.cell_bias_).fillna(0.0)
            drug_residual = work["sensitivity"] - self.global_mean_ - cell_effect
            drug_sum = drug_residual.groupby(work["drug_id"]).sum()
            drug_count = work.groupby("drug_id").size()
            self.drug_bias_ = (
                drug_sum / (drug_count + self.regularization)
            ).to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_mean_"):
            raise RuntimeError("Model must be fitted before prediction")
        cell = frame["cell_id"].map(self.cell_bias_).fillna(0.0).to_numpy()
        drug = frame["drug_id"].map(self.drug_bias_).fillna(0.0).to_numpy()
        return self.global_mean_ + cell + drug
