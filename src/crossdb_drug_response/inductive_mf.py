"""Inductive matrix factorization for seen and feature-only entities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class InductiveMatrixFactorization:
    """ALS response factors with ridge projections from entity features."""

    def __init__(
        self,
        rank: int = 32,
        regularization: float = 1.0,
        iterations: int = 30,
        projection_alpha: float = 10.0,
        seed: int = 104729,
    ):
        self.rank = int(rank)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.projection_alpha = float(projection_alpha)
        self.seed = int(seed)

    def fit(
        self,
        responses: pd.DataFrame,
        cell_features: np.ndarray,
        cell_ids: np.ndarray,
        drug_features: np.ndarray,
        drug_ids: np.ndarray,
    ) -> "InductiveMatrixFactorization":
        if responses.duplicated(["cell_id", "drug_id"]).any():
            raise ValueError("Training response edges must be unique")
        self.all_cell_ids_ = np.asarray(cell_ids, dtype=str)
        self.all_drug_ids_ = np.asarray(drug_ids, dtype=str)
        self.all_cell_lookup_ = {value: i for i, value in enumerate(self.all_cell_ids_)}
        self.all_drug_lookup_ = {value: i for i, value in enumerate(self.all_drug_ids_)}
        self.source_cell_ids_ = np.asarray(sorted(responses.cell_id.astype(str).unique()))
        self.source_drug_ids_ = np.asarray(sorted(responses.drug_id.astype(str).unique()))
        self.source_cell_lookup_ = {value: i for i, value in enumerate(self.source_cell_ids_)}
        self.source_drug_lookup_ = {value: i for i, value in enumerate(self.source_drug_ids_)}
        cell_rows = np.asarray([self.all_cell_lookup_[value] for value in self.source_cell_ids_])
        drug_rows = np.asarray([self.all_drug_lookup_[value] for value in self.source_drug_ids_])

        values = responses.sensitivity.to_numpy(dtype=float)
        self.global_mean_ = float(values.mean())
        matrix = np.full((len(self.source_cell_ids_), len(self.source_drug_ids_)), np.nan)
        ci = responses.cell_id.astype(str).map(self.source_cell_lookup_).to_numpy(dtype=int)
        di = responses.drug_id.astype(str).map(self.source_drug_lookup_).to_numpy(dtype=int)
        matrix[ci, di] = values - self.global_mean_
        observed = np.isfinite(matrix)

        rng = np.random.default_rng(self.seed)
        cell_factor = rng.normal(0, 0.05, size=(matrix.shape[0], self.rank))
        drug_factor = rng.normal(0, 0.05, size=(matrix.shape[1], self.rank))
        identity = np.eye(self.rank) * self.regularization
        for _ in range(self.iterations):
            for i in range(matrix.shape[0]):
                mask = observed[i]
                if mask.any():
                    design = drug_factor[mask]
                    cell_factor[i] = np.linalg.solve(
                        design.T @ design + identity,
                        design.T @ matrix[i, mask],
                    )
            for j in range(matrix.shape[1]):
                mask = observed[:, j]
                if mask.any():
                    design = cell_factor[mask]
                    drug_factor[j] = np.linalg.solve(
                        design.T @ design + identity,
                        design.T @ matrix[mask, j],
                    )

        self.cell_factor_ = cell_factor
        self.drug_factor_ = drug_factor
        self.cell_projection_ = Ridge(alpha=self.projection_alpha).fit(
            np.asarray(cell_features)[cell_rows], cell_factor
        )
        self.drug_projection_ = Ridge(alpha=self.projection_alpha).fit(
            np.asarray(drug_features)[drug_rows], drug_factor
        )
        self.all_cell_projected_ = self.cell_projection_.predict(cell_features)
        self.all_drug_projected_ = self.drug_projection_.predict(drug_features)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        cell_vectors = []
        drug_vectors = []
        for cell_id in frame.cell_id.astype(str):
            source = self.source_cell_lookup_.get(cell_id)
            if source is None:
                source = self.all_cell_lookup_.get(cell_id)
                if source is None:
                    raise ValueError(f"Cell lacks features: {cell_id}")
                cell_vectors.append(self.all_cell_projected_[source])
            else:
                cell_vectors.append(self.cell_factor_[source])
        for drug_id in frame.drug_id.astype(str):
            source = self.source_drug_lookup_.get(drug_id)
            if source is None:
                source = self.all_drug_lookup_.get(drug_id)
                if source is None:
                    raise ValueError(f"Drug lacks features: {drug_id}")
                drug_vectors.append(self.all_drug_projected_[source])
            else:
                drug_vectors.append(self.drug_factor_[source])
        return self.global_mean_ + np.sum(
            np.asarray(cell_vectors) * np.asarray(drug_vectors), axis=1
        )
