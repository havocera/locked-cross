"""Inductive training-edge neighborhood smoother (ITENS)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def cosine_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic top-k reference indices, similarities, and kernel weights."""
    if k < 1 or temperature <= 0:
        raise ValueError("k and temperature must be positive")
    qnorm = np.linalg.norm(query, axis=1, keepdims=True)
    rnorm = np.linalg.norm(reference, axis=1, keepdims=True)
    q = np.divide(query, qnorm, out=np.zeros_like(query), where=qnorm > 0)
    r = np.divide(reference, rnorm, out=np.zeros_like(reference), where=rnorm > 0)
    similarity = q @ r.T
    k = min(k, reference.shape[0])
    # Stable full sort makes entity-index order the deterministic tie-breaker.
    index = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(similarity, index, axis=1)
    weight = np.exp(np.clip((selected - 1.0) / temperature, -80.0, 0.0))
    return index.astype(np.int32), selected.astype(np.float32), weight.astype(np.float64)


class ITENSRegressor:
    """Continuous graph-neighborhood predictor using labeled training edges only."""

    def __init__(
        self,
        k_cell: int = 16,
        k_drug: int = 16,
        temperature_cell: float = 0.1,
        temperature_drug: float = 0.1,
        ess_shrinkage: float = 10.0,
        theta: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        if ess_shrinkage < 0:
            raise ValueError("ess_shrinkage must be nonnegative")
        if min(theta) < 0 or sum(theta) > 1 + 1e-12:
            raise ValueError("theta must be nonnegative and sum to at most one")
        self.k_cell = k_cell
        self.k_drug = k_drug
        self.temperature_cell = temperature_cell
        self.temperature_drug = temperature_drug
        self.ess_shrinkage = ess_shrinkage
        self.theta = np.asarray(theta, dtype=float)

    def fit(
        self,
        responses: pd.DataFrame,
        cell_embedding: np.ndarray,
        cell_ids: np.ndarray,
        drug_embedding: np.ndarray,
        drug_ids: np.ndarray,
    ) -> "ITENSRegressor":
        required = {"cell_id", "drug_id", "sensitivity"}
        if not required.issubset(responses.columns):
            raise ValueError(f"Training responses need {sorted(required)}")
        if responses.duplicated(["cell_id", "drug_id"]).any():
            raise ValueError("Training response edges must be unique")
        self.all_cell_ids_ = np.asarray(cell_ids, dtype=str)
        self.all_drug_ids_ = np.asarray(drug_ids, dtype=str)
        self.cell_embedding_ = np.asarray(cell_embedding, dtype=np.float64)
        self.drug_embedding_ = np.asarray(drug_embedding, dtype=np.float64)
        self.all_cell_lookup_ = {value: i for i, value in enumerate(self.all_cell_ids_)}
        self.all_drug_lookup_ = {value: i for i, value in enumerate(self.all_drug_ids_)}

        self.source_cell_ids_ = np.asarray(sorted(responses.cell_id.astype(str).unique()))
        self.source_drug_ids_ = np.asarray(sorted(responses.drug_id.astype(str).unique()))
        self.source_cell_lookup_ = {value: i for i, value in enumerate(self.source_cell_ids_)}
        self.source_drug_lookup_ = {value: i for i, value in enumerate(self.source_drug_ids_)}
        try:
            cell_rows = [self.all_cell_lookup_[value] for value in self.source_cell_ids_]
            drug_rows = [self.all_drug_lookup_[value] for value in self.source_drug_ids_]
        except KeyError as error:
            raise ValueError(f"Training entity lacks features: {error}") from error
        self.source_cell_embedding_ = self.cell_embedding_[cell_rows]
        self.source_drug_embedding_ = self.drug_embedding_[drug_rows]

        self.response_ = np.full(
            (len(self.source_cell_ids_), len(self.source_drug_ids_)), np.nan, dtype=float
        )
        cell_index = responses.cell_id.astype(str).map(self.source_cell_lookup_).to_numpy()
        drug_index = responses.drug_id.astype(str).map(self.source_drug_lookup_).to_numpy()
        values = pd.to_numeric(responses.sensitivity, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Training sensitivities must be finite")
        self.response_[cell_index, drug_index] = values
        self.observed_ = np.isfinite(self.response_)
        self.filled_response_ = np.nan_to_num(self.response_, nan=0.0)
        self.global_mean_ = float(values.mean())
        self.training_edges_ = set(zip(responses.cell_id.astype(str), responses.drug_id.astype(str)))
        return self

    def _query_neighbors(self, ids: np.ndarray, entity: str):
        if entity == "cell":
            all_lookup = self.all_cell_lookup_
            all_embedding = self.cell_embedding_
            source_embedding = self.source_cell_embedding_
            k, temperature = self.k_cell, self.temperature_cell
        else:
            all_lookup = self.all_drug_lookup_
            all_embedding = self.drug_embedding_
            source_embedding = self.source_drug_embedding_
            k, temperature = self.k_drug, self.temperature_drug
        try:
            rows = [all_lookup[value] for value in ids]
        except KeyError as error:
            raise ValueError(f"Query {entity} lacks features: {error}") from error
        return cosine_neighbors(all_embedding[rows], source_embedding, k, temperature)

    def _reliability(self, denominator: np.ndarray, squared_sum: np.ndarray) -> np.ndarray:
        ess = np.divide(
            denominator**2,
            squared_sum,
            out=np.zeros_like(denominator, dtype=float),
            where=squared_sum > 0,
        )
        return np.divide(
            ess,
            ess + self.ess_shrinkage,
            out=np.zeros_like(ess),
            where=(ess + self.ess_shrinkage) > 0,
        )

    def predict(self, frame: pd.DataFrame, return_diagnostics: bool = False):
        """Predict from entity IDs only; response columns are ignored if present."""
        query_cells = np.asarray(sorted(frame.cell_id.astype(str).unique()))
        query_drugs = np.asarray(sorted(frame.drug_id.astype(str).unique()))
        cell_query_lookup = {value: i for i, value in enumerate(query_cells)}
        drug_query_lookup = {value: i for i, value in enumerate(query_drugs)}
        ci, cs, cw = self._query_neighbors(query_cells, "cell")
        di, ds, dw = self._query_neighbors(query_drugs, "drug")

        cell_weight = np.zeros((len(query_cells), len(self.source_cell_ids_)), dtype=float)
        drug_weight = np.zeros((len(query_drugs), len(self.source_drug_ids_)), dtype=float)
        np.put_along_axis(cell_weight, ci, cw, axis=1)
        np.put_along_axis(drug_weight, di, dw, axis=1)

        observed = self.observed_.astype(float)
        filled = self.filled_response_
        pc_num = cell_weight @ filled
        pc_den = cell_weight @ observed
        pc_sq = (cell_weight**2) @ observed
        pc_all = np.divide(pc_num, pc_den, out=np.full_like(pc_num, self.global_mean_), where=pc_den > 0)
        pc_rho_all = self._reliability(pc_den, pc_sq)

        pd_num = filled @ drug_weight.T
        pd_den = observed @ drug_weight.T
        pd_sq = observed @ (drug_weight**2).T
        pd_all = np.divide(pd_num, pd_den, out=np.full_like(pd_num, self.global_mean_), where=pd_den > 0)
        pd_rho_all = self._reliability(pd_den, pd_sq)

        pcd_num = cell_weight @ filled @ drug_weight.T
        pcd_den = cell_weight @ observed @ drug_weight.T
        pcd_sq = (cell_weight**2) @ observed @ (drug_weight**2).T
        pcd_all = np.divide(
            pcd_num, pcd_den, out=np.full_like(pcd_num, self.global_mean_), where=pcd_den > 0
        )
        pcd_rho_all = self._reliability(pcd_den, pcd_sq)

        qci = frame.cell_id.astype(str).map(cell_query_lookup).to_numpy(dtype=int)
        qdi = frame.drug_id.astype(str).map(drug_query_lookup).to_numpy(dtype=int)
        source_ci = frame.cell_id.astype(str).map(self.source_cell_lookup_)
        source_di = frame.drug_id.astype(str).map(self.source_drug_lookup_)

        pc = np.full(len(frame), self.global_mean_)
        pc_rho = np.zeros(len(frame))
        valid_drug = source_di.notna().to_numpy()
        pc[valid_drug] = pc_all[qci[valid_drug], source_di[valid_drug].astype(int)]
        pc_rho[valid_drug] = pc_rho_all[qci[valid_drug], source_di[valid_drug].astype(int)]

        pd_value = np.full(len(frame), self.global_mean_)
        pd_rho = np.zeros(len(frame))
        valid_cell = source_ci.notna().to_numpy()
        pd_value[valid_cell] = pd_all[source_ci[valid_cell].astype(int), qdi[valid_cell]]
        pd_rho[valid_cell] = pd_rho_all[source_ci[valid_cell].astype(int), qdi[valid_cell]]

        pcd = pcd_all[qci, qdi]
        pcd_rho = pcd_rho_all[qci, qdi]
        expert = np.column_stack([pc, pd_value, pcd])
        reliability = np.column_stack([pc_rho, pd_rho, pcd_rho])
        prediction = self.global_mean_ + np.sum(
            self.theta * reliability * (expert - self.global_mean_), axis=1
        )
        if not np.isfinite(prediction).all():
            raise RuntimeError("ITENS produced non-finite predictions")
        if not return_diagnostics:
            return prediction
        diagnostics = pd.DataFrame(
            {
                "cell_expert": pc,
                "drug_expert": pd_value,
                "product_expert": pcd,
                "cell_reliability": pc_rho,
                "drug_reliability": pd_rho,
                "product_reliability": pcd_rho,
                "cell_nearest_similarity": cs[qci, 0],
                "drug_nearest_similarity": ds[qdi, 0],
            }
        )
        return prediction, diagnostics

    def explain_one(
        self,
        cell_id: str,
        drug_id: str,
        excluded_edges: set[tuple[str, str]] | None = None,
    ) -> tuple[float, pd.DataFrame]:
        """Return exact signed contributions of labeled training edges."""
        excluded_edges = excluded_edges or set()
        ci, cs, cw = self._query_neighbors(np.asarray([str(cell_id)]), "cell")
        di, ds, dw = self._query_neighbors(np.asarray([str(drug_id)]), "drug")
        ci, cs, cw = ci[0], cs[0], cw[0]
        di, ds, dw = di[0], ds[0], dw[0]
        contributions: dict[tuple[int, int], dict[str, float]] = defaultdict(
            lambda: {"cell_component": 0.0, "drug_component": 0.0, "product_component": 0.0}
        )

        def add_expert(edges, weights, theta, component):
            if not edges:
                return
            weights = np.asarray(weights, dtype=float)
            denominator = weights.sum()
            squared = np.square(weights).sum()
            ess = denominator**2 / squared if squared > 0 else 0.0
            reliability = ess / (ess + self.ess_shrinkage) if ess + self.ess_shrinkage > 0 else 0.0
            normalized = weights / denominator
            for edge, weight in zip(edges, normalized):
                y = self.response_[edge]
                contributions[edge][component] += theta * reliability * weight * (y - self.global_mean_)

        source_drug = self.source_drug_lookup_.get(str(drug_id))
        if source_drug is not None:
            keep = [
                self.observed_[c, source_drug]
                and (self.source_cell_ids_[c], self.source_drug_ids_[source_drug]) not in excluded_edges
                for c in ci
            ]
            edges = [(int(c), source_drug) for c, use in zip(ci, keep) if use]
            weights = [weight for weight, use in zip(cw, keep) if use]
            add_expert(edges, weights, self.theta[0], "cell_component")

        source_cell = self.source_cell_lookup_.get(str(cell_id))
        if source_cell is not None:
            keep = [
                self.observed_[source_cell, d]
                and (self.source_cell_ids_[source_cell], self.source_drug_ids_[d]) not in excluded_edges
                for d in di
            ]
            edges = [(source_cell, int(d)) for d, use in zip(di, keep) if use]
            weights = [weight for weight, use in zip(dw, keep) if use]
            add_expert(edges, weights, self.theta[1], "drug_component")

        edges, weights = [], []
        for cpos, c in enumerate(ci):
            for dpos, d in enumerate(di):
                edge_id = (self.source_cell_ids_[c], self.source_drug_ids_[d])
                if self.observed_[c, d] and edge_id not in excluded_edges:
                    edges.append((int(c), int(d)))
                    weights.append(cw[cpos] * dw[dpos])
        add_expert(edges, weights, self.theta[2], "product_component")

        rows = []
        for (c, d), components in contributions.items():
            signed = sum(components.values())
            rows.append(
                {
                    "neighbor_cell_id": self.source_cell_ids_[c],
                    "neighbor_drug_id": self.source_drug_ids_[d],
                    **components,
                    "signed_attribution": signed,
                    "absolute_attribution": abs(signed),
                }
            )
        result = pd.DataFrame(rows)
        if result.empty:
            prediction = self.global_mean_
            return prediction, result
        result = result.sort_values(
            ["absolute_attribution", "neighbor_cell_id", "neighbor_drug_id"],
            ascending=[False, True, True],
            ignore_index=True,
        )
        result["attribution_rank"] = np.arange(1, len(result) + 1)
        prediction = self.global_mean_ + result.signed_attribution.sum()
        return float(prediction), result
