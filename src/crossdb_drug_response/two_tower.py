"""Bounded inductive two-tower neural drug-response regressor."""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn


class _Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.bias = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network(values), self.bias(values).squeeze(-1)


class _TwoTowerNetwork(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int, hidden_dim: int, latent_dim: int, dropout: float, intercept: float):
        super().__init__()
        self.cell_tower = _Tower(cell_dim, hidden_dim, latent_dim, dropout)
        self.drug_tower = _Tower(drug_dim, hidden_dim, latent_dim, dropout)
        self.intercept = nn.Parameter(torch.tensor(intercept, dtype=torch.float32))
        self.scale = math.sqrt(latent_dim)

    def forward(self, cell: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        cell_latent, cell_bias = self.cell_tower(cell)
        drug_latent, drug_bias = self.drug_tower(drug)
        logit = self.intercept + cell_bias + drug_bias + (cell_latent * drug_latent).sum(dim=1) / self.scale
        return -torch.sigmoid(logit)


class TwoTowerRegressor:
    """Feature-only inductive neural model with predictions bounded to [-1, 0]."""

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        batch_size: int = 8192,
        drug_balanced: bool = True,
        seed: int = 104729,
        threads: int = 8,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.drug_balanced = drug_balanced
        self.seed = seed
        self.threads = threads

    def fit(
        self,
        responses: pd.DataFrame,
        cell_embedding: np.ndarray,
        cell_ids: np.ndarray,
        drug_embedding: np.ndarray,
        drug_ids: np.ndarray,
    ) -> "TwoTowerRegressor":
        required = {"cell_id", "drug_id", "sensitivity"}
        if not required.issubset(responses):
            raise ValueError(f"Training responses need {sorted(required)}")
        if responses.duplicated(["cell_id", "drug_id"]).any():
            raise ValueError("Training response edges must be unique")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(True)

        self.cell_ids_ = np.asarray(cell_ids, dtype=str)
        self.drug_ids_ = np.asarray(drug_ids, dtype=str)
        self.cell_lookup_ = {value: index for index, value in enumerate(self.cell_ids_)}
        self.drug_lookup_ = {value: index for index, value in enumerate(self.drug_ids_)}
        self.cell_embedding_ = np.ascontiguousarray(cell_embedding, dtype=np.float32)
        self.drug_embedding_ = np.ascontiguousarray(drug_embedding, dtype=np.float32)
        try:
            cell_index = np.fromiter((self.cell_lookup_[value] for value in responses.cell_id.astype(str)), dtype=np.int64)
            drug_index = np.fromiter((self.drug_lookup_[value] for value in responses.drug_id.astype(str)), dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Training entity lacks features: {exc}") from exc
        outcome = responses.sensitivity.to_numpy(dtype=np.float32)
        if not np.isfinite(outcome).all() or np.any(outcome < -1.0001) or np.any(outcome > 0.0001):
            raise ValueError("Two-tower bounded output requires finite sensitivity in [-1, 0]")
        probability = float(np.clip(-outcome.mean(), 1e-5, 1 - 1e-5))
        intercept = math.log(probability / (1 - probability))
        self.network_ = _TwoTowerNetwork(
            self.cell_embedding_.shape[1], self.drug_embedding_.shape[1],
            self.hidden_dim, self.latent_dim, self.dropout, intercept,
        )
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        cell_features = torch.from_numpy(self.cell_embedding_)
        drug_features = torch.from_numpy(self.drug_embedding_)
        cell_index_tensor = torch.from_numpy(cell_index)
        drug_index_tensor = torch.from_numpy(drug_index)
        outcome_tensor = torch.from_numpy(outcome)
        if self.drug_balanced:
            counts = responses.drug_id.astype(str).value_counts()
            pair_weight = responses.drug_id.astype(str).map(
                len(responses) / (len(counts) * counts)
            ).to_numpy(dtype=np.float32)
        else:
            pair_weight = np.ones(len(responses), dtype=np.float32)
        pair_weight_tensor = torch.from_numpy(pair_weight)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history_ = []
        self.network_.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(outcome), generator=generator)
            total_loss = 0.0
            for start in range(0, len(outcome), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                prediction = self.network_(
                    cell_features[cell_index_tensor[batch]], drug_features[drug_index_tensor[batch]]
                )
                batch_weight = pair_weight_tensor[batch]
                loss = torch.sum(batch_weight * (prediction - outcome_tensor[batch]) ** 2) / torch.sum(batch_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            self.loss_history_.append(total_loss / len(outcome))
        self.network_.eval()
        return self

    def predict(self, frame: pd.DataFrame, batch_size: int = 100_000) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("Model is not fitted")
        try:
            cell_index = np.fromiter((self.cell_lookup_[value] for value in frame.cell_id.astype(str)), dtype=np.int64)
            drug_index = np.fromiter((self.drug_lookup_[value] for value in frame.drug_id.astype(str)), dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Query entity lacks features: {exc}") from exc
        cell_features = torch.from_numpy(self.cell_embedding_)
        drug_features = torch.from_numpy(self.drug_embedding_)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                prediction = self.network_(
                    cell_features[torch.from_numpy(cell_index[start:stop])],
                    drug_features[torch.from_numpy(drug_index[start:stop])],
                )
                outputs.append(prediction.numpy())
        result = np.concatenate(outputs).astype(np.float64)
        if not np.isfinite(result).all() or np.any(result < -1) or np.any(result > 0):
            raise RuntimeError("Two-tower prediction violated bounded-output contract")
        return result

    @property
    def parameter_count_(self) -> int:
        return sum(parameter.numel() for parameter in self.network_.parameters())
