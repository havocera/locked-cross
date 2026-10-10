"""Deterministic, leakage-aware split generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    regime: str
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray


def _balanced_group_folds(values: pd.Series, folds: int, seed: int) -> np.ndarray:
    groups = np.asarray(sorted(values.astype(str).unique()))
    if len(groups) < folds:
        raise ValueError(f"Need at least {folds} groups, found {len(groups)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    assignment = {group: index % folds for index, group in enumerate(groups)}
    return values.astype(str).map(assignment).to_numpy(dtype=int)


def make_folds(
    frame: pd.DataFrame,
    regime: str,
    folds: int = 5,
    seed: int = 104729,
) -> list[Fold]:
    """Create random-pair or entity-disjoint folds for one study."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if regime not in {"random_pair", "cold_cell", "cold_drug", "cold_both"}:
        raise ValueError(f"Unknown regime: {regime}")

    rng = np.random.default_rng(seed)
    if regime == "random_pair":
        order = rng.permutation(len(frame))
        labels = np.empty(len(frame), dtype=int)
        labels[order] = np.arange(len(frame)) % folds
        cell_labels = drug_labels = None
    elif regime == "cold_cell":
        labels = _balanced_group_folds(frame["cell_id"], folds, seed)
        cell_labels = drug_labels = None
    elif regime == "cold_drug":
        labels = _balanced_group_folds(frame["drug_id"], folds, seed)
        cell_labels = drug_labels = None
    else:
        labels = None
        cell_labels = _balanced_group_folds(frame["cell_id"], folds, seed)
        drug_labels = _balanced_group_folds(frame["drug_id"], folds, seed + 1)

    output: list[Fold] = []
    all_rows = np.arange(len(frame))
    for fold in range(folds):
        if regime == "cold_both":
            test_mask = (cell_labels == fold) & (drug_labels == fold)
            train_mask = (cell_labels != fold) & (drug_labels != fold)
        else:
            test_mask = labels == fold
            train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Empty train or test partition for {regime} fold {fold}")
        output.append(
            Fold(
                regime=regime,
                fold=fold,
                train_index=all_rows[train_mask],
                test_index=all_rows[test_mask],
            )
        )
    return output
