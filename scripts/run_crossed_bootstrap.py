"""Crossed cell-and-drug bootstrap for shared-entity confirmation effects.

The sensitivity estimand uses seed-ensemble row predictions. Cells are sampled once
per replicate across both arms, drugs are sampled with replacement, and each effect is
computed only from paired drugs with valid correlations in both arms.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


SEED = 20_261_004


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_ensemble(paths: list[Path], regime: str) -> pd.DataFrame:
    tables = []
    for path in paths:
        table = pd.read_csv(path, dtype={"cell_id": str, "drug_id": str})
        if "regime" in table:
            table = table.loc[table.regime == regime]
        tables.append(table[["seed", "test_study", "cell_id", "drug_id", "y_true", "y_pred"]])
    rows = pd.concat(tables, ignore_index=True)
    grouped = rows.groupby(["test_study", "cell_id", "drug_id"], as_index=False).agg(
        y_true_min=("y_true", "min"), y_true_max=("y_true", "max"),
        y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), n_seeds=("seed", "nunique"),
    )
    if not np.allclose(grouped.y_true_min, grouped.y_true_max):
        raise AssertionError("Observed responses differ across seed files")
    if not (grouped.n_seeds == 10).all():
        raise AssertionError("Every ensemble row must contain all ten seeds")
    return grouped[["test_study", "cell_id", "drug_id", "y_true", "y_pred"]]


def valid_drugs(frame: pd.DataFrame) -> set[str]:
    valid = set()
    for drug, group in frame.groupby("drug_id", sort=False):
        if group.cell_id.nunique() >= 5 and group.y_true.nunique() >= 2 and group.y_pred.nunique() >= 2:
            valid.add(drug)
    return valid


@dataclass(frozen=True)
class RankPlan:
    order: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True)
class DrugGroup:
    cell_index: np.ndarray
    observed: RankPlan
    predicted: RankPlan


def rank_plan(values: np.ndarray) -> RankPlan:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(values)])
    return RankPlan(order=order, starts=starts, sizes=sizes)


def grouped_rows(frame: pd.DataFrame, drugs: set[str], cell_to_index: dict[str, int]):
    output = {}
    for drug, group in frame.loc[frame.drug_id.isin(drugs)].groupby("drug_id", sort=False):
        observed = group.y_true.to_numpy(dtype=float)
        predicted = group.y_pred.to_numpy(dtype=float)
        output[drug] = DrugGroup(
            cell_index=np.fromiter((cell_to_index[cell] for cell in group.cell_id), dtype=int),
            observed=rank_plan(observed), predicted=rank_plan(predicted),
        )
    return output


def weighted_ranks(plan: RankPlan, weights: np.ndarray) -> np.ndarray:
    sorted_weights = weights[plan.order]
    group_weights = np.add.reduceat(sorted_weights, plan.starts)
    ranks_by_group = np.cumsum(group_weights) - group_weights + (group_weights + 1.0) / 2.0
    sorted_ranks = np.repeat(ranks_by_group, plan.sizes)
    ranks = np.empty(len(weights), dtype=float)
    ranks[plan.order] = sorted_ranks
    return ranks


def correlation(group: DrugGroup, cell_counts: np.ndarray | None = None) -> float:
    weights = np.ones(len(group.cell_index), dtype=int) if cell_counts is None else cell_counts[group.cell_index]
    keep = weights > 0
    total = int(weights.sum())
    if keep.sum() < 2 or total < 5:
        return np.nan
    observed_ranks = weighted_ranks(group.observed, weights)
    predicted_ranks = weighted_ranks(group.predicted, weights)
    mean_observed = np.dot(weights, observed_ranks) / total
    mean_predicted = np.dot(weights, predicted_ranks) / total
    centered_observed = observed_ranks - mean_observed
    centered_predicted = predicted_ranks - mean_predicted
    variance_observed = np.dot(weights, centered_observed * centered_observed)
    variance_predicted = np.dot(weights, centered_predicted * centered_predicted)
    if variance_observed <= 0 or variance_predicted <= 0:
        return np.nan
    covariance = np.dot(weights, centered_observed * centered_predicted)
    return float(covariance / np.sqrt(variance_observed * variance_predicted))


def paired_effect(first, second, drug_counts: Counter, cell_counts=None) -> float:
    total, weight = 0.0, 0
    for drug, multiplicity in drug_counts.items():
        a = correlation(first[drug], cell_counts)
        b = correlation(second[drug], cell_counts)
        if np.isfinite(a) and np.isfinite(b):
            total += multiplicity * (a - b)
            weight += multiplicity
    return total / weight if weight else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-directory", type=Path, required=True)
    parser.add_argument("--external-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=9_999)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    internal_paths = sorted(args.internal_directory.glob("matched_internal_rows_seed_*.csv.gz"))
    external_paths = sorted(args.external_directory.glob("matched_rows_seed_*.csv.gz"))
    if len(internal_paths) != 10 or len(external_paths) != 10:
        raise RuntimeError("Crossed bootstrap requires ten internal and ten external row files")
    internal = read_ensemble(internal_paths, "internal_random_overlap")
    external_mf = read_ensemble(external_paths, "cross_study_overlap")
    itens = pd.read_csv(
        args.external_directory / "itens_matched_rows.csv.gz",
        dtype={"cell_id": str, "drug_id": str},
    )
    if itens.duplicated(["test_study", "cell_id", "drug_id"]).any():
        raise AssertionError("ITENS matched rows are not unique")

    summaries, distributions = [], []
    for study_index, study in enumerate(("GDSCv1", "NCI60")):
        study_internal = internal.loc[internal.test_study == study].reset_index(drop=True)
        study_mf = external_mf.loc[external_mf.test_study == study].reset_index(drop=True)
        study_itens = itens.loc[itens.test_study == study].reset_index(drop=True)
        comparisons = [
            ("C1_internal_minus_external_MF", study_internal, study_mf),
            ("C2_ITENS_minus_external_MF", study_itens, study_mf),
        ]
        for comparison_index, (comparison, first_frame, second_frame) in enumerate(comparisons):
            cells = np.asarray(sorted(set(first_frame.cell_id) & set(second_frame.cell_id)))
            if len(cells) < 2:
                raise RuntimeError(f"Too few shared cells for {study} {comparison}")
            drugs = sorted(valid_drugs(first_frame) & valid_drugs(second_frame))
            if len(drugs) < 2:
                raise RuntimeError(f"Too few paired eligible drugs for {study} {comparison}")
            drug_set = set(drugs)
            cell_to_index = {cell: index for index, cell in enumerate(cells)}
            first = grouped_rows(first_frame, drug_set, cell_to_index)
            second = grouped_rows(second_frame, drug_set, cell_to_index)
            observed = paired_effect(first, second, Counter({drug: 1 for drug in drugs}))
            checkpoint = args.output / f"distribution_{study}_{comparison}.csv"
            if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                checkpoint_table = pd.read_csv(checkpoint)
                if len(checkpoint_table) != args.resamples:
                    raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
                values = checkpoint_table.effect.to_numpy(dtype=float)
                print(f"SKIP {study} {comparison}", flush=True)
            else:
                print(f"START {study} {comparison}", flush=True)
                rng = np.random.default_rng(np.random.SeedSequence([SEED, study_index, comparison_index]))
                values = np.empty(args.resamples, dtype=float)
                for replicate in range(args.resamples):
                    cell_counts = rng.multinomial(len(cells), np.full(len(cells), 1.0 / len(cells)))
                    sampled_drugs = rng.choice(drugs, size=len(drugs), replace=True)
                    values[replicate] = paired_effect(first, second, Counter(sampled_drugs), cell_counts)
                atomic_csv(pd.DataFrame({"replicate": np.arange(args.resamples), "effect": values}), checkpoint)
                print(f"END {study} {comparison}", flush=True)
            valid = values[np.isfinite(values)]
            if len(valid) < 0.9 * args.resamples:
                raise RuntimeError(f"More than 10% invalid crossed-bootstrap replicates for {study} {comparison}")
            low, high = np.quantile(valid, [0.025, 0.975])
            summaries.append({
                "test_study": study, "comparison": comparison, "n_paired_drugs": len(drugs),
                "n_cells_in_sampling_frame": len(cells), "observed_effect": observed,
                "ci_low": float(low), "ci_high": float(high), "valid_resamples": len(valid),
                "requested_resamples": args.resamples, "seed": SEED,
                "interval": "crossed cell-and-drug percentile bootstrap",
                "seed_aggregation": "mean row prediction across ten seeds",
            })
            distributions.extend({
                "test_study": study, "comparison": comparison,
                "replicate": index, "effect": value,
            } for index, value in enumerate(values))
    summary = pd.DataFrame(summaries)
    atomic_csv(summary, args.output / "crossed_bootstrap_summary.csv")
    atomic_csv(pd.DataFrame(distributions), args.output / "crossed_bootstrap_distribution.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
