"""Bemis-Murcko scaffold cold-drug sensitivity for the locked MF configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:
    raise RuntimeError("RDKit is required for the prespecified scaffold sensitivity") from exc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEEDS = (104729, 130363, 155921, 181081, 205759, 231481, 257053, 282433, 308113, 333667)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def murcko(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return scaffold if scaffold else "ACYCLIC"


def scaffold_folds(source: pd.DataFrame, scaffold_by_drug: dict[str, str], seed: int, folds: int = 5):
    drug_edges = source.groupby("drug_id").size().to_dict()
    scaffold_drugs: dict[str, list[str]] = {}
    for drug, scaffold in scaffold_by_drug.items():
        scaffold_drugs.setdefault(scaffold, []).append(drug)
    rng = np.random.default_rng(seed)
    groups = []
    for scaffold, drugs in scaffold_drugs.items():
        weight = sum(drug_edges.get(drug, 0) for drug in drugs)
        groups.append((scaffold, sorted(drugs), weight, float(rng.random())))
    groups.sort(key=lambda item: (-item[2], item[3], item[0]))
    fold_groups = [[] for _ in range(folds)]
    fold_weight = np.zeros(folds, dtype=int)
    for group in groups:
        candidates = np.flatnonzero(fold_weight == fold_weight.min())
        fold = int(rng.choice(candidates))
        fold_groups[fold].append(group)
        fold_weight[fold] += group[2]
    for fold, assigned in enumerate(fold_groups):
        test_scaffolds = {item[0] for item in assigned}
        test_drugs = {drug for item in assigned for drug in item[1]}
        train_drugs = set(scaffold_by_drug) - test_drugs
        if test_scaffolds & {scaffold_by_drug[drug] for drug in train_drugs}:
            raise AssertionError("Scaffold leakage between train and test")
        yield fold, train_drugs, test_drugs, test_scaffolds


def per_drug(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    work = frame[["cell_id", "drug_id"]].copy()
    work["y_true"] = frame.sensitivity.to_numpy()
    work["y_pred"] = prediction
    rows = []
    for drug, group in work.groupby("drug_id", sort=True):
        value = np.nan
        if group.cell_id.nunique() >= 5 and group.y_true.nunique() >= 2 and group.y_pred.nunique() >= 2:
            value = float(spearmanr(group.y_true, group.y_pred).statistic)
        rows.append({"drug_id": drug, "n_cells": group.cell_id.nunique(), "spearman": value})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--drug-mapping", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    responses = pd.read_csv(args.responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = responses.loc[
        (responses.study == "CTRPv2")
        & responses.cell_id.isin(cell_ids)
        & responses.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    mapping = pd.read_csv(args.drug_mapping, dtype=str)
    source_drugs = set(source.drug_id)
    mapping = mapping.loc[
        (mapping.study == "CTRPv2") & mapping.drug_id.isin(source.drug_id),
        ["drug_id", "canSMILES"],
    ].dropna()
    conflicts = mapping.groupby("drug_id").canSMILES.nunique()
    if (conflicts > 1).any():
        raise RuntimeError(f"Conflicting CTRPv2 SMILES for {(conflicts > 1).sum()} drugs")
    mapping = mapping.drop_duplicates("drug_id")
    scaffold_by_drug, invalid = {}, []
    for row in mapping.itertuples(index=False):
        scaffold = murcko(row.canSMILES)
        if scaffold is None:
            invalid.append(row.drug_id)
        else:
            scaffold_by_drug[row.drug_id] = scaffold
    source = source.loc[source.drug_id.isin(scaffold_by_drug)].reset_index(drop=True)
    if source.drug_id.nunique() < 5:
        raise RuntimeError("Fewer than five drugs have valid structures")
    selected = pd.read_csv(args.selected_config).iloc[0]

    audit = pd.DataFrame([{
        "source_drugs_before_structure_filter": responses.loc[responses.study == "CTRPv2", "drug_id"].nunique(),
        "feature_complete_drugs": len(set(responses.loc[responses.study == "CTRPv2", "drug_id"]) & set(drug_ids)),
        "valid_structure_drugs": source.drug_id.nunique(),
        "invalid_structure_drugs": len(invalid),
        "missing_structure_drugs": len(source_drugs - set(mapping.drug_id)),
        "n_scaffolds": len(set(scaffold_by_drug.values())),
        "n_acyclic_drugs": sum(value == "ACYCLIC" for value in scaffold_by_drug.values()),
        "acyclic_policy": "all empty Murcko scaffolds grouped as ACYCLIC",
    }])
    atomic_csv(audit, args.output / "scaffold_audit.csv")

    for seed in SEEDS:
        metrics_path = args.output / f"metrics_seed_{seed}.csv"
        drugs_path = args.output / f"per_drug_seed_{seed}.csv"
        splits_path = args.output / f"splits_seed_{seed}.csv"
        assignments_path = args.output / f"assignments_seed_{seed}.csv"
        if all(path.is_file() and path.stat().st_size > 0 for path in (
            metrics_path, drugs_path, splits_path, assignments_path
        )):
            print(f"SKIP seed {seed}", flush=True)
            continue
        metric_tables, drug_tables, split_rows, assignment_rows = [], [], [], []
        for fold, train_drugs, test_drugs, test_scaffolds in scaffold_folds(source, scaffold_by_drug, seed):
            train = source.loc[source.drug_id.isin(train_drugs)].reset_index(drop=True)
            test = source.loc[source.drug_id.isin(test_drugs)].reset_index(drop=True)
            cell_reduced = fit_reducer(cell_matrix, cell_ids, set(train.cell_id), 64)[0]
            drug_reduced = fit_reducer(drug_matrix, drug_ids, set(train.drug_id), 64)[0]
            model = InductiveMatrixFactorization(
                rank=int(selected["rank"]), regularization=float(selected["regularization"]),
                iterations=int(selected["iterations"]), projection_alpha=float(selected["projection_alpha"]),
                seed=seed,
            ).fit(train, cell_reduced, cell_ids, drug_reduced, drug_ids)
            prediction = model.predict(test[["cell_id", "drug_id"]])
            evaluation = test[["cell_id", "drug_id"]].copy()
            evaluation["y_true"] = test.sensitivity.to_numpy()
            evaluation["y_pred"] = prediction
            metrics = evaluate_predictions(evaluation)
            metrics.insert(0, "fold", fold)
            metrics.insert(0, "seed", seed)
            metric_tables.append(metrics)
            drugs = per_drug(test, prediction)
            drugs.insert(0, "fold", fold)
            drugs.insert(0, "seed", seed)
            drug_tables.append(drugs)
            split_rows.append({
                "seed": seed, "fold": fold, "n_train_pairs": len(train), "n_test_pairs": len(test),
                "n_train_drugs": len(train_drugs), "n_test_drugs": len(test_drugs),
                "n_test_scaffolds": len(test_scaffolds),
            })
            assignment_rows.extend({
                "seed": seed, "fold": fold, "drug_id": drug,
                "murcko_scaffold": scaffold_by_drug[drug],
            } for drug in sorted(test_drugs))
        atomic_csv(pd.concat(metric_tables, ignore_index=True), metrics_path)
        atomic_csv(pd.concat(drug_tables, ignore_index=True), drugs_path)
        atomic_csv(pd.DataFrame(split_rows), splits_path)
        atomic_csv(pd.DataFrame(assignment_rows), assignments_path)
        print(f"END seed {seed}", flush=True)

    metric_files = sorted(args.output.glob("metrics_seed_*.csv"))
    drug_files = sorted(args.output.glob("per_drug_seed_*.csv"))
    split_files = sorted(args.output.glob("splits_seed_*.csv"))
    assignment_files = sorted(args.output.glob("assignments_seed_*.csv"))
    if not (len(metric_files) == len(drug_files) == len(split_files) == len(assignment_files) == len(SEEDS)):
        raise RuntimeError("Expected ten complete scaffold checkpoints")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    drugs = pd.concat([pd.read_csv(path, dtype={"drug_id": str}) for path in drug_files], ignore_index=True)
    splits = pd.concat([pd.read_csv(path) for path in split_files], ignore_index=True)
    assignments = pd.concat([
        pd.read_csv(path, dtype={"drug_id": str}) for path in assignment_files
    ], ignore_index=True)
    if assignments.duplicated(["seed", "drug_id"]).any():
        raise AssertionError("A drug appears in more than one test fold for a seed")
    if (assignments.groupby(["seed", "murcko_scaffold"]).fold.nunique() > 1).any():
        raise AssertionError("A scaffold appears in more than one test fold for a seed")
    per_seed = (
        drugs.dropna(subset=["spearman"]).groupby("seed", as_index=False)
        .agg(macro_spearman=("spearman", "mean"), n_eligible_drugs=("drug_id", "nunique"))
    )
    summary = pd.DataFrame([{
        "n_seeds": per_seed.seed.nunique(), "mean_macro_spearman": per_seed.macro_spearman.mean(),
        "seed_sd": per_seed.macro_spearman.std(ddof=1), "seed_min": per_seed.macro_spearman.min(),
        "seed_max": per_seed.macro_spearman.max(),
        "mean_eligible_drugs": per_seed.n_eligible_drugs.mean(),
    }])
    atomic_csv(metrics, args.output / "all_metrics.csv")
    atomic_csv(drugs, args.output / "all_per_drug_scores.csv")
    atomic_csv(splits, args.output / "all_splits.csv")
    atomic_csv(assignments, args.output / "all_assignments.csv")
    atomic_csv(per_seed, args.output / "per_seed_summary.csv")
    atomic_csv(summary, args.output / "scaffold_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
