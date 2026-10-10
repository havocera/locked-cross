"""Source-only selection and locked external evaluation of a bounded two-tower model."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from crossdb_drug_response.splits import make_folds  # noqa: E402
from crossdb_drug_response.two_tower import TwoTowerRegressor  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEEDS = (104729, 130363, 155921, 181081, 205759, 231481, 257053, 282433, 308113, 333667)
REGIMES = ("cold_cell", "cold_drug", "cold_both")
TUNING_SEED = SEEDS[0]
CONFIGS = (
    {"hidden_dim": 64, "latent_dim": 16, "dropout": 0.0, "weight_decay": 1e-4},
    {"hidden_dim": 64, "latent_dim": 32, "dropout": 0.0, "weight_decay": 1e-4},
    {"hidden_dim": 128, "latent_dim": 32, "dropout": 0.0, "weight_decay": 1e-4},
    {"hidden_dim": 64, "latent_dim": 32, "dropout": 0.1, "weight_decay": 1e-4},
    {"hidden_dim": 64, "latent_dim": 32, "dropout": 0.0, "weight_decay": 1e-3},
)


def config_id(config: dict) -> str:
    return f"h{config['hidden_dim']}_z{config['latent_dim']}_d{config['dropout']:g}_wd{config['weight_decay']:g}"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def fitted_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    evaluation = frame[["cell_id", "drug_id"]].copy()
    evaluation["y_true"] = frame.sensitivity.to_numpy()
    evaluation["y_pred"] = prediction
    return evaluate_predictions(evaluation)


def model_from(config: dict, seed: int) -> TwoTowerRegressor:
    return TwoTowerRegressor(
        **config, learning_rate=1e-3, epochs=15, batch_size=8192,
        drug_balanced=True, seed=seed, threads=8
    )


def select_config(tuning_metrics: pd.DataFrame) -> pd.DataFrame:
    macro = tuning_metrics.loc[tuning_metrics.metric == "spearman_drug_macro_mean"]
    mae = tuning_metrics.loc[tuning_metrics.metric == "mae"]
    macro_summary = macro.groupby("config_id", as_index=False).agg(
        selection_score=("value", "mean"), valid_folds=("value", "count"),
        parameter_count=("parameter_count", "first"),
    )
    mae_summary = mae.groupby("config_id", as_index=False).agg(mean_mae=("value", "mean"))
    summary = macro_summary.merge(mae_summary, on="config_id", validate="one_to_one")
    expected = len(REGIMES) * 5
    summary.loc[summary.valid_folds < expected, "selection_score"] = np.nan
    summary = summary.sort_values(
        ["selection_score", "mean_mae", "parameter_count", "config_id"],
        ascending=[False, True, True, True], na_position="last", ignore_index=True,
    )
    summary["selected"] = False
    if summary.selection_score.notna().any():
        summary.loc[summary.selection_score.first_valid_index(), "selected"] = True
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-responses", type=Path, required=True)
    parser.add_argument("--holdout-responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tuning_dir = args.output / "tuning_checkpoints"
    evaluation_dir = args.output / "evaluation_checkpoints"
    external_dir = args.output / "external_checkpoints"
    for directory in (tuning_dir, evaluation_dir, external_dir):
        directory.mkdir(exist_ok=True)

    development = pd.read_csv(args.development_responses, dtype={"cell_id": str, "drug_id": str})
    holdout = pd.read_csv(args.holdout_responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = development.loc[
        (development.study == "CTRPv2") & development.cell_id.isin(cell_ids) & development.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    holdout = holdout.loc[holdout.cell_id.isin(cell_ids) & holdout.drug_id.isin(drug_ids)].reset_index(drop=True)

    # Stage 1: configuration selection on one prespecified source-only split seed.
    for regime in REGIMES:
        for split in make_folds(source, regime, folds=5, seed=TUNING_SEED):
            train = source.iloc[split.train_index].reset_index(drop=True)
            test = source.iloc[split.test_index].reset_index(drop=True)
            cell_reduced = fit_reducer(cell_matrix, cell_ids, set(train.cell_id), 64)[0]
            drug_reduced = fit_reducer(drug_matrix, drug_ids, set(train.drug_id), 64)[0]
            for config in CONFIGS:
                identifier = config_id(config)
                path = tuning_dir / f"{regime}_fold{split.fold}_{identifier}.csv"
                if path.is_file() and path.stat().st_size > 0:
                    continue
                print(f"TUNE {regime} fold={split.fold} {identifier}", flush=True)
                model = model_from(config, TUNING_SEED).fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                )
                metrics = fitted_metrics(test, model.predict(test[["cell_id", "drug_id"]]))
                for column, value in (
                    ("parameter_count", model.parameter_count_), ("config_id", identifier),
                    ("fold", split.fold), ("regime", regime), ("seed", TUNING_SEED),
                ):
                    metrics.insert(0, column, value)
                atomic_csv(metrics, path)
    tuning_files = sorted(tuning_dir.glob("*.csv"))
    if len(tuning_files) != len(REGIMES) * 5 * len(CONFIGS):
        raise RuntimeError("Incomplete two-tower tuning grid")
    tuning_metrics = pd.concat([pd.read_csv(path) for path in tuning_files], ignore_index=True)
    selection = select_config(tuning_metrics)
    if selection.selected.sum() != 1:
        raise RuntimeError("Two-tower selection did not produce exactly one configuration")
    selected_id = selection.loc[selection.selected, "config_id"].iloc[0]
    selected_config = next(config for config in CONFIGS if config_id(config) == selected_id)
    atomic_csv(tuning_metrics, args.output / "tuning_metrics.csv")
    atomic_csv(selection, args.output / "tuning_summary.csv")
    (args.output / "selected_config.json").write_text(
        json.dumps({
            "config_id": selected_id, **selected_config, "learning_rate": 1e-3,
            "epochs": 15, "batch_size": 8192, "selection_seed": TUNING_SEED,
            "loss": "drug-balanced MSE",
            "selection_regimes": list(REGIMES), "selection_rule": "mean macro Spearman; MAE, parameters, lexical tie-breaks",
        }, indent=2) + "\n", encoding="utf-8",
    )
    print(selection.to_string(index=False), flush=True)

    # Stage 2: descriptive stability on all prespecified source-only seeds.
    for seed in SEEDS:
        path = evaluation_dir / f"metrics_seed_{seed}.csv"
        if path.is_file() and path.stat().st_size > 0:
            continue
        tables = []
        for regime in REGIMES:
            for split in make_folds(source, regime, folds=5, seed=seed):
                train = source.iloc[split.train_index].reset_index(drop=True)
                test = source.iloc[split.test_index].reset_index(drop=True)
                cell_reduced = fit_reducer(cell_matrix, cell_ids, set(train.cell_id), 64)[0]
                drug_reduced = fit_reducer(drug_matrix, drug_ids, set(train.drug_id), 64)[0]
                model = model_from(selected_config, seed).fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                )
                metrics = fitted_metrics(test, model.predict(test[["cell_id", "drug_id"]]))
                for column, value in (("fold", split.fold), ("regime", regime), ("seed", seed)):
                    metrics.insert(0, column, value)
                tables.append(metrics)
        atomic_csv(pd.concat(tables, ignore_index=True), path)
        print(f"EVALUATED seed {seed}", flush=True)
    evaluation_files = [evaluation_dir / f"metrics_seed_{seed}.csv" for seed in SEEDS]
    evaluation = pd.concat([pd.read_csv(path) for path in evaluation_files], ignore_index=True)
    per_seed = (
        evaluation.loc[evaluation.metric == "spearman_drug_macro_mean"]
        .groupby(["seed", "regime"], as_index=False).value.mean()
        .pivot(index="seed", columns="regime", values="value").reset_index()
    )
    per_seed["selection_score"] = per_seed[list(REGIMES)].mean(axis=1)
    stability = pd.DataFrame([{
        "config_id": selected_id, "n_seeds": per_seed.seed.nunique(),
        "mean_selection_score": per_seed.selection_score.mean(),
        "seed_sd": per_seed.selection_score.std(ddof=1),
        "seed_min": per_seed.selection_score.min(), "seed_max": per_seed.selection_score.max(),
        **{f"mean_{regime}": per_seed[regime].mean() for regime in REGIMES},
    }])
    atomic_csv(evaluation, args.output / "source_evaluation_metrics.csv")
    atomic_csv(per_seed, args.output / "source_per_seed.csv")
    atomic_csv(stability, args.output / "source_stability.csv")

    # Stage 3: no-retuning external evaluation over all ten seeds.
    cell_reduced = fit_reducer(cell_matrix, cell_ids, set(source.cell_id), 64)[0]
    drug_reduced = fit_reducer(drug_matrix, drug_ids, set(source.drug_id), 64)[0]
    source_cells, source_drugs = set(source.cell_id), set(source.drug_id)
    source_pairs = set(zip(source.cell_id, source.drug_id))
    for seed in SEEDS:
        metric_path = external_dir / f"metrics_seed_{seed}.csv"
        drug_path = external_dir / f"per_drug_seed_{seed}.csv"
        if metric_path.is_file() and drug_path.is_file() and metric_path.stat().st_size > 0 and drug_path.stat().st_size > 0:
            continue
        model = model_from(selected_config, seed).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids)
        metric_tables, drug_tables = [], []
        for study, target in holdout.groupby("study", sort=True):
            target = target.reset_index(drop=True)
            exact_mask = np.fromiter(
                ((cell, drug) in source_pairs for cell, drug in zip(target.cell_id, target.drug_id)),
                dtype=bool, count=len(target),
            )
            domains = {
                "cross_study": target,
                "cross_study_overlap": target.loc[target.cell_id.isin(source_cells) & target.drug_id.isin(source_drugs)].reset_index(drop=True),
                "cross_study_exact_pair": target.loc[exact_mask].reset_index(drop=True),
            }
            for regime, frame in domains.items():
                prediction = model.predict(frame[["cell_id", "drug_id"]])
                metrics = fitted_metrics(frame, prediction)
                drugs = per_drug(frame, prediction)
                for table in (metrics, drugs):
                    table.insert(0, "regime", regime)
                    table.insert(0, "test_study", study)
                    table.insert(0, "seed", seed)
                metric_tables.append(metrics)
                drug_tables.append(drugs)
        atomic_csv(pd.concat(metric_tables, ignore_index=True), metric_path)
        atomic_csv(pd.concat(drug_tables, ignore_index=True), drug_path)
        print(f"EXTERNAL seed {seed}", flush=True)
    external_metrics = pd.concat(
        [pd.read_csv(external_dir / f"metrics_seed_{seed}.csv") for seed in SEEDS], ignore_index=True
    )
    external_drugs = pd.concat(
        [pd.read_csv(external_dir / f"per_drug_seed_{seed}.csv", dtype={"drug_id": str}) for seed in SEEDS], ignore_index=True
    )
    external_summary = (
        external_drugs.groupby(["test_study", "regime", "drug_id"], as_index=False)
        .agg(n_cells=("n_cells", "first"), n_seeds=("seed", "nunique"), spearman=("spearman", "mean"))
        .groupby(["test_study", "regime"], as_index=False)
        .agg(n_eligible_drugs=("spearman", "count"), macro_spearman=("spearman", "mean"), median_spearman=("spearman", "median"))
    )
    atomic_csv(external_metrics, args.output / "external_metrics.csv")
    atomic_csv(external_drugs, args.output / "external_per_drug.csv")
    atomic_csv(external_summary, args.output / "external_summary.csv")
    print(stability.to_string(index=False))
    print(external_summary.to_string(index=False))


if __name__ == "__main__":
    main()
