"""Evaluate the locked inductive-MF configuration over all prespecified seeds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import t


import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from crossdb_drug_response.splits import make_folds  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEEDS = (104729, 130363, 155921, 181081, 205759, 231481, 257053, 282433, 308113, 333667)
REGIMES = ("cold_cell", "cold_drug", "cold_both")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_config).iloc[0]
    rank = int(selected["rank"])
    regularization = float(selected["regularization"])
    iterations = int(selected["iterations"])
    projection_alpha = float(selected["projection_alpha"])

    responses = pd.read_csv(args.responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = responses.loc[
        (responses.study == "CTRPv2")
        & responses.cell_id.isin(cell_ids)
        & responses.drug_id.isin(drug_ids)
    ].reset_index(drop=True)

    for seed in SEEDS:
        output_path = args.output / f"selected_mf_seed_{seed}.csv"
        if output_path.is_file() and output_path.stat().st_size > 0:
            print(f"SKIP seed {seed}", flush=True)
            continue
        cell_cache, drug_cache, rows = {}, {}, []
        print(f"START seed {seed}", flush=True)
        for regime in REGIMES:
            for split in make_folds(source, regime, folds=5, seed=seed):
                train = source.iloc[split.train_index].reset_index(drop=True)
                test = source.iloc[split.test_index].reset_index(drop=True)
                train_cells, train_drugs = frozenset(train.cell_id), frozenset(train.drug_id)
                if train_cells not in cell_cache:
                    cell_cache[train_cells] = fit_reducer(
                        cell_matrix, cell_ids, set(train_cells), components=64
                    )[0]
                if train_drugs not in drug_cache:
                    drug_cache[train_drugs] = fit_reducer(
                        drug_matrix, drug_ids, set(train_drugs), components=64
                    )[0]
                model = InductiveMatrixFactorization(
                    rank=rank,
                    regularization=regularization,
                    iterations=iterations,
                    projection_alpha=projection_alpha,
                    seed=seed,
                ).fit(
                    train, cell_cache[train_cells], cell_ids,
                    drug_cache[train_drugs], drug_ids,
                )
                prediction = model.predict(test[["cell_id", "drug_id"]])
                evaluation = test[["cell_id", "drug_id"]].copy()
                evaluation["y_true"] = test.sensitivity.to_numpy()
                evaluation["y_pred"] = prediction
                metrics = evaluate_predictions(evaluation)
                for metric in ("spearman_drug_macro_mean", "mae"):
                    value = metrics.loc[metrics.metric == metric, "value"].iloc[0]
                    rows.append(
                        {
                            "seed": seed, "regime": regime, "fold": split.fold,
                            "metric": metric, "value": value,
                            "rank": rank, "regularization": regularization,
                            "iterations": iterations, "projection_alpha": projection_alpha,
                            "n_test_pairs": len(test),
                        }
                    )
        atomic_csv(pd.DataFrame(rows), output_path)
        print(f"END seed {seed}", flush=True)

    files = sorted(args.output.glob("selected_mf_seed_*.csv"))
    if len(files) != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} seed files, found {len(files)}")
    all_results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    selection = (
        all_results.loc[all_results.metric == "spearman_drug_macro_mean"]
        .groupby("seed", as_index=False)
        .agg(selection_score=("value", "mean"))
    )
    mean = float(selection.selection_score.mean())
    sd = float(selection.selection_score.std(ddof=1))
    half = float(t.ppf(0.975, len(selection) - 1) * sd / np.sqrt(len(selection)))
    summary = pd.DataFrame(
        [{
            "rank": rank, "regularization": regularization,
            "iterations": iterations, "projection_alpha": projection_alpha,
            "n_seeds": len(selection), "mean_selection_score": mean,
            "seed_sd": sd, "ci_low": mean - half, "ci_high": mean + half,
            "min_seed_score": selection.selection_score.min(),
            "max_seed_score": selection.selection_score.max(),
        }]
    )
    atomic_csv(all_results, args.output / "selected_mf_all_metrics.csv")
    atomic_csv(selection, args.output / "selected_mf_per_seed.csv")
    atomic_csv(summary, args.output / "selected_mf_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
