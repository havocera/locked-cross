"""Run one seed of all fair baselines under leakage-safe internal regimes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization  # noqa: E402
from crossdb_drug_response.itens import ITENSRegressor  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from crossdb_drug_response.splits import make_folds  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix, make_model, pair_features  # noqa: E402


REGIMES = ("random_pair", "cold_cell", "cold_drug", "cold_both")


def per_drug(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    work = frame[["cell_id", "drug_id"]].copy()
    work["y_true"] = frame.sensitivity.to_numpy()
    work["y_pred"] = prediction
    rows = []
    for drug_id, group in work.groupby("drug_id", sort=True):
        value = np.nan
        if group.cell_id.nunique() >= 5 and group.y_true.nunique() >= 2 and group.y_pred.nunique() >= 2:
            value = float(spearmanr(group.y_true, group.y_pred).statistic)
        rows.append({"drug_id": drug_id, "n_cells": group.cell_id.nunique(), "spearman": value})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
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

    metric_tables, drug_tables, split_rows = [], [], []
    cell_reducer_cache = {}
    drug_reducer_cache = {}
    for regime in REGIMES:
        for split in make_folds(source, regime, folds=5, seed=args.seed):
            train = source.iloc[split.train_index].reset_index(drop=True)
            test = source.iloc[split.test_index].reset_index(drop=True)
            train_cells = frozenset(train.cell_id)
            train_drugs = frozenset(train.drug_id)
            if train_cells not in cell_reducer_cache:
                cell_reducer_cache[train_cells] = fit_reducer(
                    cell_matrix, cell_ids, set(train_cells), components=64
                )[0]
            if train_drugs not in drug_reducer_cache:
                drug_reducer_cache[train_drugs] = fit_reducer(
                    drug_matrix, drug_ids, set(train_drugs), components=64
                )[0]
            cell_reduced = cell_reducer_cache[train_cells]
            drug_reduced = drug_reducer_cache[train_drugs]
            cell_lookup = {value: i for i, value in enumerate(cell_ids)}
            drug_lookup = {value: i for i, value in enumerate(drug_ids)}

            def build(frame):
                return pair_features(
                    frame, cell_reduced, cell_lookup, drug_reduced, drug_lookup
                )

            train_x = build(train)
            test_x = build(test)
            models = {
                "additive_matched": make_model("additive_matched").fit(train),
                "ridge_bilinear": make_model("ridge_bilinear").fit(train_x, train.sensitivity),
                "histgb": make_model("histgb").fit(train_x, train.sensitivity),
                "lightgbm": make_model("lightgbm").fit(train_x, train.sensitivity),
                "inductive_mf": InductiveMatrixFactorization(seed=args.seed).fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                ),
                "cell_knn": ITENSRegressor(theta=(1.0, 0.0, 0.0)).fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                ),
                "drug_knn": ITENSRegressor(theta=(0.0, 1.0, 0.0)).fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                ),
                "product_knn": ITENSRegressor(theta=(0.0, 0.0, 1.0)).fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                ),
                "itens_k16": ITENSRegressor().fit(
                    train, cell_reduced, cell_ids, drug_reduced, drug_ids
                ),
            }
            for model_name, model in models.items():
                if model_name == "additive_matched":
                    prediction = model.predict(test)
                elif model_name in {"ridge_bilinear", "histgb", "lightgbm"}:
                    prediction = model.predict(test_x)
                else:
                    prediction = model.predict(test[["cell_id", "drug_id"]])
                evaluation = test[["cell_id", "drug_id"]].copy()
                evaluation["y_true"] = test.sensitivity.to_numpy()
                evaluation["y_pred"] = prediction
                metrics = evaluate_predictions(evaluation)
                for column, value in (
                    ("model", model_name), ("seed", args.seed),
                    ("regime", regime), ("fold", split.fold),
                ):
                    metrics.insert(0, column, value)
                metric_tables.append(metrics)
                drugs = per_drug(test, prediction)
                for column, value in (
                    ("model", model_name), ("seed", args.seed),
                    ("regime", regime), ("fold", split.fold),
                ):
                    drugs.insert(0, column, value)
                drug_tables.append(drugs)
            split_rows.append(
                {
                    "seed": args.seed, "regime": regime, "fold": split.fold,
                    "n_train": len(train), "n_test": len(test),
                    "train_cells": train.cell_id.nunique(), "test_cells": test.cell_id.nunique(),
                    "train_drugs": train.drug_id.nunique(), "test_drugs": test.drug_id.nunique(),
                }
            )

    metrics = pd.concat(metric_tables, ignore_index=True)
    drugs = pd.concat(drug_tables, ignore_index=True)
    metrics.to_csv(args.output / f"metrics_seed_{args.seed}.csv", index=False)
    drugs.to_csv(args.output / f"per_drug_seed_{args.seed}.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output / f"splits_seed_{args.seed}.csv", index=False)
    selection_input = metrics.loc[
        metrics.regime.isin(["cold_cell", "cold_drug", "cold_both"])
        & (metrics.metric == "spearman_drug_macro_mean")
    ]
    selection = (
        selection_input.groupby("model", as_index=False)
        .agg(
            value=("value", "mean"),
            valid_regimes=("regime", lambda values: selection_input.loc[values.index].groupby("regime").value.apply(lambda x: x.notna().any()).sum()),
        )
    )
    selection.loc[selection.valid_regimes < 3, "value"] = np.nan
    selection = selection.sort_values(["value", "model"], ascending=[False, True], na_position="last")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
