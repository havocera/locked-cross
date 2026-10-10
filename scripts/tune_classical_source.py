"""Complete the frozen Ridge/HGB/LightGBM source-only grids on seed 104729."""

from __future__ import annotations

import argparse
import itertools
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
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from crossdb_drug_response.splits import make_folds  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix, pair_features  # noqa: E402


SEED = 104729
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


def configurations():
    for alpha in (1.0, 10.0, 100.0):
        yield f"ridge_alpha{alpha:g}", "ridge", {"alpha": alpha}
    for learning_rate, leaves, l2 in itertools.product((0.05, 0.10), (15, 31), (1.0, 10.0)):
        yield (
            f"histgb_lr{learning_rate:g}_leaves{leaves}_l2{l2:g}", "histgb",
            {"learning_rate": learning_rate, "max_leaf_nodes": leaves, "l2_regularization": l2},
        )
    for leaves, min_leaf, l1, l2 in itertools.product((15, 31), (20, 50), (0.0, 1.0), (1.0, 10.0)):
        yield (
            f"lightgbm_leaves{leaves}_min{min_leaf}_l1{l1:g}_l2{l2:g}", "lightgbm",
            {"num_leaves": leaves, "min_child_samples": min_leaf, "reg_alpha": l1, "reg_lambda": l2},
        )


def make_model(family: str, parameters: dict):
    if family == "ridge":
        return Ridge(alpha=parameters["alpha"], solver="lsqr", tol=1e-4)
    if family == "histgb":
        return HistGradientBoostingRegressor(
            **parameters, max_iter=150, min_samples_leaf=30, max_bins=128,
            early_stopping=False, random_state=SEED,
        )
    if family == "lightgbm":
        return LGBMRegressor(
            **parameters, objective="regression_l2", n_estimators=300, learning_rate=0.05,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True,
        )
    raise ValueError(family)


def complexity(model, family: str) -> int:
    if family == "ridge":
        return int(model.coef_.size + 1)
    if family == "histgb":
        return int(sum(len(predictor.nodes) for stage in model._predictors for predictor in stage))
    dump = model.booster_.dump_model()

    def leaves(node):
        if "leaf_index" in node:
            return 1
        return leaves(node["left_child"]) + leaves(node["right_child"])

    return int(sum(leaves(tree["tree_structure"]) for tree in dump["tree_info"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)

    responses = pd.read_csv(args.responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = responses.loc[
        (responses.study == "CTRPv2") & responses.cell_id.isin(cell_ids) & responses.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    configs = list(configurations())
    for regime in REGIMES:
        for split in make_folds(source, regime, folds=5, seed=SEED):
            train = source.iloc[split.train_index].reset_index(drop=True)
            test = source.iloc[split.test_index].reset_index(drop=True)
            cell_reduced, cell_lookup, _ = fit_reducer(cell_matrix, cell_ids, set(train.cell_id), 64)
            drug_reduced, drug_lookup, _ = fit_reducer(drug_matrix, drug_ids, set(train.drug_id), 64)
            train_x = pair_features(train, cell_reduced, cell_lookup, drug_reduced, drug_lookup)
            test_x = pair_features(test, cell_reduced, cell_lookup, drug_reduced, drug_lookup)
            for identifier, family, parameters in configs:
                path = checkpoint / f"{regime}_fold{split.fold}_{identifier}.csv"
                if path.is_file() and path.stat().st_size > 0:
                    continue
                print(f"START {regime} fold={split.fold} {identifier}", flush=True)
                model = make_model(family, parameters).fit(train_x, train.sensitivity.to_numpy())
                prediction = model.predict(test_x)
                evaluation = test[["cell_id", "drug_id"]].copy()
                evaluation["y_true"] = test.sensitivity.to_numpy()
                evaluation["y_pred"] = prediction
                metrics = evaluate_predictions(evaluation)
                row = {
                    "config_id": identifier, "family": family, "seed": SEED,
                    "regime": regime, "fold": split.fold, "n_test_pairs": len(test),
                    "macro_spearman": metrics.loc[metrics.metric == "spearman_drug_macro_mean", "value"].iloc[0],
                    "mae": metrics.loc[metrics.metric == "mae", "value"].iloc[0],
                    "complexity_proxy": complexity(model, family),
                    **parameters,
                }
                atomic_csv(pd.DataFrame([row]), path)
    files = sorted(checkpoint.glob("*.csv"))
    expected = len(configs) * len(REGIMES) * 5
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} checkpoints, found {len(files)}")
    ledger = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    summary = ledger.groupby(["config_id", "family"], as_index=False).agg(
        selection_score=("macro_spearman", "mean"), mean_mae=("mae", "mean"),
        fold_sd=("macro_spearman", "std"), n_folds=("fold", "size"),
        valid_folds=("macro_spearman", "count"), valid_regimes=("regime", "nunique"),
        complexity_proxy=("complexity_proxy", "first"),
    )
    summary.loc[
        (summary.valid_folds < 15) | (summary.valid_regimes < 3), "selection_score"
    ] = np.nan
    summary = summary.sort_values(
        ["selection_score", "mean_mae", "complexity_proxy", "config_id"],
        ascending=[False, True, True, True], na_position="last", ignore_index=True,
    )
    atomic_csv(ledger, args.output / "grid_ledger.csv")
    atomic_csv(summary, args.output / "grid_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
