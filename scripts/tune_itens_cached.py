"""Evaluate the frozen kNN/ITENS grid using cached expert predictions.

All 3,942 named configurations are retained in the alias ledger. Exactly
equivalent pure-expert aliases are evaluated once, yielding 3,213 distinct
prediction functions per fold.
"""

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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.itens import ITENSRegressor  # noqa: E402
from crossdb_drug_response.splits import make_folds  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEED = 104729
REGIMES = ("cold_cell", "cold_drug", "cold_both")
K_VALUES = (8, 16, 32)
TEMPERATURES = (0.05, 0.10, 0.20)
SHRINKAGES = (0.0, 10.0, 50.0)
THETAS = tuple(
    (a / 4, b / 4, c / 4)
    for a in range(5) for b in range(5 - a) for c in [4 - a - b]
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def theta_id(theta) -> str:
    return "_".join(f"{value:g}" for value in theta)


def basis_id(kc, kd, tc, td, ess) -> str:
    return f"kc{kc}_kd{kd}_tc{tc:g}_td{td:g}_ess{ess:g}"


def canonical_id(kc, kd, tc, td, ess, theta) -> str:
    if theta == (1.0, 0.0, 0.0):
        return f"cell_k{kc}_t{tc:g}_ess{ess:g}"
    if theta == (0.0, 1.0, 0.0):
        return f"drug_k{kd}_t{td:g}_ess{ess:g}"
    if theta == (0.0, 0.0, 1.0):
        return f"product_{basis_id(kc, kd, tc, td, ess)}"
    return f"itens_{basis_id(kc, kd, tc, td, ess)}_theta{theta_id(theta)}"


def alias_ledger() -> pd.DataFrame:
    rows = []
    for expert in ("cell", "drug"):
        for k, temperature, ess in itertools.product(K_VALUES, TEMPERATURES, SHRINKAGES):
            canonical = f"{expert}_k{k}_t{temperature:g}_ess{ess:g}"
            rows.append({"named_config_id": f"standalone_{canonical}", "canonical_config_id": canonical, "alias": False})
    for kc, kd, tc, td, ess in itertools.product(K_VALUES, K_VALUES, TEMPERATURES, TEMPERATURES, SHRINKAGES):
        canonical = canonical_id(kc, kd, tc, td, ess, (0.0, 0.0, 1.0))
        rows.append({"named_config_id": f"standalone_{canonical}", "canonical_config_id": canonical, "alias": False})
        for theta in THETAS:
            canonical = canonical_id(kc, kd, tc, td, ess, theta)
            named = f"itens_{basis_id(kc, kd, tc, td, ess)}_theta{theta_id(theta)}"
            rows.append({"named_config_id": named, "canonical_config_id": canonical, "alias": named != canonical})
    return pd.DataFrame(rows)


def score_predictions(
    test: pd.DataFrame, predictions: np.ndarray, identifiers: list[str], parameters: list[dict]
) -> pd.DataFrame:
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    true = test.sensitivity.to_numpy(dtype=float)
    correlations = [[] for _ in identifiers]
    for _, indices in test.groupby("drug_id", sort=False).indices.items():
        indices = np.asarray(indices)
        if len(indices) < 5 or np.unique(true[indices]).size < 2:
            continue
        true_rank = rankdata(true[indices], method="average")
        true_centered = true_rank - true_rank.mean()
        true_norm = np.sqrt(np.dot(true_centered, true_centered))
        predicted_rank = rankdata(predictions[indices], method="average", axis=0)
        predicted_centered = predicted_rank - predicted_rank.mean(axis=0)
        predicted_norm = np.sqrt(np.sum(predicted_centered * predicted_centered, axis=0))
        numerator = true_centered @ predicted_centered
        values = np.divide(
            numerator, true_norm * predicted_norm,
            out=np.full(len(identifiers), np.nan), where=predicted_norm > 0,
        )
        for column, value in enumerate(values):
            if np.isfinite(value):
                correlations[column].append(float(value))
    mae = np.mean(np.abs(predictions - true[:, None]), axis=0)
    rows = []
    for index, identifier in enumerate(identifiers):
        rows.append({
            "config_id": identifier,
            "macro_spearman": float(np.mean(correlations[index])) if correlations[index] else np.nan,
            "n_eligible_drugs": len(correlations[index]), "mae": float(mae[index]),
            **parameters[index],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    aliases = alias_ledger()
    if len(aliases) != 3_942 or aliases.canonical_config_id.nunique() != 3_213:
        raise AssertionError("Unexpected named or canonical kNN/ITENS grid size")
    atomic_csv(aliases, args.output / "alias_ledger.csv")

    responses = pd.read_csv(args.responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = responses.loc[
        (responses.study == "CTRPv2") & responses.cell_id.isin(cell_ids) & responses.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    for regime in REGIMES:
        for split in make_folds(source, regime, folds=5, seed=SEED):
            path = checkpoints / f"{regime}_fold{split.fold}.csv"
            if path.is_file() and len(pd.read_csv(path)) == 3_213:
                print(f"SKIP {regime} fold={split.fold}", flush=True)
                continue
            print(f"START {regime} fold={split.fold}", flush=True)
            train = source.iloc[split.train_index].reset_index(drop=True)
            test = source.iloc[split.test_index].reset_index(drop=True)
            cell_reduced = fit_reducer(cell_matrix, cell_ids, set(train.cell_id), 64)[0]
            drug_reduced = fit_reducer(drug_matrix, drug_ids, set(train.drug_id), 64)[0]
            model = ITENSRegressor(theta=(0.0, 0.0, 0.0)).fit(
                train, cell_reduced, cell_ids, drug_reduced, drug_ids
            )
            fold_tables = []
            for kc, kd, tc, td, ess in itertools.product(K_VALUES, K_VALUES, TEMPERATURES, TEMPERATURES, SHRINKAGES):
                model.k_cell, model.k_drug = kc, kd
                model.temperature_cell, model.temperature_drug = tc, td
                model.ess_shrinkage = ess
                _, diagnostics = model.predict(test[["cell_id", "drug_id"]], return_diagnostics=True)
                expert = diagnostics[["cell_expert", "drug_expert", "product_expert"]].to_numpy()
                reliability = diagnostics[["cell_reliability", "drug_reliability", "product_reliability"]].to_numpy()
                components = reliability * (expert - model.global_mean_)
                theta_batch = [theta for theta in THETAS if theta not in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
                if kd == K_VALUES[0] and td == TEMPERATURES[0]:
                    theta_batch.append((1.0, 0.0, 0.0))
                if kc == K_VALUES[0] and tc == TEMPERATURES[0]:
                    theta_batch.append((0.0, 1.0, 0.0))
                theta_matrix = np.asarray(theta_batch)
                prediction = model.global_mean_ + components @ theta_matrix.T
                identifiers = [canonical_id(kc, kd, tc, td, ess, theta) for theta in theta_batch]
                parameters = [{
                    "k_cell": kc, "k_drug": kd, "temperature_cell": tc,
                    "temperature_drug": td, "ess_shrinkage": ess,
                    "theta_cell": theta[0], "theta_drug": theta[1], "theta_product": theta[2],
                } for theta in theta_batch]
                fold_tables.append(score_predictions(test, prediction, identifiers, parameters))
            fold = pd.concat(fold_tables, ignore_index=True)
            if len(fold) != 3_213 or fold.config_id.nunique() != 3_213:
                raise AssertionError("Incomplete canonical grid in fold")
            fold.insert(0, "fold", split.fold)
            fold.insert(0, "regime", regime)
            fold.insert(0, "seed", SEED)
            atomic_csv(fold, path)
            print(f"END {regime} fold={split.fold}", flush=True)

    files = [checkpoints / f"{regime}_fold{fold}.csv" for regime in REGIMES for fold in range(5)]
    ledger = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    summary = ledger.groupby("config_id", as_index=False).agg(
        selection_score=("macro_spearman", "mean"), mean_mae=("mae", "mean"),
        fold_sd=("macro_spearman", "std"), n_folds=("fold", "size"),
        valid_folds=("macro_spearman", "count"), valid_regimes=("regime", "nunique"),
        n_eligible_drugs_min=("n_eligible_drugs", "min"),
    )
    summary.loc[(summary.valid_folds < 15) | (summary.valid_regimes < 3), "selection_score"] = np.nan
    summary = summary.sort_values(
        ["selection_score", "mean_mae", "config_id"],
        ascending=[False, True, True], na_position="last", ignore_index=True,
    )
    atomic_csv(ledger, args.output / "grid_ledger.csv")
    atomic_csv(summary, args.output / "grid_summary.csv")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
