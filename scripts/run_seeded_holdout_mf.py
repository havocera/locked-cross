"""Complete the prespecified ten-seed external evaluation of the locked MF.

This writes beside, and never overwrites, the original single-seed confirmation.
Per-seed checkpoints make the long run resumable. Matched-domain row predictions are
retained for the prespecified crossed cell-and-drug bootstrap sensitivity analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization  # noqa: E402
from crossdb_drug_response.itens import ITENSRegressor  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEEDS = (104729, 130363, 155921, 181081, 205759, 231481, 257053, 282433, 308113, 333667)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    suffix = ".csv.gz" if path.name.endswith(".csv.gz") else ".csv"
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=suffix)
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False, compression="gzip" if suffix.endswith(".gz") else None)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def predict_chunks(model, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(frame), chunk_size):
        outputs.append(model.predict(frame.iloc[start : start + chunk_size][["cell_id", "drug_id"]]))
    return np.concatenate(outputs)


def drug_scores(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
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


def diagnostics(prediction: np.ndarray, source_min: float, source_max: float) -> dict[str, float]:
    return {
        "prediction_min": float(np.min(prediction)),
        "prediction_q001": float(np.quantile(prediction, 0.001)),
        "prediction_q01": float(np.quantile(prediction, 0.01)),
        "prediction_median": float(np.median(prediction)),
        "prediction_q99": float(np.quantile(prediction, 0.99)),
        "prediction_q999": float(np.quantile(prediction, 0.999)),
        "prediction_max": float(np.max(prediction)),
        "fraction_outside_auc_range": float(np.mean((prediction < -1.0) | (prediction > 0.0))),
        "fraction_outside_source_range": float(np.mean((prediction < source_min) | (prediction > source_max))),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-responses", type=Path, required=True)
    parser.add_argument("--holdout-responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if not args.model_lock.is_file():
        raise RuntimeError("Model lock is missing")
    args.output.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(args.development_responses, dtype={"cell_id": str, "drug_id": str})
    holdout = pd.read_csv(args.holdout_responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = development.loc[
        (development.study == "CTRPv2")
        & development.cell_id.isin(cell_ids)
        & development.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    holdout = holdout.loc[
        holdout.cell_id.isin(cell_ids) & holdout.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    if source.empty or holdout.empty:
        raise RuntimeError("Source or holdout feature-complete response is empty")

    cell_reduced, _, cell_audit = fit_reducer(cell_matrix, cell_ids, set(source.cell_id), 64)
    drug_reduced, _, drug_audit = fit_reducer(drug_matrix, drug_ids, set(source.drug_id), 64)
    atomic_csv(pd.DataFrame([
        {"item": "cell_reducer", **cell_audit},
        {"item": "drug_reducer", **drug_audit},
    ]), args.output / "model_audit.csv")

    selected = pd.read_csv(args.selected_config).iloc[0]
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    if sha256(args.selected_config) != lock["grid_sha256"]:
        raise RuntimeError("Selected configuration does not match the frozen grid hash")
    for key in ("rank", "regularization", "iterations", "projection_alpha"):
        if float(selected[key]) != float(lock["selected_configuration"][key]):
            raise RuntimeError(f"Selected configuration differs from model lock: {key}")
    source_cells, source_drugs = set(source.cell_id), set(source.drug_id)
    source_pairs = set(zip(source.cell_id, source.drug_id))
    source_min, source_max = float(source.sensitivity.min()), float(source.sensitivity.max())

    for seed in SEEDS:
        metrics_path = args.output / f"metrics_seed_{seed}.csv"
        scores_path = args.output / f"per_drug_seed_{seed}.csv"
        rows_path = args.output / f"matched_rows_seed_{seed}.csv.gz"
        diagnostics_path = args.output / f"diagnostics_seed_{seed}.csv"
        if all(path.is_file() and path.stat().st_size > 0 for path in (
            metrics_path, scores_path, rows_path, diagnostics_path
        )):
            print(f"SKIP seed {seed}", flush=True)
            continue
        print(f"START seed {seed}", flush=True)
        model = InductiveMatrixFactorization(
            rank=int(selected["rank"]),
            regularization=float(selected["regularization"]),
            iterations=int(selected["iterations"]),
            projection_alpha=float(selected["projection_alpha"]),
            seed=seed,
        ).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids)
        metric_tables, score_tables, matched_tables, diagnostic_rows = [], [], [], []
        for study, target in holdout.groupby("study", sort=True):
            target = target.reset_index(drop=True)
            exact_mask = np.fromiter(
                ((cell, drug) in source_pairs for cell, drug in zip(target.cell_id, target.drug_id)),
                dtype=bool,
                count=len(target),
            )
            regimes = {
                "cross_study": target,
                "cross_study_overlap": target.loc[
                    target.cell_id.isin(source_cells) & target.drug_id.isin(source_drugs)
                ].reset_index(drop=True),
                "cross_study_exact_pair": target.loc[exact_mask].reset_index(drop=True),
            }
            for regime, frame in regimes.items():
                if frame.empty:
                    continue
                prediction = predict_chunks(model, frame, args.chunk_size)
                evaluation = frame[["cell_id", "drug_id"]].copy()
                evaluation["y_true"] = frame.sensitivity.to_numpy()
                evaluation["y_pred"] = prediction
                metrics = evaluate_predictions(evaluation)
                for column, value in (("seed", seed), ("model", "selected_inductive_mf"),
                                      ("regime", regime), ("test_study", study)):
                    metrics.insert(0, column, value)
                metric_tables.append(metrics)
                scores = drug_scores(frame, prediction)
                for column, value in (("seed", seed), ("model", "selected_inductive_mf"),
                                      ("regime", regime), ("test_study", study)):
                    scores.insert(0, column, value)
                score_tables.append(scores)
                diagnostic_rows.append({
                    "seed": seed, "test_study": study, "regime": regime,
                    "n_pairs": len(frame), **diagnostics(prediction, source_min, source_max),
                })
                if regime == "cross_study_overlap":
                    matched = evaluation.copy()
                    matched.insert(0, "seed", seed)
                    matched.insert(0, "test_study", study)
                    matched_tables.append(matched)
        atomic_csv(pd.concat(metric_tables, ignore_index=True), metrics_path)
        atomic_csv(pd.concat(score_tables, ignore_index=True), scores_path)
        atomic_csv(pd.concat(matched_tables, ignore_index=True), rows_path)
        atomic_csv(pd.DataFrame(diagnostic_rows), diagnostics_path)
        print(f"END seed {seed}", flush=True)

    metric_files = sorted(args.output.glob("metrics_seed_*.csv"))
    score_files = sorted(args.output.glob("per_drug_seed_*.csv"))
    diagnostic_files = sorted(args.output.glob("diagnostics_seed_*.csv"))
    if not (len(metric_files) == len(score_files) == len(diagnostic_files) == len(SEEDS)):
        raise RuntimeError("Expected ten complete seeded MF checkpoints")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    scores = pd.concat([pd.read_csv(path, dtype={"drug_id": str}) for path in score_files], ignore_index=True)
    diagnostic_table = pd.concat([pd.read_csv(path) for path in diagnostic_files], ignore_index=True)
    seed_summary = (
        metrics.groupby(["test_study", "regime", "metric"], as_index=False)
        .agg(n_seeds=("seed", "nunique"), mean=("value", "mean"), seed_sd=("value", "std"),
             minimum=("value", "min"), maximum=("value", "max"))
    )
    per_drug_mean = (
        scores.groupby(["test_study", "regime", "drug_id"], as_index=False)
        .agg(n_cells=("n_cells", "first"), n_seeds=("seed", "nunique"), spearman=("spearman", "mean"))
    )
    atomic_csv(metrics, args.output / "all_metrics.csv")
    atomic_csv(scores, args.output / "all_per_drug_scores.csv")
    atomic_csv(diagnostic_table, args.output / "all_diagnostics.csv")
    atomic_csv(seed_summary, args.output / "seed_variability.csv")
    atomic_csv(per_drug_mean, args.output / "per_drug_seed_mean.csv")

    itens_path = args.output / "itens_matched_rows.csv.gz"
    if not itens_path.is_file() or itens_path.stat().st_size == 0:
        itens = ITENSRegressor().fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids)
        tables = []
        for study, target in holdout.groupby("study", sort=True):
            matched = target.loc[
                target.cell_id.isin(source_cells) & target.drug_id.isin(source_drugs)
            ].reset_index(drop=True)
            prediction = predict_chunks(itens, matched, args.chunk_size)
            table = matched[["cell_id", "drug_id"]].copy()
            table["y_true"] = matched.sensitivity.to_numpy()
            table["y_pred"] = prediction
            table.insert(0, "test_study", study)
            tables.append(table)
        atomic_csv(pd.concat(tables, ignore_index=True), itens_path)
    print(seed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
