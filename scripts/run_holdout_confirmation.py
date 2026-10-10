"""One-time evaluation of locked models on GDSCv1 and NCI60 holdouts."""

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
from run_feature_baseline import fit_reducer, load_matrix, make_model, pair_features, predict_chunks  # noqa: E402


SEED = 104729


def drug_scores(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    work = frame[["cell_id", "drug_id"]].copy()
    work["y_true"] = frame.sensitivity.to_numpy()
    work["y_pred"] = prediction
    rows = []
    for drug_id, group in work.groupby("drug_id", sort=True):
        n = group.cell_id.nunique()
        value = np.nan
        if n >= 5 and group.y_true.nunique() >= 2 and group.y_pred.nunique() >= 2:
            value = float(spearmanr(group.y_true, group.y_pred).statistic)
        rows.append({"drug_id": drug_id, "n_cells": n, "spearman": value})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-responses", type=Path, required=True)
    parser.add_argument("--holdout-responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    cell_lookup = {value: i for i, value in enumerate(cell_ids)}
    drug_lookup = {value: i for i, value in enumerate(drug_ids)}

    def build(frame):
        return pair_features(frame, cell_reduced, cell_lookup, drug_reduced, drug_lookup)

    source_x = build(source)
    selected = pd.read_csv(args.selected_config).iloc[0]
    models = {
        "additive_matched": make_model("additive_matched").fit(source),
        "ridge_bilinear": make_model("ridge_bilinear").fit(source_x, source.sensitivity),
        "histgb": make_model("histgb").fit(source_x, source.sensitivity),
        "lightgbm": make_model("lightgbm").fit(source_x, source.sensitivity),
        "selected_inductive_mf": InductiveMatrixFactorization(
            rank=int(selected["rank"]),
            regularization=float(selected["regularization"]),
            iterations=int(selected["iterations"]),
            projection_alpha=float(selected["projection_alpha"]),
            seed=SEED,
        ).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids),
        "cell_knn": ITENSRegressor(theta=(1.0, 0.0, 0.0)).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids),
        "drug_knn": ITENSRegressor(theta=(0.0, 1.0, 0.0)).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids),
        "product_knn": ITENSRegressor(theta=(0.0, 0.0, 1.0)).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids),
        "itens_k16": ITENSRegressor().fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids),
    }

    source_cells, source_drugs = set(source.cell_id), set(source.drug_id)
    source_pairs = set(zip(source.cell_id, source.drug_id))
    metric_tables, score_tables, coverage = [], [], []
    for study, target in holdout.groupby("study", sort=True):
        target = target.reset_index(drop=True)
        regimes = {
            "cross_study": target,
            "cross_study_overlap": target.loc[
                target.cell_id.isin(source_cells) & target.drug_id.isin(source_drugs)
            ].reset_index(drop=True),
            "cross_study_exact_pair": target.loc[
                [(cell, drug) in source_pairs for cell, drug in zip(target.cell_id, target.drug_id)]
            ].reset_index(drop=True),
        }
        coverage.append({"study": study, **{f"n_{name}": len(frame) for name, frame in regimes.items()}})
        for regime, frame in regimes.items():
            if frame.empty:
                continue
            for model_name, model in models.items():
                if model_name == "additive_matched":
                    prediction = model.predict(frame)
                elif model_name in {"ridge_bilinear", "histgb", "lightgbm"}:
                    prediction = predict_chunks(model, frame, build, chunk_size=50_000)
                else:
                    prediction = model.predict(frame[["cell_id", "drug_id"]])
                evaluation = frame[["cell_id", "drug_id"]].copy()
                evaluation["y_true"] = frame.sensitivity.to_numpy()
                evaluation["y_pred"] = prediction
                metrics = evaluate_predictions(evaluation)
                metrics.insert(0, "model", model_name)
                metrics.insert(0, "regime", regime)
                metrics.insert(0, "test_study", study)
                metric_tables.append(metrics)
                scores = drug_scores(frame, prediction)
                scores.insert(0, "model", model_name)
                scores.insert(0, "regime", regime)
                scores.insert(0, "test_study", study)
                score_tables.append(scores)

    metrics = pd.concat(metric_tables, ignore_index=True)
    scores = pd.concat(score_tables, ignore_index=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    scores.to_csv(args.output / "per_drug_scores.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    pd.DataFrame([
        {"item": "cell_reducer", **cell_audit},
        {"item": "drug_reducer", **drug_audit},
    ]).to_csv(args.output / "model_audit.csv", index=False)
    print(metrics.loc[metrics.metric == "spearman_drug_macro_mean"].to_string(index=False))


if __name__ == "__main__":
    main()
