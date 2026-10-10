"""Diagnose target-drug OOD distance and source-defined prediction safeguards.

The feature transform, nearest-source distances, model parameters, and clipping
bounds are all fitted or specified without target outcomes. Target labels are read
only to evaluate the frozen diagnostics and safeguards.
"""

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
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEEDS = (104729, 130363, 155921, 181081, 205759, 231481, 257053, 282433, 308113, 333667)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    compressed = path.name.endswith(".csv.gz")
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp-", suffix=".csv.gz" if compressed else ".csv"
    )
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False, compression="gzip" if compressed else None)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def nearest_source_diagnostics(
    embedding: np.ndarray,
    entity_ids: np.ndarray,
    source_entities: set[str],
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    source_mask = np.isin(entity_ids, list(source_entities))
    source = embedding[source_mask].astype(np.float64)
    source_ids = entity_ids[source_mask]
    source_norm = np.linalg.norm(source, axis=1, keepdims=True)
    source_unit = np.divide(source, source_norm, out=np.zeros_like(source), where=source_norm > 0)
    rows = []
    for start in range(0, len(entity_ids), chunk_size):
        query = embedding[start : start + chunk_size].astype(np.float64)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query_unit = np.divide(query, query_norm, out=np.zeros_like(query), where=query_norm > 0)
        cosine = query_unit @ source_unit.T
        nearest_index = np.argmax(cosine, axis=1)
        nearest_similarity = cosine[np.arange(len(query)), nearest_index]
        # PCA-space standardized Euclidean distance is reported as a complementary
        # radial diagnostic; dimensions are scaled by source score SD only.
        scale = source.std(axis=0, ddof=1)
        scale[scale <= 0] = 1.0
        query_scaled = query / scale
        source_scaled = source / scale
        query_sq = np.sum(query_scaled * query_scaled, axis=1, keepdims=True)
        source_sq = np.sum(source_scaled * source_scaled, axis=1)
        distance_sq = np.maximum(query_sq + source_sq[None, :] - 2 * query_scaled @ source_scaled.T, 0)
        nearest_distance = np.sqrt(np.min(distance_sq, axis=1))
        for offset, entity_id in enumerate(entity_ids[start : start + chunk_size]):
            rows.append({
                "drug_id": entity_id,
                "seen_in_source": bool(source_mask[start + offset]),
                "nearest_source_drug_id": source_ids[nearest_index[offset]],
                "nearest_source_cosine": float(nearest_similarity[offset]),
                "nearest_source_distance": float(nearest_distance[offset]),
                "pca_radius": float(np.linalg.norm(query_scaled[offset])),
            })
    return pd.DataFrame(rows)


def calibration(frame: pd.DataFrame) -> tuple[float, float]:
    if len(frame) < 2 or frame.y_pred.nunique() < 2:
        return np.nan, np.nan
    model = LinearRegression().fit(frame[["y_pred"]], frame.y_true)
    return float(model.coef_[0]), float(model.intercept_)


def evaluate_variant(frame: pd.DataFrame, prediction: np.ndarray, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = frame[["cell_id", "drug_id"]].copy()
    evaluation["y_true"] = frame.sensitivity.to_numpy()
    evaluation["y_pred"] = prediction
    metrics = evaluate_predictions(evaluation)
    slope, intercept = calibration(evaluation)
    extra = pd.DataFrame([
        {"metric": "calibration_slope", "value": slope},
        {"metric": "calibration_intercept", "value": intercept},
        {"metric": "prediction_min", "value": float(np.min(prediction))},
        {"metric": "prediction_max", "value": float(np.max(prediction))},
        {"metric": "fraction_outside_auc_range", "value": float(np.mean((prediction < -1) | (prediction > 0)))},
    ])
    metrics = pd.concat([metrics, extra], ignore_index=True)
    metrics.insert(0, "variant", variant)
    return metrics, evaluation


def drug_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, group in evaluation.groupby("drug_id", sort=True):
        rho = np.nan
        if group.cell_id.nunique() >= 5 and group.y_true.nunique() >= 2 and group.y_pred.nunique() >= 2:
            rho = float(spearmanr(group.y_true, group.y_pred).statistic)
        error = group.y_pred - group.y_true
        rows.append({
            "drug_id": drug, "n_cells": group.cell_id.nunique(), "spearman": rho,
            "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error * error))),
            "mean_abs_prediction": float(np.mean(np.abs(group.y_pred))),
            "fraction_outside_auc_range": float(np.mean((group.y_pred < -1) | (group.y_pred > 0))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-responses", type=Path, required=True)
    parser.add_argument("--holdout-responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(args.development_responses, dtype={"cell_id": str, "drug_id": str})
    holdout = pd.read_csv(args.holdout_responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    source = development.loc[
        (development.study == "CTRPv2") & development.cell_id.isin(cell_ids) & development.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    holdout = holdout.loc[holdout.cell_id.isin(cell_ids) & holdout.drug_id.isin(drug_ids)].reset_index(drop=True)
    cell_reduced = fit_reducer(cell_matrix, cell_ids, set(source.cell_id), 64)[0]
    drug_reduced = fit_reducer(drug_matrix, drug_ids, set(source.drug_id), 64)[0]
    selected = pd.read_csv(args.selected_config).iloc[0]
    source_min, source_max = float(source.sensitivity.min()), float(source.sensitivity.max())
    theoretical_min, theoretical_max = -1.0, 0.0

    distance_path = args.output / "drug_ood_diagnostics.csv"
    if distance_path.is_file() and distance_path.stat().st_size > 0:
        drug_ood = pd.read_csv(distance_path, dtype={"drug_id": str, "nearest_source_drug_id": str})
    else:
        drug_ood = nearest_source_diagnostics(drug_reduced, drug_ids, set(source.drug_id))
        atomic_csv(drug_ood, distance_path)

    for study, target in holdout.groupby("study", sort=True):
        prediction_path = args.output / f"ensemble_predictions_{study}.csv.gz"
        if prediction_path.is_file() and prediction_path.stat().st_size > 0:
            continue
        target = target.reset_index(drop=True)
        prediction_sum = np.zeros(len(target), dtype=np.float64)
        prediction_sq_sum = np.zeros(len(target), dtype=np.float64)
        for seed in SEEDS:
            print(f"{study} seed {seed}", flush=True)
            model = InductiveMatrixFactorization(
                rank=int(selected["rank"]), regularization=float(selected["regularization"]),
                iterations=int(selected["iterations"]), projection_alpha=float(selected["projection_alpha"]),
                seed=seed,
            ).fit(source, cell_reduced, cell_ids, drug_reduced, drug_ids)
            chunks = []
            for start in range(0, len(target), args.chunk_size):
                chunks.append(model.predict(target.iloc[start : start + args.chunk_size][["cell_id", "drug_id"]]))
            prediction = np.concatenate(chunks)
            prediction_sum += prediction
            prediction_sq_sum += prediction * prediction
        mean_prediction = prediction_sum / len(SEEDS)
        prediction_sd = np.sqrt(np.maximum(prediction_sq_sum / len(SEEDS) - mean_prediction * mean_prediction, 0))
        output = target[["study", "cell_id", "drug_id", "sensitivity"]].copy()
        output["y_pred"] = mean_prediction
        output["seed_prediction_sd"] = prediction_sd
        atomic_csv(output, prediction_path)

    metric_tables, drug_tables, quartile_tables, association_rows = [], [], [], []
    for study in sorted(holdout.study.unique()):
        rows = pd.read_csv(
            args.output / f"ensemble_predictions_{study}.csv.gz",
            dtype={"cell_id": str, "drug_id": str},
        ).rename(columns={"sensitivity": "y_true"})
        rows = rows.merge(drug_ood, on="drug_id", how="left", validate="many_to_one")
        if rows.nearest_source_distance.isna().any():
            raise AssertionError("Missing OOD diagnostics for holdout drugs")
        variants = {
            "raw": rows.y_pred.to_numpy(),
            "theoretical_clip": np.clip(rows.y_pred.to_numpy(), theoretical_min, theoretical_max),
            "source_range_clip": np.clip(rows.y_pred.to_numpy(), source_min, source_max),
        }
        for variant, prediction in variants.items():
            frame = rows[["cell_id", "drug_id", "y_true"]].rename(columns={"y_true": "sensitivity"})
            metrics, evaluation = evaluate_variant(frame, prediction, variant)
            metrics.insert(0, "test_study", study)
            metric_tables.append(metrics)
            drugs = drug_summary(evaluation).merge(drug_ood, on="drug_id", validate="one_to_one")
            drugs.insert(0, "variant", variant)
            drugs.insert(0, "test_study", study)
            drug_tables.append(drugs)

        raw_drugs = drug_tables[-3]
        target_only = raw_drugs.loc[~raw_drugs.seen_in_source].copy()
        # Quartile boundaries are defined from target feature distances without labels.
        target_only["ood_quartile"] = pd.qcut(
            target_only.nearest_source_distance.rank(method="first"), 4,
            labels=("Q1_near", "Q2", "Q3", "Q4_far"),
        )
        quartile = target_only.groupby("ood_quartile", observed=True, as_index=False).agg(
            n_drugs=("drug_id", "nunique"), mean_distance=("nearest_source_distance", "mean"),
            macro_spearman=("spearman", "mean"), median_spearman=("spearman", "median"),
            mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
            mean_abs_prediction=("mean_abs_prediction", "mean"),
            mean_outside_fraction=("fraction_outside_auc_range", "mean"),
        )
        quartile.insert(0, "test_study", study)
        quartile_tables.append(quartile)
        for outcome in ("spearman", "mae", "rmse", "mean_abs_prediction", "fraction_outside_auc_range"):
            valid = target_only[["nearest_source_distance", outcome]].dropna()
            rho = spearmanr(valid.nearest_source_distance, valid[outcome]).statistic if len(valid) >= 2 else np.nan
            association_rows.append({
                "test_study": study, "population": "target_only_drugs", "distance": "nearest_source_distance",
                "outcome": outcome, "n_drugs": len(valid), "spearman": float(rho),
            })

    atomic_csv(pd.concat(metric_tables, ignore_index=True), args.output / "safeguard_metrics.csv")
    atomic_csv(pd.concat(drug_tables, ignore_index=True), args.output / "per_drug_ood_metrics.csv")
    atomic_csv(pd.concat(quartile_tables, ignore_index=True), args.output / "ood_quartiles.csv")
    atomic_csv(pd.DataFrame(association_rows), args.output / "ood_associations.csv")
    atomic_csv(pd.DataFrame([{
        "source_response_min": source_min, "source_response_max": source_max,
        "theoretical_min": theoretical_min, "theoretical_max": theoretical_max,
        "n_seeds": len(SEEDS), "seed_list": ";".join(map(str, SEEDS)),
    }]), args.output / "safeguard_definition.csv")
    print(pd.concat(metric_tables, ignore_index=True).to_string(index=False))
    print(pd.concat(quartile_tables, ignore_index=True).to_string(index=False))
    print(pd.DataFrame(association_rows).to_string(index=False))


if __name__ == "__main__":
    main()
