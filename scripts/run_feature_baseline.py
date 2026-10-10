"""Run expression/descriptor baselines without target-label preprocessing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:  # LightGBM is an optional baseline dependency.
    LGBMRegressor = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossdb_drug_response.baseline import AdditiveBiasRegressor  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from crossdb_drug_response.splits import make_folds  # noqa: E402


SEED = 104729
PCA_COMPONENTS = 64
INTERACTION_COMPONENTS = 16


def load_matrix(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(directory / f"{name}.npz")["values"]
    entities = pd.read_csv(directory / f"{name}_entities.csv", dtype=str)[
        "entity_id"
    ].to_numpy()
    if len(values) != len(entities):
        raise ValueError(f"Entity count mismatch for {name}")
    return values, entities


def fit_reducer(
    matrix: np.ndarray,
    entity_ids: np.ndarray,
    source_entities: set[str],
    components: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    source_mask = np.isin(entity_ids, list(source_entities))
    if not source_mask.any():
        raise ValueError("No source entities found in feature matrix")
    source = matrix[source_mask].astype(np.float64)
    valid = np.isfinite(source).any(axis=0)
    medians = np.nanmedian(source[:, valid], axis=0)
    source = source[:, valid]
    source = np.where(np.isnan(source), medians, source)
    variable = np.nanstd(source, axis=0) > 0
    source = source[:, variable]
    medians = medians[variable]

    scaler = StandardScaler().fit(source)
    source_scaled = scaler.transform(source)
    n_components = min(components, source_scaled.shape[0] - 1, source_scaled.shape[1])
    reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
    reducer.fit(source_scaled)

    all_values = matrix[:, valid][:, variable].astype(np.float64)
    all_values = np.where(np.isnan(all_values), medians, all_values)
    reduced = reducer.transform(scaler.transform(all_values)).astype(np.float32)
    entity_lookup = {entity: index for index, entity in enumerate(entity_ids)}
    audit = {
        "input_features": int(matrix.shape[1]),
        "numeric_nonconstant_source_features": int(variable.sum()),
        "pca_components": int(n_components),
        "source_entities": int(source_mask.sum()),
    }
    return reduced, entity_lookup, audit


def pair_features(
    frame: pd.DataFrame,
    cell_reduced: np.ndarray,
    cell_lookup: dict[str, int],
    drug_reduced: np.ndarray,
    drug_lookup: dict[str, int],
) -> np.ndarray:
    cell_index = np.fromiter((cell_lookup[x] for x in frame.cell_id), dtype=int)
    drug_index = np.fromiter((drug_lookup[x] for x in frame.drug_id), dtype=int)
    cell = cell_reduced[cell_index]
    drug = drug_reduced[drug_index]
    rank = min(INTERACTION_COMPONENTS, cell.shape[1], drug.shape[1])
    interactions = (cell[:, :rank, None] * drug[:, None, :rank]).reshape(len(frame), -1)
    return np.concatenate([cell, drug, interactions], axis=1).astype(np.float32)


def make_model(name: str):
    if name == "ridge_bilinear":
        return Ridge(alpha=10.0, solver="lsqr", tol=1e-4)
    if name == "histgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            max_bins=128,
            early_stopping=False,
            random_state=SEED,
        )
    if name == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm==4.6.0 is required for the LightGBM baseline")
        return LGBMRegressor(
            objective="regression_l2",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=8,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    if name == "additive_matched":
        return AdditiveBiasRegressor()
    raise ValueError(f"Unknown model: {name}")


def predict_chunks(model, frame: pd.DataFrame, feature_builder, chunk_size: int = 100_000):
    outputs = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        outputs.append(model.predict(feature_builder(chunk)))
    return np.concatenate(outputs)


def prediction_table(frame: pd.DataFrame, values: np.ndarray, model: str, regime: str, fold: int) -> pd.DataFrame:
    result = frame[["cell_id", "drug_id", "fit_r2"]].copy()
    result["y_true"] = frame.sensitivity.to_numpy()
    result["y_pred"] = values
    result.insert(0, "model", model)
    result.insert(0, "seed", SEED)
    result.insert(0, "fold", fold)
    result.insert(0, "regime", regime)
    result.insert(0, "test_study", "CTRPv2" if regime == "random_pair" else "PRISM")
    result.insert(0, "train_study", "CTRPv2")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["additive_matched", "ridge_bilinear", "histgb", "lightgbm"],
        choices=["additive_matched", "ridge_bilinear", "histgb", "lightgbm"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    responses = pd.read_csv(args.responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_entities = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_entities = load_matrix(args.features, "drug_descriptors")
    available_cells = set(cell_entities)
    available_drugs = set(drug_entities)
    complete = responses.loc[
        responses.cell_id.isin(available_cells) & responses.drug_id.isin(available_drugs)
    ].reset_index(drop=True)
    source = complete.loc[complete.study == "CTRPv2"].reset_index(drop=True)
    target = complete.loc[complete.study == "PRISM"].reset_index(drop=True)
    shared_cells = set(source.cell_id) & set(target.cell_id)
    shared_drugs = set(source.drug_id) & set(target.drug_id)
    target_overlap = target.loc[
        target.cell_id.isin(shared_cells) & target.drug_id.isin(shared_drugs)
    ].reset_index(drop=True)

    cell_reduced, cell_lookup, cell_audit = fit_reducer(
        cell_matrix, cell_entities, set(source.cell_id), PCA_COMPONENTS
    )
    drug_reduced, drug_lookup, drug_audit = fit_reducer(
        drug_matrix, drug_entities, set(source.drug_id), PCA_COMPONENTS
    )
    audit = pd.DataFrame(
        [
            {"item": "cell_reducer", **cell_audit},
            {"item": "drug_reducer", **drug_audit},
            {
                "item": "feature_complete_responses",
                "input_features": np.nan,
                "numeric_nonconstant_source_features": np.nan,
                "pca_components": np.nan,
                "source_entities": len(source),
                "target_pairs": len(target),
                "target_overlap_pairs": len(target_overlap),
                "excluded_response_pairs": len(responses) - len(complete),
            },
        ]
    )
    audit.to_csv(args.output / "feature_model_audit.csv", index=False)

    def build(frame):
        return pair_features(frame, cell_reduced, cell_lookup, drug_reduced, drug_lookup)

    predictions = []
    folds = make_folds(source, "random_pair", folds=5, seed=SEED)
    for model_name in args.models:
        for split in folds:
            train = source.iloc[split.train_index]
            test = source.iloc[split.test_index]
            model = make_model(model_name)
            if model_name == "additive_matched":
                model.fit(train)
                values = model.predict(test)
            else:
                model.fit(build(train), train.sensitivity.to_numpy())
                values = model.predict(build(test))
            predictions.append(
                prediction_table(test, values, model_name, "random_pair", split.fold)
            )

        model = make_model(model_name)
        if model_name == "additive_matched":
            model.fit(source)
            target_values = model.predict(target)
            overlap_values = model.predict(target_overlap)
        else:
            model.fit(build(source), source.sensitivity.to_numpy())
            target_values = predict_chunks(model, target, build)
            overlap_values = predict_chunks(model, target_overlap, build)
        predictions.append(
            prediction_table(target, target_values, model_name, "cross_study", -1)
        )
        predictions.append(
            prediction_table(
                target_overlap,
                overlap_values,
                model_name,
                "cross_study_overlap",
                -1,
            )
        )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_rows = []
    group_columns = ["model", "regime", "seed"]
    for keys, group in prediction_frame.groupby(group_columns, sort=True):
        metrics = evaluate_predictions(group)
        for column, value in zip(group_columns, keys):
            metrics.insert(0, column, value)
        metric_rows.append(metrics)
    metrics = pd.concat(metric_rows, ignore_index=True)
    prediction_frame.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    print(audit.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
