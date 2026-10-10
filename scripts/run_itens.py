"""Run the leakage-safe ITENS cross-study baseline and export explanations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossdb_drug_response.itens import ITENSRegressor  # noqa: E402
from crossdb_drug_response.metrics import evaluate_predictions  # noqa: E402
from run_feature_baseline import fit_reducer, load_matrix  # noqa: E402


SEED = 104729


def make_prediction_table(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    diagnostics: pd.DataFrame,
    regime: str,
    test_study: str,
) -> pd.DataFrame:
    output = frame[["cell_id", "drug_id", "fit_r2"]].reset_index(drop=True).copy()
    output["y_true"] = frame.sensitivity.to_numpy()
    output["y_pred"] = prediction
    output = pd.concat([output, diagnostics.reset_index(drop=True)], axis=1)
    output.insert(0, "model", "itens_k16")
    output.insert(0, "seed", SEED)
    output.insert(0, "fold", -1)
    output.insert(0, "regime", regime)
    output.insert(0, "test_study", test_study)
    output.insert(0, "train_study", "CTRPv2")
    return output


def explanation_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    ordered = frame.sort_values(["drug_id", "cell_id"], ignore_index=True)
    if len(ordered) <= n:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, n, dtype=int)
    return ordered.iloc[positions].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explanation-sample", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    responses = pd.read_csv(args.responses, dtype={"cell_id": str, "drug_id": str})
    cell_matrix, cell_ids = load_matrix(args.features, "cell_expression")
    drug_matrix, drug_ids = load_matrix(args.features, "drug_descriptors")
    complete = responses.loc[
        responses.cell_id.isin(cell_ids) & responses.drug_id.isin(drug_ids)
    ].reset_index(drop=True)
    source = complete.loc[complete.study == "CTRPv2"].reset_index(drop=True)
    target = complete.loc[complete.study == "PRISM"].reset_index(drop=True)
    shared_cells = set(source.cell_id) & set(target.cell_id)
    shared_drugs = set(source.drug_id) & set(target.drug_id)
    overlap = target.loc[
        target.cell_id.isin(shared_cells) & target.drug_id.isin(shared_drugs)
    ].reset_index(drop=True)

    cell_reduced, _, cell_audit = fit_reducer(
        cell_matrix, cell_ids, set(source.cell_id), components=64
    )
    drug_reduced, _, drug_audit = fit_reducer(
        drug_matrix, drug_ids, set(source.drug_id), components=64
    )
    model = ITENSRegressor().fit(
        source,
        cell_reduced,
        cell_ids,
        drug_reduced,
        drug_ids,
    )

    outputs = []
    for regime, frame in (("cross_study", target), ("cross_study_overlap", overlap)):
        values, diagnostics = model.predict(frame, return_diagnostics=True)
        outputs.append(make_prediction_table(frame, values, diagnostics, regime, "PRISM"))
    predictions = pd.concat(outputs, ignore_index=True)
    metrics = []
    for regime, group in predictions.groupby("regime", sort=True):
        table = evaluate_predictions(group)
        table.insert(0, "regime", regime)
        table.insert(0, "model", "itens_k16")
        metrics.append(table)
    metrics = pd.concat(metrics, ignore_index=True)

    explanation_rows = []
    explanation_audit = []
    rng = np.random.default_rng(SEED)
    sampled = explanation_sample(overlap, args.explanation_sample)
    direct = model.predict(sampled[["cell_id", "drug_id"]])
    for position, row in enumerate(sampled.itertuples(index=False)):
        explained, contributions = model.explain_one(row.cell_id, row.drug_id)
        error = abs(explained - direct[position])
        if error > 1e-8:
            raise RuntimeError(f"Contribution reconstruction failed by {error}")
        if contributions.empty:
            top = contributions
            total_signed = total_absolute = omitted_signed = omitted_absolute = 0.0
        else:
            top = contributions.head(20).copy()
            total_signed = float(contributions.signed_attribution.sum())
            total_absolute = float(contributions.absolute_attribution.sum())
            omitted_signed = float(contributions.iloc[20:].signed_attribution.sum())
            omitted_absolute = float(contributions.iloc[20:].absolute_attribution.sum())
            top.insert(0, "drug_id", row.drug_id)
            top.insert(0, "cell_id", row.cell_id)
            top.insert(0, "test_study", "PRISM")
            top.insert(0, "train_study", "CTRPv2")
            top["explanation_method"] = "itens_exact_training_edge_contribution"
            explanation_rows.append(top)
        edge_ids = list(zip(contributions.neighbor_cell_id, contributions.neighbor_drug_id))
        top_edges = set(edge_ids[:20])
        top_deleted, _ = model.explain_one(row.cell_id, row.drug_id, excluded_edges=top_edges)
        random_deltas = []
        if len(edge_ids) >= len(top_edges) and top_edges:
            for _ in range(20):
                selected = rng.choice(len(edge_ids), size=len(top_edges), replace=False)
                random_edges = {edge_ids[index] for index in selected}
                random_deleted, _ = model.explain_one(
                    row.cell_id, row.drug_id, excluded_edges=random_edges
                )
                random_deltas.append(abs(direct[position] - random_deleted))
        explanation_audit.append(
            {
                "cell_id": row.cell_id,
                "drug_id": row.drug_id,
                "prediction": direct[position],
                "global_mean": model.global_mean_,
                "reconstructed_prediction": explained,
                "reconstruction_error": error,
                "n_contributing_edges": len(contributions),
                "total_signed_attribution": total_signed,
                "total_absolute_attribution": total_absolute,
                "omitted_top20_signed_attribution": omitted_signed,
                "omitted_top20_absolute_attribution": omitted_absolute,
                "top20_deletion_absolute_delta": abs(direct[position] - top_deleted),
                "random20_deletion_absolute_delta_mean": float(np.mean(random_deltas)) if random_deltas else 0.0,
            }
        )

    explanations = pd.concat(explanation_rows, ignore_index=True) if explanation_rows else pd.DataFrame()
    audit = pd.DataFrame(
        [
            {"item": "cell_reducer", **cell_audit},
            {"item": "drug_reducer", **drug_audit},
            {
                "item": "itens",
                "k_cell": model.k_cell,
                "k_drug": model.k_drug,
                "temperature_cell": model.temperature_cell,
                "temperature_drug": model.temperature_drug,
                "ess_shrinkage": model.ess_shrinkage,
                "theta_cell": model.theta[0],
                "theta_drug": model.theta[1],
                "theta_product": model.theta[2],
                "n_training_edges": len(model.training_edges_),
            },
        ]
    )
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    explanations.to_csv(args.output / "explanations_top20.csv", index=False)
    pd.DataFrame(explanation_audit).to_csv(args.output / "explanation_audit.csv", index=False)
    audit.to_csv(args.output / "model_audit.csv", index=False)
    print(metrics.to_string(index=False))
    print(pd.DataFrame(explanation_audit).describe(include="all").to_string())


if __name__ == "__main__":
    main()
