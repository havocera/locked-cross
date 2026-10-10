"""Command-line runner for the additive baseline benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baseline import AdditiveBiasRegressor
from .metrics import evaluate_predictions
from .schema import validate_responses
from .splits import make_folds


INTERNAL_REGIMES = ("random_pair", "cold_cell", "cold_drug", "cold_both")


def _prediction_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    regime: str,
    fold: int,
    seed: int,
    train_study: str,
    test_study: str,
) -> pd.DataFrame:
    model = AdditiveBiasRegressor().fit(train)
    result = test[["cell_id", "drug_id"]].copy()
    result["y_true"] = test["sensitivity"].to_numpy()
    result["y_pred"] = model.predict(test)
    result.insert(0, "model", "additive_bias")
    result.insert(0, "seed", seed)
    result.insert(0, "fold", fold)
    result.insert(0, "regime", regime)
    result.insert(0, "test_study", test_study)
    result.insert(0, "train_study", train_study)
    return result


def run_benchmark(
    responses: pd.DataFrame,
    train_study: str,
    test_study: str,
    seeds: list[int],
    folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validate_responses(responses)
    source = frame.loc[frame["study"] == train_study].reset_index(drop=True)
    target = frame.loc[frame["study"] == test_study].reset_index(drop=True)
    if source.empty:
        raise ValueError(f"No rows found for training study {train_study}")
    if target.empty:
        raise ValueError(f"No rows found for test study {test_study}")

    predictions = []
    for seed in seeds:
        for regime in INTERNAL_REGIMES:
            for split in make_folds(source, regime, folds=folds, seed=seed):
                predictions.append(
                    _prediction_rows(
                        source.iloc[split.train_index],
                        source.iloc[split.test_index],
                        regime,
                        split.fold,
                        seed,
                        train_study,
                        train_study,
                    )
                )

        predictions.append(
            _prediction_rows(
                source,
                target,
                "cross_study",
                -1,
                seed,
                train_study,
                test_study,
            )
        )

        overlap_target = target.loc[
            target["cell_id"].isin(source["cell_id"])
            & target["drug_id"].isin(source["drug_id"])
        ]
        if not overlap_target.empty:
            predictions.append(
                _prediction_rows(
                    source,
                    overlap_target,
                    "cross_study_overlap",
                    -1,
                    seed,
                    train_study,
                    test_study,
                )
            )

    prediction_table = pd.concat(predictions, ignore_index=True)
    metric_tables = []
    group_columns = ["train_study", "test_study", "regime", "fold", "seed", "model"]
    for keys, group in prediction_table.groupby(group_columns, sort=True):
        metrics = evaluate_predictions(group)
        for column, value in zip(group_columns, keys):
            metrics.insert(0, column, value)
        metric_tables.append(metrics)
    internal = prediction_table.loc[prediction_table["fold"] >= 0]
    aggregate_columns = ["train_study", "test_study", "regime", "seed", "model"]
    for keys, group in internal.groupby(aggregate_columns, sort=True):
        metrics = evaluate_predictions(group)
        metrics.insert(0, "fold", -2)
        for column, value in zip(aggregate_columns, keys):
            metrics.insert(0, column, value)
        metric_tables.append(metrics)
    metric_table = pd.concat(metric_tables, ignore_index=True)
    return prediction_table, metric_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--train-study", required=True)
    parser.add_argument("--test-study", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[104729])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses = pd.read_csv(args.responses)
    predictions, metrics = run_benchmark(
        responses,
        train_study=args.train_study,
        test_study=args.test_study,
        seeds=args.seeds,
        folds=args.folds,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
