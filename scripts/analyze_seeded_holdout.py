"""Recompute confirmation summaries after completing external MF seed repeats."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, permutation_test


RESAMPLES = 9_999
SEED = 20_261_004


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bca(values) -> tuple[float, float, float]:
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else np.nan, np.nan, np.nan)
    interval = bootstrap(
        (values,), np.mean, method="BCa", n_resamples=RESAMPLES,
        confidence_level=0.95, random_state=SEED, batch=256,
    ).confidence_interval
    return float(values.mean()), float(interval.low), float(interval.high)


def paired_test(difference: pd.Series) -> tuple[float, float, float, float]:
    difference = difference.dropna()
    effect, low, high = bca(difference)
    pvalue = float(permutation_test(
        (difference.to_numpy(),), np.mean, permutation_type="samples",
        alternative="two-sided", n_resamples=RESAMPLES, random_state=SEED, batch=256,
    ).pvalue) if len(difference) >= 2 else np.nan
    return effect, low, high, pvalue


def holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeded-scores", type=Path, required=True)
    parser.add_argument("--seeded-raw-scores", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--original-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mf = pd.read_csv(args.seeded_scores, dtype={"drug_id": str})
    mf["model"] = "selected_inductive_mf"
    raw_mf = pd.read_csv(args.seeded_raw_scores, dtype={"drug_id": str})
    original = pd.read_csv(args.original_scores, dtype={"drug_id": str})
    scores = pd.concat(
        [original.loc[original.model != "selected_inductive_mf"], mf],
        ignore_index=True,
    )

    summaries = []
    for keys, group in scores.groupby(["test_study", "regime", "model"], sort=True):
        effect, low, high = bca(group.spearman)
        values = group.spearman.dropna()
        summaries.append({
            "test_study": keys[0], "regime": keys[1], "model": keys[2],
            "n_eligible_drugs": len(values), "macro_spearman": effect,
            "ci_low": low, "ci_high": high,
            "median_spearman": float(values.median()) if len(values) else np.nan,
            "mf_seed_aggregation": "mean of per-seed per-drug Spearman" if keys[2] == "selected_inductive_mf" else "deterministic",
        })
    summary = pd.DataFrame(summaries)

    family1 = []
    primary = scores.loc[
        (scores.test_study == "GDSCv1") & (scores.regime == "cross_study_overlap")
    ].pivot(index="drug_id", columns="model", values="spearman")
    for candidate in sorted(set(primary.columns) - {"selected_inductive_mf"}):
        paired = primary[[candidate, "selected_inductive_mf"]].dropna()
        effect, low, high, pvalue = paired_test(paired[candidate] - paired.selected_inductive_mf)
        family1.append({
            "family": "candidate_vs_seed_averaged_locked_mf_on_GDSCv1_overlap",
            "test_study": "GDSCv1", "regime": "cross_study_overlap",
            "comparison": f"{candidate}_minus_selected_inductive_mf",
            "n_paired_drugs": len(paired), "effect": effect, "ci_low": low,
            "ci_high": high, "p_raw": pvalue,
        })
    family1 = pd.DataFrame(family1)
    family1["p_adjusted"] = holm(family1.p_raw.to_numpy())
    family1["adjustment"] = "Holm"
    family1["family_size"] = len(family1)

    family2 = []
    for study in ("GDSCv1", "NCI60"):
        wide = scores.loc[
            (scores.test_study == study) & (scores.regime == "cross_study_overlap")
        ].pivot(index="drug_id", columns="model", values="spearman")
        paired = wide[["itens_k16", "selected_inductive_mf"]].dropna()
        effect, low, high, pvalue = paired_test(paired.itens_k16 - paired.selected_inductive_mf)
        family2.append({
            "family": "ITENS_vs_seed_averaged_locked_MF_across_confirmation_screens",
            "test_study": study, "regime": "cross_study_overlap",
            "comparison": "itens_k16_minus_selected_inductive_mf",
            "n_paired_drugs": len(paired), "effect": effect, "ci_low": low,
            "ci_high": high, "p_raw": pvalue,
        })
    family2 = pd.DataFrame(family2)
    family2["p_adjusted"] = holm(family2.p_raw.to_numpy())
    family2["adjustment"] = "Holm"
    family2["family_size"] = len(family2)

    internal = pd.read_csv(args.internal, dtype={"drug_id": str})
    regime_map = {
        "internal_random_overlap": "cross_study_overlap",
        "internal_random_exact_pair": "cross_study_exact_pair",
    }
    gaps = []
    for (study, internal_regime), frame in internal.groupby(["test_study", "regime"], sort=True):
        external_regime = regime_map[internal_regime]
        internal_mean = (
            frame.groupby("drug_id", as_index=False)
            .agg(internal_spearman=("spearman", "mean"), n_internal_seeds=("seed", "nunique"))
        )
        external_mean = mf.loc[
            (mf.test_study == study) & (mf.regime == external_regime),
            ["drug_id", "spearman", "n_seeds"],
        ].rename(columns={"spearman": "external_spearman", "n_seeds": "n_external_seeds"})
        paired = internal_mean.merge(external_mean, on="drug_id").dropna()
        effect, low, high = bca(paired.internal_spearman - paired.external_spearman)
        gaps.append({
            "test_study": study, "internal_regime": internal_regime,
            "external_regime": external_regime, "n_paired_drugs": len(paired),
            "mean_internal_spearman": paired.internal_spearman.mean(),
            "mean_external_spearman": paired.external_spearman.mean(),
            "gap_internal_minus_external": effect, "ci_low": low, "ci_high": high,
            "material_threshold": 0.10, "supports_C1": bool(low > 0.10),
            "bootstrap_method": "BCa drug cluster", "resamples": RESAMPLES, "seed": SEED,
            "internal_seed_aggregation": "mean of per-seed per-drug Spearman",
            "external_seed_aggregation": "mean of per-seed per-drug Spearman",
        })
    gaps = pd.DataFrame(gaps)

    seed_variability = (
        raw_mf.dropna(subset=["spearman"])
        .groupby(["seed", "test_study", "regime"], as_index=False)
        .agg(macro_spearman=("spearman", "mean"), n_eligible_drugs=("drug_id", "nunique"))
        .groupby(["test_study", "regime"], as_index=False)
        .agg(n_seeds=("seed", "nunique"), seed_mean=("macro_spearman", "mean"),
             seed_sd=("macro_spearman", "std"), seed_min=("macro_spearman", "min"),
             seed_max=("macro_spearman", "max"))
    )
    diagnostics = pd.read_csv(args.diagnostics)
    extrapolation = (
        diagnostics.groupby(["test_study", "regime"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"), prediction_min=("prediction_min", "min"),
            mean_q001=("prediction_q001", "mean"), mean_q01=("prediction_q01", "mean"),
            mean_median=("prediction_median", "mean"), mean_q99=("prediction_q99", "mean"),
            mean_q999=("prediction_q999", "mean"), prediction_max=("prediction_max", "max"),
            mean_fraction_outside_auc_range=("fraction_outside_auc_range", "mean"),
            mean_fraction_outside_source_range=("fraction_outside_source_range", "mean"),
        )
    )

    atomic_csv(summary, args.output / "seeded_holdout_summary.csv")
    atomic_csv(pd.concat([family1, family2], ignore_index=True), args.output / "seeded_comparisons.csv")
    atomic_csv(gaps, args.output / "seeded_matched_gap.csv")
    atomic_csv(seed_variability, args.output / "seeded_macro_variability.csv")
    atomic_csv(extrapolation, args.output / "seeded_extrapolation_summary.csv")
    print(gaps.to_string(index=False))
    print(family2.to_string(index=False))
    print(seed_variability.to_string(index=False))
    print(extrapolation.to_string(index=False))


if __name__ == "__main__":
    main()
