"""Confirmatory drug-level inference for locked GDSCv1/NCI60 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, permutation_test, spearmanr


RESAMPLES = 9_999
SEED = 20_261_004


def bca(values) -> tuple[float, float, float]:
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        return (float(values.mean()) if len(values) else np.nan, np.nan, np.nan)
    ci = bootstrap(
        (values,), np.mean, method="BCa", n_resamples=RESAMPLES,
        confidence_level=0.95, random_state=SEED, batch=256,
    ).confidence_interval
    return float(values.mean()), float(ci.low), float(ci.high)


def holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def per_drug_observed(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, group in frame.groupby("drug_id", sort=True):
        value = np.nan
        if group.cell_id.nunique() >= 5 and group.fit_auc_source.nunique() >= 2 and group.fit_auc_target.nunique() >= 2:
            value = float(spearmanr(group.fit_auc_source, group.fit_auc_target).statistic)
        rows.append({"drug_id": drug, "n_cells": group.cell_id.nunique(), "spearman": value})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--development-responses", type=Path, required=True)
    parser.add_argument("--holdout-responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores, dtype={"drug_id": str})

    summaries = []
    for keys, group in scores.groupby(["test_study", "regime", "model"], sort=True):
        values = group.spearman.dropna().to_numpy(dtype=float)
        mean, low, high = bca(values)
        summaries.append(
            {
                "test_study": keys[0], "regime": keys[1], "model": keys[2],
                "n_eligible_drugs": len(values), "macro_spearman": mean,
                "ci_low": low, "ci_high": high,
                "median_spearman": float(np.median(values)) if len(values) else np.nan,
            }
        )
    summary = pd.DataFrame(summaries)

    # Family 1: every candidate versus the locked selected MF on primary GDSCv1.
    family1 = []
    primary = scores.loc[
        (scores.test_study == "GDSCv1") & (scores.regime == "cross_study_overlap")
    ].pivot(index="drug_id", columns="model", values="spearman")
    for candidate in sorted(set(primary.columns) - {"selected_inductive_mf"}):
        paired = primary[[candidate, "selected_inductive_mf"]].dropna()
        difference = paired[candidate] - paired.selected_inductive_mf
        effect, low, high = bca(difference)
        pvalue = float(permutation_test(
            (difference.to_numpy(),), lambda x: np.mean(x), permutation_type="samples",
            alternative="two-sided", n_resamples=RESAMPLES, random_state=SEED,
            batch=256,
        ).pvalue) if len(difference) >= 2 else np.nan
        family1.append(
            {
                "family": "candidate_vs_locked_mf_on_GDSCv1_overlap",
                "test_study": "GDSCv1", "regime": "cross_study_overlap",
                "comparison": f"{candidate}_minus_selected_inductive_mf",
                "n_paired_drugs": len(paired), "effect": effect,
                "ci_low": low, "ci_high": high, "p_raw": pvalue,
            }
        )
    family1 = pd.DataFrame(family1)
    valid = family1.p_raw.notna()
    family1["p_adjusted"] = np.nan
    family1.loc[valid, "p_adjusted"] = holm(family1.loc[valid, "p_raw"].to_numpy())
    family1["adjustment"] = "Holm"
    family1["family_size"] = int(valid.sum())

    # Family 2 is frozen explicitly in the protocol: ITENS vs strongest baseline
    # (the locked MF) on both confirmation screens.
    family2 = []
    for study in ("GDSCv1", "NCI60"):
        wide = scores.loc[
            (scores.test_study == study) & (scores.regime == "cross_study_overlap")
        ].pivot(index="drug_id", columns="model", values="spearman")
        paired = wide[["itens_k16", "selected_inductive_mf"]].dropna()
        difference = paired.itens_k16 - paired.selected_inductive_mf
        effect, low, high = bca(difference)
        pvalue = float(permutation_test(
            (difference.to_numpy(),), lambda x: np.mean(x), permutation_type="samples",
            alternative="two-sided", n_resamples=RESAMPLES, random_state=SEED,
            batch=256,
        ).pvalue) if len(difference) >= 2 else np.nan
        family2.append(
            {
                "family": "ITENS_vs_locked_MF_across_confirmation_screens",
                "test_study": study, "regime": "cross_study_overlap",
                "comparison": "itens_k16_minus_selected_inductive_mf",
                "n_paired_drugs": len(paired), "effect": effect,
                "ci_low": low, "ci_high": high, "p_raw": pvalue,
            }
        )
    family2 = pd.DataFrame(family2)
    valid = family2.p_raw.notna()
    family2["p_adjusted"] = np.nan
    family2.loc[valid, "p_adjusted"] = holm(family2.loc[valid, "p_raw"].to_numpy())
    family2["adjustment"] = "Holm"
    family2["family_size"] = int(valid.sum())

    development = pd.read_csv(args.development_responses, dtype={"cell_id": str, "drug_id": str})
    holdout = pd.read_csv(args.holdout_responses, dtype={"cell_id": str, "drug_id": str})
    source = development.loc[development.study == "CTRPv2", ["cell_id", "drug_id", "fit_auc"]].rename(
        columns={"fit_auc": "fit_auc_source"}
    )
    concordance = []
    for study, target in holdout.groupby("study", sort=True):
        shared = source.merge(
            target[["cell_id", "drug_id", "fit_auc"]].rename(columns={"fit_auc": "fit_auc_target"}),
            on=["cell_id", "drug_id"], validate="one_to_one",
        )
        drug = per_drug_observed(shared)
        mean, low, high = bca(drug.spearman)
        concordance.append(
            {
                "test_study": study, "n_exact_pairs": len(shared),
                "n_cells": shared.cell_id.nunique(), "n_drugs": shared.drug_id.nunique(),
                "n_eligible_drugs": drug.spearman.notna().sum(),
                "observed_macro_spearman": mean, "ci_low": low, "ci_high": high,
            }
        )

    summary.to_csv(args.output / "holdout_summary.csv", index=False)
    pd.concat([family1, family2], ignore_index=True).to_csv(
        args.output / "holdout_comparisons.csv", index=False
    )
    pd.DataFrame(concordance).to_csv(args.output / "holdout_observed_concordance.csv", index=False)
    print(summary.to_string(index=False))
    print(family1.to_string(index=False))
    print(family2.to_string(index=False))
    print(pd.DataFrame(concordance).to_string(index=False))


if __name__ == "__main__":
    main()
