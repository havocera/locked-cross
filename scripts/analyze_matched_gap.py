"""Test the prespecified internal-to-external gap on matched entity domains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


RESAMPLES = 9_999
SEED = 20_261_004


def bca(values) -> tuple[float, float, float]:
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    ci = bootstrap(
        (values,), np.mean, method="BCa", n_resamples=RESAMPLES,
        confidence_level=0.95, random_state=SEED, batch=256,
    ).confidence_interval
    return float(values.mean()), float(ci.low), float(ci.high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    internal = pd.read_csv(args.internal, dtype={"drug_id": str})
    external = pd.read_csv(args.external, dtype={"drug_id": str})
    external = external.loc[external.model == "selected_inductive_mf"]
    rows = []
    regime_map = {
        "internal_random_overlap": "cross_study_overlap",
        "internal_random_exact_pair": "cross_study_exact_pair",
    }
    for (study, internal_regime), frame in internal.groupby(["test_study", "regime"], sort=True):
        external_regime = regime_map[internal_regime]
        averaged = (
            frame.groupby("drug_id", as_index=False)
            .agg(internal_spearman=("spearman", "mean"), n_internal_seeds=("seed", "nunique"))
        )
        target = external.loc[
            (external.test_study == study) & (external.regime == external_regime),
            ["drug_id", "spearman"],
        ].rename(columns={"spearman": "external_spearman"})
        paired = averaged.merge(target, on="drug_id").dropna()
        difference = paired.internal_spearman - paired.external_spearman
        effect, low, high = bca(difference)
        rows.append(
            {
                "test_study": study,
                "internal_regime": internal_regime,
                "external_regime": external_regime,
                "n_paired_drugs": len(paired),
                "mean_internal_spearman": paired.internal_spearman.mean(),
                "mean_external_spearman": paired.external_spearman.mean(),
                "gap_internal_minus_external": effect,
                "ci_low": low,
                "ci_high": high,
                "material_threshold": 0.10,
                "supports_C1": bool(low > 0.10),
                "bootstrap_method": "BCa drug cluster",
                "resamples": RESAMPLES,
                "seed": SEED,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
