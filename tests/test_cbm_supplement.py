"""Focused invariants for the CBM supplementary analyses."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_crossed_bootstrap import DrugGroup, correlation, paired_effect, rank_plan  # noqa: E402
from run_scaffold_sensitivity import scaffold_folds  # noqa: E402
from tune_itens_cached import alias_ledger, score_predictions  # noqa: E402
from evaluate_tuned_itens import holm_adjust


def test_scaffold_folds_keep_groups_disjoint_and_cover_each_drug_once():
    source = pd.DataFrame({
        "drug_id": ["d1", "d1", "d2", "d3", "d4", "d5", "d6"],
        "cell_id": ["c1", "c2", "c1", "c1", "c1", "c1", "c1"],
    })
    scaffolds = {"d1": "s1", "d2": "s1", "d3": "s2", "d4": "s3", "d5": "s4", "d6": "s5"}
    seen = []
    for _, train, test, test_scaffolds in scaffold_folds(source, scaffolds, seed=104729, folds=3):
        assert train.isdisjoint(test)
        assert {scaffolds[drug] for drug in train}.isdisjoint(test_scaffolds)
        seen.extend(test)
    assert sorted(seen) == sorted(scaffolds)


def test_crossed_bootstrap_effect_is_paired_by_drug():
    cells = np.arange(5)
    observed = np.arange(5, dtype=float)
    positive = DrugGroup(cells, rank_plan(observed), rank_plan(observed))
    negative = DrugGroup(cells, rank_plan(observed), rank_plan(-observed))
    first = {"d1": positive, "d2": positive}
    second = {"d1": negative, "d2": negative}
    effect = paired_effect(first, second, Counter({"d1": 1, "d2": 2}))
    assert np.isclose(effect, 2.0)
    weighted = correlation(first["d1"], np.asarray([2, 0, 1, 1, 1]))
    assert np.isclose(weighted, 1.0)


def test_weighted_spearman_matches_explicit_replication_with_ties():
    observed = np.asarray([1.0, 1.0, 2.0, 3.0])
    predicted = np.asarray([3.0, 2.0, 1.0, 1.0])
    weights = np.asarray([2, 0, 1, 3])
    group = DrugGroup(np.arange(4), rank_plan(observed), rank_plan(predicted))
    expected = spearmanr(np.repeat(observed, weights), np.repeat(predicted, weights)).statistic
    assert np.isclose(correlation(group, weights), expected)


def test_itens_grid_aliases_and_vectorized_macro_spearman():
    aliases = alias_ledger()
    assert len(aliases) == 3942
    assert aliases.canonical_config_id.nunique() == 3213
    frame = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(5)] * 2,
        "drug_id": ["d1"] * 5 + ["d2"] * 5,
        "sensitivity": np.r_[np.arange(5), np.arange(5)],
    })
    prediction = np.column_stack([frame.sensitivity, -frame.sensitivity])
    scores = score_predictions(frame, prediction, ["positive", "negative"], [{}, {}])
    assert np.allclose(scores.macro_spearman, [1.0, -1.0])


def test_holm_stepdown_not_bonferroni():
    assert np.allclose(holm_adjust(np.array([0.0062, 0.7644])), [0.0124, 0.7644])
    assert np.allclose(holm_adjust(np.array([0.03, 0.01, 0.02])), [0.04, 0.03, 0.04])
