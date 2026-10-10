"""Dependency-free runner for the core test functions."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_core  # noqa: E402
import test_itens  # noqa: E402
import test_inductive_mf  # noqa: E402
import test_cbm_supplement  # noqa: E402
import test_two_tower  # noqa: E402


TESTS = (
    test_core.test_schema_rejects_duplicate_keys,
    test_core.test_entity_splits_are_disjoint,
    test_core.test_baseline_handles_unseen_entities,
    test_core.test_metrics_equal_weight_per_drug,
    test_itens.test_itens_ignores_target_labels_and_contributions_reconstruct,
    test_inductive_mf.test_inductive_mf_predicts_seen_and_unseen_entities_without_target_labels,
    test_cbm_supplement.test_scaffold_folds_keep_groups_disjoint_and_cover_each_drug_once,
    test_cbm_supplement.test_crossed_bootstrap_effect_is_paired_by_drug,
    test_cbm_supplement.test_weighted_spearman_matches_explicit_replication_with_ties,
    test_cbm_supplement.test_itens_grid_aliases_and_vectorized_macro_spearman,
    test_cbm_supplement.test_holm_stepdown_not_bonferroni,
    test_two_tower.test_two_tower_is_bounded_and_ignores_query_labels,
)


def main() -> None:
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(TESTS)} core tests")


if __name__ == "__main__":
    main()
