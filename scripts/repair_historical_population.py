"""B1 historical-population sensitivity; preserves original inputs and results.

Run --prepare to audit inputs, no flag to train/resume, --verify to validate final
artifacts in a fresh process. No target response is used to select parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'

import joblib
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, norm, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization
from crossdb_drug_response.splits import make_folds
from run_feature_baseline import load_matrix, fit_reducer
from run_seeded_holdout_mf import SEEDS, atomic_csv, drug_scores

OUT = ROOT / 'results/repair_historical_population'
CONFIG = dict(rank=64, regularization=1.0, iterations=30, projection_alpha=10.0)
KEYS = ['seed', 'test_study', 'regime', 'domain', 'drug_id']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def responses(path):
    return pd.read_csv(path, usecols=['study', 'cell_id', 'drug_id', 'sensitivity'],
                       dtype={'cell_id': str, 'drug_id': str})


def population(frame):
    return dict(n_pairs=len(frame), n_cells=frame.cell_id.nunique(), n_drugs=frame.drug_id.nunique())


def inputs():
    historical_dir = ROOT / 'data/processed/features'
    target_dir = ROOT / 'data/holdout_features'
    expanded_dir = ROOT / 'data/processed_multistudy/features'
    mats, ids, feature_checks = {}, {}, []
    for name in ('cell_expression', 'drug_descriptors'):
        old_columns = pd.read_csv(historical_dir / f'{name}_features.csv', dtype=str)
        new_columns = pd.read_csv(target_dir / f'{name}_features.csv', dtype=str)
        pd.testing.assert_frame_equal(old_columns, new_columns)
        old, old_ids = load_matrix(historical_dir, name)
        new, new_ids = load_matrix(target_dir, name)
        assert old.shape[1] == new.shape[1] == len(old_columns)
        assert len(set(old_ids)) == len(old_ids) and len(set(new_ids)) == len(new_ids)
        # Historical rows always win. Target-only rows are transformed, never fit.
        extra = ~np.isin(new_ids, old_ids)
        mats[name] = np.concatenate([old, new[extra]], axis=0)
        ids[name] = np.concatenate([old_ids, new_ids[extra]])
        feature_checks.append(dict(feature=name, same_ordered_columns=True,
                                   historical_entities=len(old_ids), target_only_entities=int(extra.sum())))
    historical_raw = responses(ROOT / 'data/confirmatory/responses.csv')
    historical = historical_raw.loc[(historical_raw.study == 'CTRPv2') &
        historical_raw.cell_id.isin(load_matrix(historical_dir, 'cell_expression')[1]) &
        historical_raw.drug_id.isin(load_matrix(historical_dir, 'drug_descriptors')[1])].reset_index(drop=True)
    assert population(historical) == dict(n_pairs=293194, n_cells=819, n_drugs=451), population(historical)
    assert not historical.duplicated(['cell_id', 'drug_id']).any()
    assert np.isfinite(historical.sensitivity).all()
    expanded_raw = responses(ROOT / 'data/processed_multistudy/responses.csv')
    expanded = expanded_raw.loc[(expanded_raw.study == 'CTRPv2') &
        expanded_raw.cell_id.isin(load_matrix(expanded_dir, 'cell_expression')[1]) &
        expanded_raw.drug_id.isin(load_matrix(expanded_dir, 'drug_descriptors')[1])].reset_index(drop=True)
    assert population(expanded) == dict(n_pairs=300507, n_cells=822, n_drugs=459)
    comparison = historical.merge(expanded, on=['study', 'cell_id', 'drug_id'], suffixes=('_historical', '_expanded'), validate='one_to_one')
    assert len(comparison) == len(historical)
    assert np.array_equal(comparison.sensitivity_historical, comparison.sensitivity_expanded)
    target_raw = responses(ROOT / 'data/holdout_processed/responses.csv')
    targets = target_raw.loc[target_raw.cell_id.isin(load_matrix(target_dir, 'cell_expression')[1]) &
                             target_raw.drug_id.isin(load_matrix(target_dir, 'drug_descriptors')[1])].reset_index(drop=True)
    assert set(targets.study) == {'GDSCv1', 'NCI60'}
    assert not targets.duplicated(['study', 'cell_id', 'drug_id']).any()
    lock = json.loads((ROOT / 'results/model_lock.json').read_text())
    assert all(lock['selected_configuration'][k] == v for k, v in CONFIG.items())
    for key, relative in [('grid_sha256', 'results/tuning_inductive_mf/selected_config.csv'),
                          ('ten_seed_summary_sha256', 'results/internal_seeds/selection_summary.csv'),
                          ('confirmatory_protocol_sha256', 'confirmatory_protocol_v1.md')]:
        assert digest(ROOT / relative) == lock[key], key
    stages = [dict(stage='historical_raw_source', study='CTRPv2', **population(historical_raw.loc[historical_raw.study == 'CTRPv2'])),
              dict(stage='historical_feature_complete_source', study='CTRPv2', **population(historical)),
              dict(stage='current_expanded_feature_complete_source', study='CTRPv2', **population(expanded))]
    frames = {}
    source_pairs = pd.MultiIndex.from_frame(historical[['cell_id', 'drug_id']])
    for study in ('GDSCv1', 'NCI60'):
        raw = target_raw.loc[target_raw.study == study]
        target = targets.loc[targets.study == study].reset_index(drop=True)
        stages.extend([dict(stage='target_raw', study=study, **population(raw)),
                       dict(stage='target_feature_complete', study=study, **population(target))])
        for domain in ('overlap', 'exact_pair'):
            if domain == 'overlap':
                ext = target.loc[target.cell_id.isin(historical.cell_id) & target.drug_id.isin(historical.drug_id)]
                internal = historical.loc[historical.cell_id.isin(ext.cell_id) & historical.drug_id.isin(ext.drug_id)]
            else:
                ext = target.loc[pd.MultiIndex.from_frame(target[['cell_id', 'drug_id']]).isin(source_pairs)]
                internal = historical.loc[source_pairs.isin(pd.MultiIndex.from_frame(ext[['cell_id', 'drug_id']]))]
            frames[(study, domain, 'external')] = ext.reset_index(drop=True)
            frames[(study, domain, 'internal')] = internal.reset_index(drop=True)
            for regime, frame in [('external', ext), ('internal', internal)]:
                stages.append(dict(stage=f'{regime}_{domain}', study=study, **population(frame)))
    return historical, expanded, mats, ids, frames, pd.DataFrame(stages), feature_checks


def prepare(data):
    source, expanded, mats, ids, frames, stages, checks = data
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in ('populations', 'checkpoints'):
        (OUT / directory).mkdir(exist_ok=True)
    atomic_csv(source, OUT / 'populations/historical_edges.csv')
    atomic_csv(expanded, OUT / 'populations/current_expanded_edges.csv')
    for (study, domain, regime), frame in frames.items():
        atomic_csv(frame, OUT / f'populations/{study}_{regime}_{domain}.csv')
    atomic_csv(stages, OUT / 'stage_populations.csv')
    paths = [ROOT / 'results/model_lock.json', ROOT / 'results/tuning_inductive_mf/selected_config.csv',
             ROOT / 'results/internal_seeds/selection_summary.csv', ROOT / 'confirmatory_protocol_v1.md',
             ROOT / 'data/confirmatory/responses.csv', ROOT / 'data/processed_multistudy/responses.csv',
             ROOT / 'data/holdout_processed/responses.csv', Path(__file__), OUT / 'populations/historical_edges.csv',
             OUT / 'populations/current_expanded_edges.csv']
    for directory in ('processed/features', 'processed_multistudy/features', 'holdout_features'):
        paths.extend(sorted((ROOT / 'data' / directory).glob('*')))
    manifest = dict(hash_interpretation='Current audit hashes only; not retrospective proof of registration time or target-access chronology.',
                    configuration=CONFIG, seeds=list(SEEDS), threads=2, memory_budget_gb=8,
                    feature_policy='Historical source rows retained; source-only fit; target-only rows transformed using identical ordered columns.',
                    feature_checks=checks, files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=digest(p), size_bytes=p.stat().st_size) for p in paths if p.is_file()])
    previous = OUT / 'manifest_hashes.json'
    if previous.exists():
        assert json.loads(previous.read_text()) == manifest, 'Resume manifest differs'
    else:
        write_json(manifest, previous)
    print(stages.to_string(index=False), flush=True)


def model_fit(frame, reduced, ids, seed):
    return InductiveMatrixFactorization(**CONFIG, seed=seed).fit(
        frame, reduced['cell_expression'], ids['cell_expression'], reduced['drug_descriptors'], ids['drug_descriptors'])


def scored(frame, prediction, seed, study, domain, regime):
    assert len(frame) == len(prediction) and np.isfinite(prediction).all()
    result = drug_scores(frame, prediction)
    for key, value in dict(seed=seed, test_study=study, domain=domain, regime=regime).items():
        result[key] = value
    return result


def independent_bca(values):
    """Explicit percentile bias correction and jackknife acceleration for a mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(20261004)
    n = len(values)
    distribution = np.concatenate([values[rng.randint(0, n, (min(256, 9999-start), n))].mean(axis=1)
                                   for start in range(0, 9999, 256)])
    theta = values.mean()
    z0 = norm.ppf(((distribution < theta).sum() + (distribution <= theta).sum()) / (2 * len(distribution)))
    jackknife = (values.sum() - values) / (n - 1)
    influence = jackknife.mean() - jackknife
    acceleration = np.sum(influence**3) / (6 * np.sum(influence**2)**1.5)
    z = norm.ppf([0.025, 0.975])
    q = norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
    return np.quantile(distribution, q)


def aggregate():
    scores = pd.concat([pd.read_csv(OUT / f'checkpoints/per_drug_seed_{seed}.csv', dtype={'drug_id': str}) for seed in SEEDS], ignore_index=True)
    atomic_csv(scores, OUT / 'per_drug.csv')
    means = scores.groupby(['test_study', 'domain', 'regime', 'drug_id'], as_index=False).agg(
        spearman=('spearman', 'mean'), n_seeds=('seed', 'nunique'), n_cells=('n_cells', 'first'))
    atomic_csv(means, OUT / 'per_drug_seed_mean.csv')
    metrics = scores.groupby(['seed', 'test_study', 'domain', 'regime'], as_index=False).agg(
        macro_spearman=('spearman', 'mean'), n_drugs=('spearman', 'count'))
    atomic_csv(metrics, OUT / 'metrics.csv')
    variability = metrics.groupby(['test_study', 'domain', 'regime'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), seed_mean=('macro_spearman', 'mean'), seed_sd=('macro_spearman', 'std'),
        seed_min=('macro_spearman', 'min'), seed_max=('macro_spearman', 'max'))
    atomic_csv(variability, OUT / 'seed_variability.csv')
    gaps, paired_tables, seed_gaps = [], [], []
    for (study, domain), frame in means.groupby(['test_study', 'domain']):
        paired = frame.pivot(index='drug_id', columns='regime', values='spearman').dropna()
        difference = (paired.internal - paired.external).to_numpy()
        low, high = independent_bca(difference)
        reference = bootstrap((difference,), np.mean, method='BCa', n_resamples=9999,
            confidence_level=0.95, random_state=20261004, batch=256).confidence_interval
        assert np.allclose([low, high], [reference.low, reference.high], atol=1e-12, rtol=0)
        gaps.append(dict(test_study=study, domain=domain, n_drugs=len(paired),
            mean_internal_spearman=float(paired.internal.mean()), mean_external_spearman=float(paired.external.mean()),
            gap_internal_minus_external=float(difference.mean()), ci_low=float(low), ci_high=float(high),
            supports_C1=bool(low > 0.10), material_threshold=0.10, resamples=9999, bootstrap_seed=20261004,
            bootstrap_method='independent explicit BCa mean; verified against scipy', scipy_ci_low=float(reference.low), scipy_ci_high=float(reference.high)))
        paired = paired.reset_index().assign(test_study=study, domain=domain)
        paired['gap'] = paired.internal - paired.external
        paired_tables.append(paired)
        for seed in SEEDS:
            selected = scores.loc[(scores.seed == seed) & (scores.test_study == study) & (scores.domain == domain)]
            wide = selected.pivot(index='drug_id', columns='regime', values='spearman').loc[paired.drug_id].dropna()
            seed_gaps.append(dict(seed=seed, test_study=study, domain=domain, n_drugs=len(wide), gap=float((wide.internal-wide.external).mean())))
    gap_frame = pd.DataFrame(gaps)
    atomic_csv(gap_frame, OUT / 'gaps.csv')
    atomic_csv(pd.concat(paired_tables, ignore_index=True), OUT / 'paired_per_drug_gaps.csv')
    seed_gap_frame = pd.DataFrame(seed_gaps)
    atomic_csv(seed_gap_frame, OUT / 'seed_gaps.csv')
    atomic_csv(seed_gap_frame.groupby(['test_study', 'domain'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), seed_mean=('gap', 'mean'), seed_sd=('gap', 'std'), seed_min=('gap', 'min'), seed_max=('gap', 'max')),
        OUT / 'gap_seed_variability.csv')
    write_json(dict(status='complete', source_population=dict(n_pairs=293194, n_cells=819, n_drugs=451),
        configuration=CONFIG, seeds=list(SEEDS), gaps=gaps,
        seed_aggregation='Mean per-seed per-drug Spearman, then paired drug macro mean; seeds descriptive, drugs inferential.',
        hash_interpretation='Current audit hashes; no retrospective registration-time proof.',
        scope='Historical-population sensitivity, shared entities and exact pairs only; target labels never used for selection.'), OUT / 'summary.json')
    print(gap_frame.to_string(index=False), flush=True)


def verify(data):
    source, expanded, mats, ids, frames, stages, checks = data
    saved = responses(OUT / 'populations/historical_edges.csv')
    pd.testing.assert_frame_equal(source, saved, check_exact=True)
    pd.testing.assert_frame_equal(expanded, responses(OUT / 'populations/current_expanded_edges.csv'), check_exact=True)
    manifest = json.loads((OUT / 'manifest_hashes.json').read_text())
    for entry in manifest['files']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    scores = pd.read_csv(OUT / 'per_drug.csv', dtype={'drug_id': str})
    assert set(scores.seed) == set(SEEDS) and not scores.duplicated(KEYS).any()
    assert scores.spearman.notna().all()
    expected_counts = []
    for (study, domain, regime), frame in frames.items():
        pd.testing.assert_frame_equal(frame, responses(OUT / f'populations/{study}_{regime}_{domain}.csv'), check_exact=True)
        expected_drugs = set(frame.groupby('drug_id').filter(lambda g: g.cell_id.nunique() >= 5 and g.sensitivity.nunique() >= 2).drug_id)
        for seed in SEEDS:
            table = scores.loc[(scores.seed == seed) & (scores.test_study == study) & (scores.domain == domain) & (scores.regime == regime)]
            assert set(table.drug_id) == expected_drugs
        expected_counts.append(dict(test_study=study, domain=domain, regime=regime, expected_n_drugs=len(expected_drugs)))
    metrics = pd.read_csv(OUT / 'metrics.csv')
    assert len(metrics) == 80 and not metrics.duplicated(['seed', 'test_study', 'domain', 'regime']).any()
    for seed in SEEDS:
        oof = pd.read_csv(OUT / f'checkpoints/oof_seed_{seed}.csv.gz', dtype={'cell_id': str, 'drug_id': str})
        assert len(oof) == len(source) and not oof.duplicated(['cell_id', 'drug_id']).any()
        pd.testing.assert_frame_equal(oof[['cell_id', 'drug_id']], source[['cell_id', 'drug_id']], check_exact=True)
        assert np.array_equal(oof.y_true, source.sensitivity)
        assert np.isfinite(oof.y_pred).all() and set(oof.fold) == set(range(5))
        for split in make_folds(source, 'random_pair', folds=5, seed=seed):
            assert (oof.iloc[split.test_index].fold == split.fold).all()
            assert len(np.intersect1d(split.train_index, split.test_index)) == 0
        for (study, domain, regime), frame in frames.items():
            if regime == 'internal':
                selected = frame.merge(oof, on=['cell_id', 'drug_id'], validate='one_to_one')
                actual = drug_scores(selected.rename(columns={'sensitivity': 'unused'}).assign(sensitivity=selected.y_true), selected.y_pred.to_numpy()).sort_values('drug_id')
                saved_scores = scores.loc[(scores.seed == seed) & (scores.test_study == study) & (scores.domain == domain) & (scores.regime == regime)].sort_values('drug_id')
                assert np.allclose(actual.spearman, saved_scores.spearman, atol=1e-14)
    gaps = pd.read_csv(OUT / 'gaps.csv')
    pairs = pd.read_csv(OUT / 'paired_per_drug_gaps.csv', dtype={'drug_id': str})
    assert len(gaps) == 4
    for row in gaps.itertuples():
        group = pairs.loc[(pairs.test_study == row.test_study) & (pairs.domain == row.domain)]
        assert len(group) == row.n_drugs
        assert np.isclose(group.gap.mean(), row.gap_internal_minus_external, atol=1e-14)
        assert np.allclose(independent_bca(group.gap), [row.ci_low, row.ci_high], atol=1e-12)
    write_json(dict(status='passed', full_seed_set=list(SEEDS), unique_per_drug_keys=KEYS,
        input_population_equality=True, oof_partition_and_score_recomputation=True,
        current_input_hashes_match=True, independent_bca_matches=True, expected_drug_counts=expected_counts), OUT / 'verification.json')
    print('PASS: population equality, hashes, full seeds, unique keys, expected drug counts, OOF partitions/scores, BCa', flush=True)


def run(data):
    source, expanded, mats, ids, frames, stages, checks = data
    reduced, audits = {}, []
    for name, column in [('cell_expression', 'cell_id'), ('drug_descriptors', 'drug_id')]:
        reduced[name], _, audit = fit_reducer(mats[name], ids[name], set(source[column]), 64)
        audits.append(dict(feature=name, **audit))
    atomic_csv(pd.DataFrame(audits), OUT / 'reducer_audit.csv')
    joblib.dump(dict(reduced=reduced, ids=ids, source_only_audit=audits), OUT / 'checkpoints/reduced_features.joblib')
    for seed in SEEDS:
        final_path = OUT / f'checkpoints/per_drug_seed_{seed}.csv'
        if final_path.exists():
            print(f'SKIP complete seed {seed}', flush=True)
            continue
        print(f'START seed {seed}', flush=True)
        model_path = OUT / f'checkpoints/external_model_seed_{seed}.joblib'
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = model_fit(source, reduced, ids, seed)
            joblib.dump(model, model_path)
        score_tables, external_rows = [], []
        for (study, domain, regime), frame in frames.items():
            if regime != 'external':
                continue
            prediction = model.predict(frame[['cell_id', 'drug_id']])
            score_tables.append(scored(frame, prediction, seed, study, domain, regime))
            external_rows.append(frame[['cell_id', 'drug_id']].assign(y_true=frame.sensitivity, y_pred=prediction,
                seed=seed, test_study=study, domain=domain))
        atomic_csv(pd.concat(external_rows, ignore_index=True), OUT / f'checkpoints/external_rows_seed_{seed}.csv.gz')
        all_predictions = np.full(len(source), np.nan)
        fold_labels = np.full(len(source), -1)
        for split in make_folds(source, 'random_pair', folds=5, seed=seed):
            train = source.iloc[split.train_index].reset_index(drop=True)
            # Label-free reducer reuse is exactly valid only for identical fit entity sets.
            assert set(train.cell_id) == set(source.cell_id) and set(train.drug_id) == set(source.drug_id)
            path = OUT / f'checkpoints/oof_seed_{seed}_fold_{split.fold}.npz'
            if path.exists():
                checkpoint = np.load(path)
                assert np.array_equal(checkpoint['index'], split.test_index)
                prediction = checkpoint['prediction']
            else:
                fold_model = model_fit(train, reduced, ids, seed)
                prediction = fold_model.predict(source.iloc[split.test_index][['cell_id', 'drug_id']])
                np.savez_compressed(path, index=split.test_index, prediction=prediction)
            assert np.isfinite(prediction).all()
            all_predictions[split.test_index] = prediction
            fold_labels[split.test_index] = split.fold
            print(f'END seed {seed} fold {split.fold}', flush=True)
        oof = source[['cell_id', 'drug_id']].assign(y_true=source.sensitivity, y_pred=all_predictions, fold=fold_labels)
        atomic_csv(oof, OUT / f'checkpoints/oof_seed_{seed}.csv.gz')
        for (study, domain, regime), frame in frames.items():
            if regime == 'internal':
                selected = frame.merge(oof[['cell_id', 'drug_id', 'y_pred']], on=['cell_id', 'drug_id'], validate='one_to_one')
                score_tables.append(scored(selected, selected.y_pred.to_numpy(), seed, study, domain, regime))
        atomic_csv(pd.concat(score_tables, ignore_index=True), final_path)
        print(f'END seed {seed}', flush=True)
    aggregate()
    verify(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        data = inputs()
        if args.verify:
            verify(data)
        else:
            prepare(data)
            if not args.prepare:
                run(data)


if __name__ == '__main__':
    main()
