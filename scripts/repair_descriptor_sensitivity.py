"""B2 post-hoc descriptor audit and source-only min/max sensitivity.

Run from project root. Owns only results/repair_descriptor_sensitivity.
No descriptor regeneration is claimed without the original Mordred environment.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_feature_baseline import load_matrix, fit_reducer
from analyze_ood_safety import SEEDS
from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization
from crossdb_drug_response.metrics import evaluate_predictions

OUT = ROOT / 'results/repair_descriptor_sensitivity'
FEATURES = ROOT / 'data/holdout_features'
DEVELOPMENT = ROOT / 'data/processed_multistudy/responses.csv'
TARGET = ROOT / 'data/holdout_processed/responses.csv'
CONFIG = ROOT / 'results/tuning_inductive_mf/selected_config.csv'
NAMES = ('VR1_A', 'VR2_A', 'WPath')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def csv(frame, name):
    frame.to_csv(OUT / name, index=False)

def inputs():
    paths = [DEVELOPMENT, TARGET, CONFIG, ROOT/'results/model_lock.json',
             ROOT/'results/ood_safety/ensemble_predictions_NCI60.csv.gz']
    paths += list(FEATURES.glob('*'))
    paths += list((ROOT/'data/raw').glob('**/*drug*.tsv.gz'))
    return sorted(set(p for p in paths if p.is_file()))

def guarded_reducer(matrix, ids, source_ids, names):
    mask = np.isin(ids, sorted(source_ids))
    source = matrix[mask].astype(float)
    source[~np.isfinite(source)] = np.nan
    valid = np.isfinite(source).any(axis=0)
    low = np.full(matrix.shape[1], np.nan)
    high = low.copy(); med = low.copy()
    low[valid] = np.nanmin(source[:, valid], axis=0)
    high[valid] = np.nanmax(source[:, valid], axis=0)
    med[valid] = np.nanmedian(source[:, valid], axis=0)
    all_values = matrix.astype(float)
    all_values[~np.isfinite(all_values)] = np.nan
    clipped = np.clip(all_values[:, valid], low[valid], high[valid])
    clipped = np.where(np.isnan(clipped), med[valid], clipped)
    variable = np.std(clipped[mask], axis=0) > 0
    used = np.flatnonzero(valid)[variable]
    scaler = StandardScaler().fit(clipped[mask][:, variable])
    pca = PCA(n_components=min(64, mask.sum()-1, variable.sum()),
              svd_solver='randomized', random_state=104729)
    pca.fit(scaler.transform(clipped[mask][:, variable]))
    embedding = pca.transform(scaler.transform(clipped[:, variable])).astype(np.float32)
    changed = np.isfinite(all_values) & ((all_values < low) | (all_values > high))
    bounds = pd.DataFrame({'feature_id': names, 'source_min': low, 'source_max': high,
                          'source_median': med, 'source_finite_count': np.isfinite(source).sum(axis=0),
                          'retained': np.isin(np.arange(len(names)), used),
                          'source_entity_count': int(mask.sum())})
    return embedding, bounds, changed, {'scaler': scaler, 'pca': pca, 'used_columns': used,
                                       'source_min': low, 'source_max': high, 'source_median': med}

def audit_raw(matrix, ids, names):
    lookup = {x:i for i,x in enumerate(ids)}
    fl = {x:i for i,x in enumerate(names)}
    extreme_ids = {'SMI_48082'}
    for name in NAMES:
        if name in fl:
            vals = matrix[:, fl[name]]
            finite = np.flatnonzero(np.isfinite(vals))
            extreme_ids.update(ids[finite[np.argsort(np.abs(vals[finite]))[-20:]]])
            if name == 'WPath': extreme_ids.update(ids[np.isfinite(vals) & (vals >= 1e10)])
    rows = []
    for path in sorted((ROOT/'data/raw').glob('**/*drug_descriptors.tsv.gz')):
        print('raw trace', path.name, flush=True)
        for c in pd.read_csv(path, sep='\t', dtype=str, chunksize=250000,
                             usecols=['improve_drug_id','structural_descriptor','descriptor_value']):
            c = c[c.improve_drug_id.isin(extreme_ids) & c.structural_descriptor.isin(NAMES)]
            for r in c.itertuples(index=False):
                try: value = float(r.descriptor_value)
                except ValueError: value = np.nan
                retained = float(matrix[lookup[r.improve_drug_id], fl[r.structural_descriptor]]) if r.improve_drug_id in lookup and r.structural_descriptor in fl else np.nan
                rows.append({'raw_file': str(path.relative_to(ROOT)), 'drug_id': r.improve_drug_id,
                             'descriptor': r.structural_descriptor, 'raw_value_text': r.descriptor_value,
                             'raw_value_float64': value, 'retained_float32': retained,
                             'matches_float32_cast': bool(np.float32(value) == np.float32(retained)) if np.isfinite(value) else None})
    csv(pd.DataFrame(rows), 'source_trace.csv')
    structure_rows = []
    import rdkit
    from rdkit import Chem
    for path in sorted((ROOT/'data/raw').glob('**/*_drugs.tsv.gz')):
        t = pd.read_csv(path, sep='\t', dtype=str)
        t = t[t.improve_drug_id.isin(extreme_ids)].drop_duplicates(['improve_drug_id','canSMILES'])
        for r in t.itertuples(index=False):
            s = r.canSMILES
            mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            row = {'raw_file':str(path.relative_to(ROOT)), 'drug_id':r.improve_drug_id,
                   'raw_canSMILES':s, 'raw_InChIKey':getattr(r,'InChIKey',None),
                   'rdkit_version':rdkit.__version__, 'valid_sanitized':mol is not None}
            if mol is not None:
                fragments = Chem.GetMolFrags(mol)
                distances = Chem.GetDistanceMatrix(mol, useBO=False, useAtomWts=False)
                row.update(canonical_isomeric_smiles=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                           n_atoms=mol.GetNumAtoms(), n_fragments=len(fragments),
                           fragment_sizes=';'.join(map(str,map(len,fragments))),
                           rdkit_unweighted_distance_sum_half=float(distances.sum()/2),
                           distance_max=float(distances.max()),
                           interfragment_atom_pairs=sum(len(a)*len(b) for i,a in enumerate(fragments) for b in fragments[i+1:]))
                w = matrix[lookup[r.improve_drug_id], fl['WPath']] if r.improve_drug_id in lookup else np.nan
                row['retained_WPath'] = float(w)
                row['WPath_matches_distance_sum_float32'] = bool(np.float32(w) == np.float32(distances.sum()/2))
            structure_rows.append(row)
    csv(pd.DataFrame(structure_rows), 'structures_audit.csv')
    return {'rdkit_version':rdkit.__version__, 'mordred_installed':bool(importlib.util.find_spec('mordred')),
            'audited_drug_count':len(extreme_ids)}

def contributions(source_ids):
    original = pd.read_csv(ROOT/'results/ood_safety/ensemble_predictions_NCI60.csv.gz')
    original['squared_error'] = (original.y_pred-original.sensitivity)**2
    t = original.groupby('drug_id').agg(n_pairs=('squared_error','size'), n_cells=('cell_id','nunique'),
        squared_error=('squared_error','sum'), prediction_min=('y_pred','min'),prediction_max=('y_pred','max')).reset_index()
    t = t.sort_values(['squared_error','drug_id'],ascending=[False,True]).reset_index(drop=True)
    t['error_share'] = t.squared_error/t.squared_error.sum()
    t['cumulative_error_share'] = t.error_share.cumsum()
    t['seen_in_source'] = t.drug_id.isin(source_ids)
    csv(t,'original_NCI60_error_contributions.csv')
    return {'n_pairs':len(original), 'n_drugs':len(t), 'rmse':float(np.sqrt(original.squared_error.mean())),
            'total_squared_error':float(original.squared_error.sum()),
            'top1_drug':t.iloc[0].drug_id, 'top1_share':float(t.error_share.iloc[0]),
            'top1_pairs':int(t.n_pairs.iloc[0]), 'top10_share':float(t.error_share.iloc[:10].sum()),
            'top10_pairs':int(t.n_pairs.iloc[:10].sum())}

def metrics(predictions, study, variant, seed, source_ids):
    tables = []
    for population, mask in [('all_target',np.ones(len(predictions),dtype=bool)),
                             ('seen_drugs',predictions.drug_id.isin(source_ids)),
                             ('target_only_drugs',~predictions.drug_id.isin(source_ids))]:
        p = predictions.loc[mask]
        if len(p):
            m = evaluate_predictions(p.rename(columns={'sensitivity':'y_true'}))
            m.insert(0,'population',population); m.insert(0,'seed',seed)
            m.insert(0,'variant',variant); m.insert(0,'study',study)
            tables.append(m)
    return pd.concat(tables,ignore_index=True)

def main(n_seeds):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'checkpoints').mkdir(exist_ok=True)
    checklist = {'outputs':['source_trace.csv','structures_audit.csv','original_NCI60_error_contributions.csv',
                  'source_descriptor_bounds.csv','descriptor_guard_support.csv','seed_metrics.csv','ensemble_metrics.csv',
                  'summary.json','summary.md','manifest.json','verification.json'],
                 'method':'source CTRPv2 expanded feature-complete population; locked rank64; source descriptor min/max clipping before source median/scaler/PCA; no output clipping',
                 'metrics':'global and eligible equal-drug Spearman, RMSE/MAE; support pairs/cells/drugs/eligible drugs; all, seen, target-only; min 5 cells and nonconstant labels/predictions',
                 'conditions':'all feature-complete GDSCv1 and NCI60 target pairs, unchanged between arms; sensitivity post-hoc, not full confirmation',
                 'seeds':list(SEEDS[:n_seeds]), 'units':'native sensitivity=-AUC; descriptors retain original conventions; no rounding of CSV metrics'}
    (OUT/'deliverable_checklist.json').write_text(json.dumps(checklist,indent=2))
    before = {str(p.relative_to(ROOT)):sha(p) for p in inputs()}
    (OUT/'input_hashes_before.json').write_text(json.dumps(before,indent=2))
    cm, ci = load_matrix(FEATURES,'cell_expression'); dm, di = load_matrix(FEATURES,'drug_descriptors')
    names = pd.read_csv(FEATURES/'drug_descriptors_features.csv').feature_id.to_numpy()
    dev = pd.read_csv(DEVELOPMENT, dtype={'cell_id':str,'drug_id':str})
    target_input = pd.read_csv(TARGET,dtype={'cell_id':str,'drug_id':str})
    source = dev[(dev.study=='CTRPv2') & dev.cell_id.isin(ci) & dev.drug_id.isin(di)].reset_index(drop=True)
    target = target_input[target_input.cell_id.isin(ci) & target_input.drug_id.isin(di)].reset_index(drop=True)
    source_ids = set(source.drug_id)
    csv(source[['study','cell_id','drug_id','sensitivity']], 'expanded_source_responses.csv.gz')
    chem = audit_raw(dm,di,names)
    error = contributions(source_ids)
    cell = fit_reducer(cm,ci,set(source.cell_id),64)[0]
    raw = fit_reducer(dm,di,source_ids,64)[0]
    guarded,bounds,changed,state = guarded_reducer(dm,di,source_ids,names)
    csv(bounds,'source_descriptor_bounds.csv')
    csv(pd.DataFrame({'drug_id':di,'seen_in_source':np.isin(di,sorted(source_ids)),
                      'clipped_descriptor_count':changed.sum(axis=1)}),'descriptor_guard_support.csv')
    with open(OUT/'checkpoints/descriptor_transform.pkl','wb') as f: pickle.dump(state,f)
    assert not changed[np.isin(di,sorted(source_ids))].any()
    assert np.array_equal(raw[np.isin(di,sorted(source_ids))],guarded[np.isin(di,sorted(source_ids))])
    selected = pd.read_csv(CONFIG).iloc[0]
    assert int(selected['rank']) == 64
    seed_tables=[]; ensemble_tables=[]
    for variant, embedding in [('raw',raw),('guarded',guarded)]:
        sums = {s:np.zeros(len(t)) for s,t in target.groupby('study',sort=True)}
        for seed in SEEDS[:n_seeds]:
            print('training',variant,seed,len(source),flush=True)
            checkpoint=OUT/f'checkpoints/{variant}_{seed}.pkl'
            if checkpoint.exists():
                with open(checkpoint,'rb') as f: model=pickle.load(f)
            else:
                model=InductiveMatrixFactorization(rank=64,regularization=float(selected.regularization),
                    iterations=int(selected.iterations),projection_alpha=float(selected.projection_alpha),seed=seed)
                model.fit(source,cell,ci,embedding,di)
                with open(checkpoint,'wb') as f: pickle.dump(model,f)
            for study,t in target.groupby('study',sort=True):
                pred = t[['study','cell_id','drug_id','sensitivity']].copy()
                pred['y_pred'] = np.concatenate([model.predict(t.iloc[i:i+50000]) for i in range(0,len(t),50000)])
                csv(pred,f'{variant}_{seed}_{study}_predictions.csv.gz')
                seed_tables.append(metrics(pred,study,variant,seed,source_ids))
                sums[study]+=pred.y_pred.to_numpy()
            csv(pd.concat(seed_tables,ignore_index=True),'seed_metrics.csv')
        for study,t in target.groupby('study',sort=True):
            pred=t[['study','cell_id','drug_id','sensitivity']].copy()
            pred['y_pred']=sums[study]/n_seeds
            csv(pred,f'ensemble_{variant}_{study}.csv.gz')
            ensemble_tables.append(metrics(pred,study,variant,'ensemble',source_ids))
    csv(pd.concat(ensemble_tables,ignore_index=True),'ensemble_metrics.csv')
    summary={'source':{'pairs':len(source),'cells':source.cell_id.nunique(),'drugs':source.drug_id.nunique()},
             'seeds':list(SEEDS[:n_seeds]),'original_NCI60':error,'chemistry':chem,
             'descriptor_recomputed_with_original_mordred':False,
             'interpretation':'Numerical failure in the current descriptor/preprocessing representation. Disconnected structures can explain WPath distance-sentinel extremes when matched, but valid SMILES do not establish VR1_A/VR2_A correctness. Original Mordred version/settings and exact recomputation unavailable.',
             'definition_sources':['https://mordred-descriptor.github.io/documentation/master/matrix.html',
                                   'https://mordred-descriptor.github.io/documentation/v0.3.0/_modules/mordred/WienerIndex.html'],
             'source_access':'Search snippets located Mordred definitions; full pages blocked by network non-public-address policy. RDKit distance sums are independent diagnostics, not claimed original Mordred regeneration.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    m=pd.concat(ensemble_tables)
    display=m[m.metric.isin(['rmse','spearman_drug_macro_mean'])]
    text='# B2 descriptor sensitivity (post-hoc)\n\n'+json.dumps(summary,indent=2)+'\n\n## Ensemble metrics\n\n'+display.to_csv(index=False)
    text+='\nTen paired seeds are robustness evidence, not a repeat of the full confirmatory framework. No target labels define bounds, medians, scaling or PCA; no pairs are removed by guarding. Rank correlation eligibility follows the existing >=5-cell, nonconstant rule.\n'
    (OUT/'summary.md').write_text(text,encoding='utf-8')
    after={str(p.relative_to(ROOT)):sha(p) for p in inputs()}
    assert before==after, 'Original input hashes changed'
    (OUT/'manifest.json').write_text(json.dumps({'inputs':before,'script_sha256':sha(Path(__file__)),
        'outputs':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='manifest.json'},
        'checkpoint_files':[str(p.relative_to(OUT)) for p in (OUT/'checkpoints').glob('*')]},indent=2))
    verify()

def verify():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in manifest['inputs'].items())
    assert all(sha(OUT/p)==h for p,h in manifest['outputs'].items())
    dm,di=load_matrix(FEATURES,'drug_descriptors')
    names=pd.read_csv(FEATURES/'drug_descriptors_features.csv').feature_id.to_numpy()
    source=pd.read_csv(OUT/'expanded_source_responses.csv.gz',dtype={'cell_id':str,'drug_id':str})
    sm=np.isin(di,source.drug_id.unique())
    b=pd.read_csv(OUT/'source_descriptor_bounds.csv')
    assert np.array_equal(names,b.feature_id)
    finite=np.isfinite(dm[sm]).any(axis=0)
    assert np.allclose(np.nanmin(dm[sm][:,finite].astype(float),axis=0),b.loc[finite,'source_min'],rtol=1e-12,atol=1e-12)
    assert np.allclose(np.nanmax(dm[sm][:,finite].astype(float),axis=0),b.loc[finite,'source_max'],rtol=1e-12,atol=1e-12)
    assert np.allclose(np.nanmedian(dm[sm][:,finite].astype(float),axis=0),b.loc[finite,'source_median'],rtol=1e-12,atol=1e-12)
    metrics_saved=pd.read_csv(OUT/'ensemble_metrics.csv')
    tgt=pd.read_csv(TARGET,dtype={'cell_id':str,'drug_id':str})
    cm,ci=load_matrix(FEATURES,'cell_expression')
    target=tgt[tgt.cell_id.isin(ci) & tgt.drug_id.isin(di)]
    assert len(target)>0 and set(target.study)=={'GDSCv1','NCI60'}
    dev=pd.read_csv(DEVELOPMENT,dtype={'cell_id':str,'drug_id':str})
    expected_source=dev[(dev.study=='CTRPv2') & dev.cell_id.isin(ci) & dev.drug_id.isin(di)]
    assert source.reset_index(drop=True).equals(expected_source[['study','cell_id','drug_id','sensitivity']].reset_index(drop=True))
    checks=[]
    for study,t in target.groupby('study',sort=True):
        r=pd.read_csv(OUT/f'ensemble_raw_{study}.csv.gz',dtype={'cell_id':str,'drug_id':str}); g=pd.read_csv(OUT/f'ensemble_guarded_{study}.csv.gz',dtype={'cell_id':str,'drug_id':str})
        assert r[['study','cell_id','drug_id','sensitivity']].equals(g[['study','cell_id','drug_id','sensitivity']])
        assert r[['study','cell_id','drug_id','sensitivity']].reset_index(drop=True).equals(t[['study','cell_id','drug_id','sensitivity']].reset_index(drop=True))
        assert np.array_equal(r.loc[r.drug_id.isin(source.drug_id),'y_pred'],g.loc[g.drug_id.isin(source.drug_id),'y_pred'])
        for variant,p in [('raw',r),('guarded',g)]:
            rmse=float(np.sqrt(np.dot(p.y_pred-p.sensitivity,p.y_pred-p.sensitivity)/len(p)))
            saved=metrics_saved[(metrics_saved.study==study)&(metrics_saved.variant==variant)&(metrics_saved.population=='all_target')&(metrics_saved.metric=='rmse')].value.iloc[0]
            assert np.isclose(rmse,saved,rtol=1e-12)
            checks.append({'study':study,'variant':variant,'independent_rmse':rmse,'unchanged_pairs':len(p)})
            from scipy.stats import rankdata
            correlations=[]
            for _,group in p.groupby('drug_id',sort=True):
                if group.cell_id.nunique()>=5 and group.sensitivity.nunique()>1 and group.y_pred.nunique()>1:
                    x=rankdata(group.sensitivity); y=rankdata(group.y_pred)
                    x=x-x.mean(); y=y-y.mean()
                    correlations.append(np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)))
            row=metrics_saved[(metrics_saved.study==study)&(metrics_saved.variant==variant)&(metrics_saved.population=='all_target')&(metrics_saved.metric=='spearman_drug_macro_mean')].iloc[0]
            assert len(correlations)==row.n_eligible_drugs
            assert np.isclose(np.mean(correlations),row.value,rtol=1e-12,atol=1e-14)
            checks[-1]['independent_macro_spearman']=float(np.mean(correlations))
            checks[-1]['eligible_drugs']=len(correlations)
    seeds=pd.read_csv(OUT/'seed_metrics.csv')
    keys=['study','seed','population','metric']
    paired=seeds[seeds.variant=='raw'].merge(seeds[seeds.variant=='guarded'],on=keys,suffixes=('_raw','_guarded'),validate='one_to_one')
    for count in ['n_pairs','n_cells','n_drugs','n_eligible_drugs']:
        assert np.array_equal(paired[count+'_raw'],paired[count+'_guarded'])
    assert len(seeds.seed.unique())==len(summary_seeds := json.loads((OUT/'summary.json').read_text())['seeds'])
    original=pd.read_csv(ROOT/'results/ood_safety/ensemble_predictions_NCI60.csv.gz')
    e=(original.y_pred.to_numpy()-original.sensitivity.to_numpy())**2
    saved=pd.read_csv(OUT/'original_NCI60_error_contributions.csv')
    assert np.isclose(e.sum(),saved.squared_error.sum(),rtol=1e-12)
    summary=json.loads((OUT/'summary.json').read_text())
    assert np.isclose(saved.iloc[0].squared_error/e.sum(),summary['original_NCI60']['top1_share'],rtol=1e-12)
    (OUT/'verification.json').write_text(json.dumps({'passed':True,'checks':checks,
        'original_input_hashes_unchanged':True,'bounds_recomputed_from_source_only':True,
        'seen_drug_predictions_identical':True,'original_error_share_recomputed':True},indent=2))
    print('B2 verification PASS',json.dumps(checks),flush=True)

def finalize():
    """Summarize existing completed runs without retraining or replacing them."""
    summary=json.loads((OUT/'summary.json').read_text())
    audit=pd.read_csv(OUT/'structures_audit.csv')
    trace=pd.read_csv(OUT/'source_trace.csv',dtype={'raw_value_text':str})
    special=audit[audit.drug_id=='SMI_48082'].iloc[0]
    summary['SMI_48082']={'canonical_isomeric_smiles':special.canonical_isomeric_smiles,
        'valid_sanitized':bool(special.valid_sanitized),'n_atoms':int(special.n_atoms),'n_fragments':int(special.n_fragments),
        'raw_values':dict(zip(trace[trace.drug_id=='SMI_48082'].descriptor,trace[trace.drug_id=='SMI_48082'].raw_value_text))}
    unique=audit.drop_duplicates('drug_id')
    w=unique[unique.retained_WPath>=1e10]
    summary['WPath_audit']={'extreme_threshold':1e10,'extreme_drugs':len(w),
        'disconnected':int((w.n_fragments>1).sum()),'valid_sanitized':int(w.valid_sanitized.sum()),
        'matches_RDKit_distance_sum_float32':int(w.WPath_matches_distance_sum_float32.sum()),
        'distance_sentinel':float(w.distance_max.max()),
        'meaning':'Unweighted all-atom-pair shortest-path distance sum (hydrogens implicit). RDKit uses 1e8 for interfragment distances; sentinel pairs explain huge WPath. Not physical connected-parent path lengths.'}
    numeric_trace=trace[np.isfinite(trace.raw_value_float64)]
    assert numeric_trace.matches_float32_cast.all()
    summary['raw_trace_audit']={'rows':len(trace),'finite_numeric_rows':len(numeric_trace),
        'finite_numeric_float32_matches':int(numeric_trace.matches_float32_cast.sum()),
        'nonnumeric_or_missing_rows':int((~np.isfinite(trace.raw_value_float64)).sum()),
        'note':'Text and missing/error outputs retained verbatim; finite raw values compared after original float32 cast.'}
    guard=pd.read_csv(OUT/'descriptor_guard_support.csv')
    summary['guard_support']={}
    for study in ('GDSCv1','NCI60'):
        t=pd.read_csv(OUT/f'ensemble_raw_{study}.csv.gz',dtype={'cell_id':str,'drug_id':str})
        joined=t.merge(guard,on='drug_id',validate='many_to_one')
        summary['guard_support'][study]={'changed_drugs':int(joined.loc[joined.clipped_descriptor_count>0,'drug_id'].nunique()),
            'pairs_with_changed_drug_descriptors':int((joined.clipped_descriptor_count>0).sum())}
    contributions_table=pd.read_csv(OUT/'original_NCI60_error_contributions.csv')
    summary['original_NCI60']['other_drugs_squared_error']=float(contributions_table.squared_error.iloc[1:].sum())
    summary['original_NCI60']['other_drugs_error_share']=float(contributions_table.squared_error.iloc[1:].sum()/contributions_table.squared_error.sum())
    summary['original_NCI60']['top1_share_note']='Float64 rounds top1/top10 share to 1.0; explicit remaining error share is reported separately.'
    seeds=pd.read_csv(OUT/'seed_metrics.csv')
    p=seeds[seeds.variant=='raw'].merge(seeds[seeds.variant=='guarded'],
        on=['study','seed','population','metric'],suffixes=('_raw','_guarded'),validate='one_to_one')
    p['delta_guarded_minus_raw']=p.value_guarded-p.value_raw
    csv(p,'paired_seed_deltas.csv')
    stats=p.groupby(['study','population','metric']).agg(n_seeds=('seed','nunique'),
        raw_mean=('value_raw','mean'),guarded_mean=('value_guarded','mean'),
        raw_min=('value_raw','min'),raw_max=('value_raw','max'),
        guarded_min=('value_guarded','min'),guarded_max=('value_guarded','max'),
        paired_delta_mean=('delta_guarded_minus_raw','mean'),paired_delta_min=('delta_guarded_minus_raw','min'),
        paired_delta_max=('delta_guarded_minus_raw','max')).reset_index()
    csv(stats,'seed_robustness_summary.csv')
    old=pd.read_csv(ROOT/'results/ood_safety/ensemble_predictions_NCI60.csv.gz',dtype={'cell_id':str,'drug_id':str})
    new=pd.read_csv(OUT/'ensemble_raw_NCI60.csv.gz',dtype={'cell_id':str,'drug_id':str})
    assert old[['study','cell_id','drug_id','sensitivity']].equals(new[['study','cell_id','drug_id','sensitivity']])
    summary['raw_rerun_vs_original']={'max_absolute_prediction_difference':float(np.max(np.abs(old.y_pred-new.y_pred))),
        'rmse_relative_difference':float(abs(np.sqrt(np.mean((new.y_pred-new.sensitivity)**2))/summary['original_NCI60']['rmse']-1)),
        'note':'Two-thread rerun versus original numerical execution; values differ at small relative floating-point scale, not byte-identical predictions.'}
    summary['interpretation']='SMI_48082 is a valid connected 102-heavy-atom structure, and its enormous VR1_A/VR2_A are already in the raw descriptor file (not introduced by merge/PCA). VR2_A is numerically consistent with VR1_A/102. Eigenvector-based adjacency indices can be highly sensitive to tiny eigenvector entries; without original Mordred version/settings and recomputation the VR values cannot be classified as calculation artifacts or valid descriptor extremes. WPath extremes match RDKit disconnected-distance sentinel sums, identifying a representation convention rather than biological domain-shift evidence. Guarding removes catastrophic amplitude but target-only NCI60 macro ranking remains near zero.'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    ensemble=pd.read_csv(OUT/'ensemble_metrics.csv')
    text='# B2 descriptor audit and sensitivity\n\nPost-hoc robustness sensitivity; not full confirmatory replication. Expanded original OOD source: 300,507 pairs, 822 cells, 459 drugs. Ten paired seeds; locked rank64, 30 ALS iterations, regularization=1, projection_alpha=10. Source descriptor min/max clipping precedes source median imputation, source scaler and 64-component source PCA. No target labels define preprocessing; no output clipping.\n\n## Chemistry and error attribution\n\n'+json.dumps(summary,indent=2)
    text+='\n\n## Ensemble ranking and error, with support\n\n'+ensemble[ensemble.metric.isin(['rmse','spearman_drug_macro_mean'])].to_csv(index=False)
    text+='\n\n## Seed robustness ranges\n\n'+stats[stats.metric.isin(['rmse','spearman_drug_macro_mean'])].to_csv(index=False)
    text+='\n\n## Limits\n\nMordred not installed; original engine/version/settings unavailable. No original descriptor recomputation claimed. Definitions lookup returned snippets only; full pages blocked by non-public-address network policy. RDKit audit uses the drawn structure, canonical isomeric SMILES with default sanitization, no salt stripping, no charge or tautomer normalization, implicit hydrogens. WPath distance sum is an independent diagnostic. Valid syntax does not validate chemical identity, the measured sample, or every stereochemical detail. Source min/max guarding is a post-hoc source-defined sensitivity, not proof of a generally optimal guard. Seed replicates describe numerical robustness; they are not independent biological replicates and no seed-level significance test is reported. Original prediction RMSE and rerun RMSE have a relative floating-point discrepancy reported above.\n'
    (OUT/'summary.md').write_text(text,encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['script_sha256']=sha(Path(__file__))
    inventory=[str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()]
    inventory+=['scripts/repair_descriptor_sensitivity.py','results/repair_descriptor_sensitivity/output_inventory.txt']
    (OUT/'output_inventory.txt').write_text('\n'.join(sorted(set(inventory)))+'\n')
    manifest['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name not in ('manifest.json','verification.json')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    verify()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--verify',action='store_true'); parser.add_argument('--finalize',action='store_true'); a=parser.parse_args()
    with threadpool_limits(limits=2):
        if a.verify: verify()
        elif a.finalize: finalize()
        else:
            assert 3<=a.seeds<=10
            main(a.seeds)
