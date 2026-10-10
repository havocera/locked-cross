"""Correct saved retrospective Holm values without refitting or altering raw tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_tuned_itens import holm_adjust, atomic_csv

ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / "results/tuned_itens_sensitivity/external_comparison_vs_locked_mf.csv"
    out = ROOT / "results/repair_holm"
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(path)
    original = out / "original_comparison_bonferroni_labelled_holm.csv"
    if not original.exists():
        atomic_csv(frame, original)
    before = pd.read_csv(original)
    frame["p_holm"] = holm_adjust(frame.p_raw.to_numpy())
    # Independent recurrence, without the production function.
    order = np.argsort(frame.p_raw.to_numpy())
    running = 0.0
    expected = np.zeros(len(frame))
    for rank, index in enumerate(order):
        running = max(running, (len(frame)-rank)*frame.p_raw.iloc[index])
        expected[index] = min(1.0, running)
    assert np.allclose(expected,frame.p_holm,rtol=0,atol=1e-15)
    assert frame.drop(columns='p_holm').equals(before.drop(columns='p_holm'))
    atomic_csv(frame,path)
    atomic_csv(frame,out / "corrected_comparison.csv")
    records = [{"study":row.test_study,"p_raw":row.p_raw,"previous":old.p_holm,"corrected":row.p_holm}
               for row,old in zip(frame.itertuples(),before.itertuples())]
    (out / "summary.json").write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(frame.to_string(index=False))
    print('PASS: independent Holm recurrence; all raw tests and effects unchanged')

if __name__=='__main__':
    main()
