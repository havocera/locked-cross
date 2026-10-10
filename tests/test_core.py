import numpy as np
import pandas as pd

from crossdb_drug_response.baseline import AdditiveBiasRegressor
from crossdb_drug_response.metrics import evaluate_predictions
from crossdb_drug_response.schema import validate_responses
from crossdb_drug_response.splits import make_folds


def response_table(study="A", cells=12, drugs=10):
    rows = []
    for cell in range(cells):
        for drug in range(drugs):
            rows.append(
                {
                    "study": study,
                    "cell_id": f"C{cell:02d}",
                    "drug_id": f"D{drug:02d}",
                    "sensitivity": cell * 0.1 - drug * 0.2 + ((cell * drug) % 7) * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_schema_rejects_duplicate_keys():
    frame = response_table()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    try:
        validate_responses(duplicated)
    except ValueError as error:
        assert "not unique" in str(error)
    else:
        raise AssertionError("Duplicate response key was accepted")


def test_entity_splits_are_disjoint():
    frame = response_table()
    for split in make_folds(frame, "cold_cell", folds=5, seed=11):
        train = frame.iloc[split.train_index]
        test = frame.iloc[split.test_index]
        assert set(train.cell_id).isdisjoint(test.cell_id)
    for split in make_folds(frame, "cold_drug", folds=5, seed=11):
        train = frame.iloc[split.train_index]
        test = frame.iloc[split.test_index]
        assert set(train.drug_id).isdisjoint(test.drug_id)
    for split in make_folds(frame, "cold_both", folds=5, seed=11):
        train = frame.iloc[split.train_index]
        test = frame.iloc[split.test_index]
        assert set(train.cell_id).isdisjoint(test.cell_id)
        assert set(train.drug_id).isdisjoint(test.drug_id)


def test_baseline_handles_unseen_entities():
    frame = response_table()
    model = AdditiveBiasRegressor().fit(frame)
    unseen = pd.DataFrame({"cell_id": ["NEW"], "drug_id": ["NEW"]})
    prediction = model.predict(unseen)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert prediction[0] == model.global_mean_


def test_metrics_equal_weight_per_drug():
    rows = []
    for drug in ["D1", "D2"]:
        for index in range(6):
            rows.append(
                {
                    "cell_id": f"C{index}",
                    "drug_id": drug,
                    "y_true": float(index),
                    "y_pred": float(index if drug == "D1" else -index),
                }
            )
    metrics = evaluate_predictions(pd.DataFrame(rows))
    macro = metrics.loc[metrics.metric == "spearman_drug_macro_mean", "value"].iloc[0]
    assert abs(macro) < 1e-12
