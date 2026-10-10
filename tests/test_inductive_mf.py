import numpy as np
import pandas as pd

from crossdb_drug_response.inductive_mf import InductiveMatrixFactorization


def test_inductive_mf_predicts_seen_and_unseen_entities_without_target_labels():
    cells = np.asarray(["C0", "C1", "C2"])
    drugs = np.asarray(["D0", "D1", "D2"])
    cell_features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    drug_features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3]])
    train = pd.DataFrame(
        [("C0", "D0", 0.1), ("C0", "D1", 0.3), ("C1", "D0", 0.7), ("C1", "D1", 0.9)],
        columns=["cell_id", "drug_id", "sensitivity"],
    )
    model = InductiveMatrixFactorization(rank=2, iterations=5).fit(
        train, cell_features, cells, drug_features, drugs
    )
    query = pd.DataFrame({"cell_id": ["C0", "C2"], "drug_id": ["D0", "D2"], "sensitivity": [9.0, 9.0]})
    first = model.predict(query)
    query["sensitivity"] = -9.0
    second = model.predict(query)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
