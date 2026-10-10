import numpy as np
import pandas as pd

from crossdb_drug_response.itens import ITENSRegressor


def test_itens_ignores_target_labels_and_contributions_reconstruct():
    cells = np.asarray(["C0", "C1", "C2", "C3"])
    drugs = np.asarray(["D0", "D1", "D2"])
    cell_x = np.asarray([[1, 0], [0.8, 0.2], [0.1, 0.9], [0, 1]], dtype=float)
    drug_x = np.asarray([[1, 0], [0.5, 0.5], [0, 1]], dtype=float)
    train = pd.DataFrame(
        [
            ("C0", "D0", 0.1), ("C0", "D1", 0.2),
            ("C1", "D0", 0.3), ("C1", "D1", 0.4),
            ("C2", "D0", 0.7), ("C2", "D1", 0.8),
        ],
        columns=["cell_id", "drug_id", "sensitivity"],
    )
    model = ITENSRegressor(k_cell=3, k_drug=2, ess_shrinkage=1).fit(
        train, cell_x, cells, drug_x, drugs
    )
    query = pd.DataFrame({"cell_id": ["C3"], "drug_id": ["D2"], "sensitivity": [999.0]})
    first = model.predict(query)
    query["sensitivity"] = -999.0
    second = model.predict(query)
    assert np.array_equal(first, second)
    explained, contributions = model.explain_one("C3", "D2")
    assert abs(explained - first[0]) < 1e-10
    assert set(zip(contributions.neighbor_cell_id, contributions.neighbor_drug_id)).issubset(
        model.training_edges_
    )
    top_edge = {
        (contributions.iloc[0].neighbor_cell_id, contributions.iloc[0].neighbor_drug_id)
    }
    deleted, deleted_contributions = model.explain_one("C3", "D2", excluded_edges=top_edge)
    assert top_edge.isdisjoint(
        set(zip(deleted_contributions.neighbor_cell_id, deleted_contributions.neighbor_drug_id))
    )
    assert deleted != explained
