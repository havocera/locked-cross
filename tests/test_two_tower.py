import numpy as np
import pandas as pd

from crossdb_drug_response.two_tower import TwoTowerRegressor


def test_two_tower_is_bounded_and_ignores_query_labels():
    cells = np.asarray(["c1", "c2", "c3"])
    drugs = np.asarray(["d1", "d2", "d3"])
    cell_x = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    drug_x = np.asarray([[1, 0], [0, 1], [1, -1]], dtype=np.float32)
    train = pd.DataFrame({
        "cell_id": ["c1", "c1", "c2", "c2"],
        "drug_id": ["d1", "d2", "d1", "d2"],
        "sensitivity": [-0.8, -0.2, -0.6, -0.3],
    })
    model = TwoTowerRegressor(hidden_dim=4, latent_dim=2, epochs=2, batch_size=4, threads=1).fit(
        train, cell_x, cells, drug_x, drugs
    )
    query = pd.DataFrame({"cell_id": ["c3", "c1"], "drug_id": ["d3", "d3"]})
    first = model.predict(query)
    second = model.predict(query.assign(sensitivity=[-0.99, -0.01]))
    assert np.allclose(first, second)
    assert np.all((first >= -1) & (first <= 0))
