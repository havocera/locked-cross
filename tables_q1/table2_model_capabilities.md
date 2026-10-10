| Model | Information used | Cold cell | Cold drug | Cold both | Exact decomposition | Primary limitation |
| --- | --- | --- | --- | --- | --- | --- |
| Additive bias | Entity IDs | No | Yes | No | No | Global fallback for unseen drugs; no drug-specific effect |
| Ridge | Expression + descriptors + bilinear terms | Yes | Yes | Yes | No | Linear extrapolation |
| HistGradientBoosting | Same pair features | Yes | Yes | Yes | No | Tree ensemble |
| LightGBM | Same pair features | Yes | Yes | Yes | No | Boosted trees |
| Cell kNN | Expression neighbors + response edges | Yes | No | No | Exact weights | Undefined for unseen drugs |
| Drug kNN | Descriptor neighbors + response edges | No | Yes | No | Exact weights | Undefined for unseen cells |
| Product kNN | Cell x drug neighborhoods | Yes | Yes | Yes | Exact weights | Sparse support |
| Inductive MF | Response matrix + feature projections | Yes | Yes | Yes | No | ALS + ridge projection |
| ITENS | Three training-edge neighborhood experts | Yes | Yes | Yes | Exact signed edges | ESS shrinkage |