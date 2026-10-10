# Reproducibility audit of the motivating model repositories

Audit date: 2026-10-04

## Scope

This audit determines which parts of DRExplainer and ExplainMIX can be used as
comparators in the cross-database benchmark. It audits the downloaded public code,
not unpublished author code and not the intent of the papers.

Frozen archives:

- DRExplainer `main` archive SHA-256:
  `dc80d2f9cc4f19e39a64424fc0b0ed2a51066c8784b37a6225bd335bf3a53ed0`
- ExplainMIX `master` archive SHA-256:
  `bd0faef4687660086839c699b5b86ae2ae5ecc69a0643f83fb59c1605801b4ab`

## Blocking findings

### 1. DRExplainer's released evaluation adjacency contains positive test edges

`prediction/code/dgcn_eval.py:32-33` loads training and positive test triples.
Line 55 concatenates both into `ADJACENCY_DATA`, line 57 constructs the graph
adjacency from that combined table, and lines 74-81 use that adjacency to predict
the test triples. Therefore the queried positive edges are present in the graph
used for message passing during evaluation.

The training script itself uses only training triples for adjacency
(`prediction/code/DGCN.py:311`), so this is specifically an evaluation-path issue.
Published repository metrics from `dgcn_eval.py` cannot be treated as leakage-free
held-out prediction evidence. Our comparator must construct every evaluation
adjacency from training edges only.

### 2. DRExplainer explanation targets are internally generated

`explanation/ground_truth.py:80-90` samples 0.8% of the response graph and loads
cell/drug similarity edges. Lines 24-76 define explanation targets by searching the
same response and similarity graphs for matching relations. These targets measure
agreement with a graph rule derived from model inputs, not independent biological
validity. They can be retained as a synthetic fidelity check but not as evidence
that an explanation recovers causal biology.

Our primary explanation evaluation therefore uses edge/feature deletion fidelity,
seed and database stability, and external target/dependency annotations excluded
from graph construction.

## Important observations

### 3. Only one fold and one split seed are exercised in the DRExplainer scripts

Although the utility code implements five splits, the training loop is
`for fold in range(0,1)` (`prediction/code/DGCN.py:281`) and the evaluation list is
`folds = [0]` (`prediction/code/dgcn_eval.py:21`). Modes for held-out cells and drugs
are implemented in `prediction/code/utils.py:89-137`, but the released training
loop runs `for mode in range(0,1)` (`DGCN.py:276`). The repository therefore does
not supply repeated cold-start estimates.

### 4. ExplainMIX uses a fixed random pair split

`code/CCLE_input_data.py:45` makes one random pair split with `random_state=42` and
a fixed test size. Cell-cell and drug-drug similarity edges are then added to the
training graph (`CCLE_input_data.py:64-90`). This supports a within-network test but
does not test unseen cells, unseen drugs, or another screening platform.

The main explanation/evaluation path builds adjacency from training triples only
(`code/ExplainMIX_main.py:105-125`), so the direct DRExplainer leakage described
above is not present in this path.

### 5. ExplainMIX predicts thresholded response relations

`code/CCLE_input_data.py:23-45` loads precomputed sensitive and non-sensitive pair
lists and encodes them as relation types 1 and 0. The code repository does not show
how those two source lists were derived from continuous response values. Its task
and metrics are therefore not directly comparable with continuous cross-database
`fit_auc` prediction without rederiving and documenting a threshold.

## Decision

1. Keep directed heterogeneous graph message passing and relation-specific
   explanations as candidate methods.
2. Rebuild splits from the common benchmark contract; never use test response
   edges in adjacency construction.
3. Predict continuous sensitivity as the primary task. A binary sensitive/resistant
   task is secondary and its threshold must be fitted within the training study.
4. Report five folds and ten seeds for final internal results, with cold-cell,
   cold-drug, cold-both, and cross-study arms.
5. Treat graph-derived explanation targets as synthetic fidelity checks only.
