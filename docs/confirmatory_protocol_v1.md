# Confirmatory amendment: cross-screen drug-response prediction

**Frozen:** 2026-10-04, after exploratory CTRPv2 -> PRISM/GDSC2/gCSI analyses and
before acquiring or inspecting GDSCv1 and NCI60 response files.

## Status of existing results

All results in `report.md` involving PRISM, GDSC2, or gCSI are exploratory model
development evidence. They may motivate hypotheses and provide variance estimates,
but they are not used as untouched confirmation of ITENS superiority.

GDSCv1 is the primary untouched confirmation screen. NCI60 is the secondary
untouched stress-test screen. Neither response file was present in the frozen input
manifest when this amendment was written. GDSCv1 is related to GDSC2, so confirmation
there tests transfer to a held-out screen release rather than independence from the
GDSC assay family. NCI60 addresses a more distinct but smaller experimental domain.

## Claims and decision rules

### C1: random pair validation overstates external transfer

For a model selected without target labels, its macro-mean per-drug Spearman on the
primary external screen is at least 0.10 below its CTRPv2 random-pair estimate.

Support requires a paired drug-bootstrap 95% confidence interval for the gap whose
lower bound exceeds 0.10. The internal estimate is restricted to the target screen's
eligible drug universe and feature-complete entity domain.

### C2: ITENS improves shared-entity transfer

On GDSCv1 cell/drug entities represented in CTRPv2, ITENS exceeds the strongest
non-ITENS baseline in macro-mean per-drug Spearman.

Support requires a paired drug-level ITENS-minus-baseline estimate above zero after
Holm correction across the two confirmation screens. The full-domain result is
co-primary descriptively but not an ITENS superiority test, because ITENS was designed
to exploit training response neighborhoods.

### C3: explanation stability and biological validity

ITENS top-20 training-edge explanations are more stable across split seeds than
size-, degree-, and expert-composition-matched random edge sets, and show independent
biological enrichment not used in feature construction.

Support requires both: (a) a positive paired stability effect with Holm-adjusted
two-sided p < 0.05; and (b) at least one prespecified independent biological endpoint
with BH-adjusted q < 0.05. Computational reconstruction and deletion fidelity alone
do not support C3.

## Data and identifier rules

- CoderData version 2.2.1, Figshare article version 4, remains frozen.
- Response is `sensitivity = -fit_auc`; all finite successful fits are primary.
- `fit_r2 >= 0.8` is a sensitivity analysis.
- Cell and drug identifiers flagged ambiguous by `audits/identifier_resolution.csv`
  are excluded before splitting. Exclusion is based only on mappings, never outcomes.
- The same canonical Broad expression matrix is used for every model. A target entity
  without that feature source is excluded from feature-model comparisons and counted.
- Drug descriptors are harmonized parent-compound descriptors. Conflicting structures
  are excluded and counted.
- A per-drug correlation requires at least five nonconstant observed responses.

## Models and selection

Every candidate receives identical response rows, cell features, drug features, and
outer splits.

1. Global/additive cell-drug bias.
2. Ridge on PCA expression, PCA descriptors, and bilinear interactions.
3. HistGradientBoosting on the same pair representation.
4. LightGBM on the same pair representation.
5. Feature kNN: cell-only, drug-only, and cell-drug product neighborhoods.
6. Inductive matrix factorization with feature-to-latent projection for unseen entities.
7. ITENS, whose three experts subsume the transparent kNN components.

The primary model is selected by mean macro-per-drug Spearman across source-only inner
`cold_cell`, `cold_drug`, and `cold_both` folds. Tie-breaks, in order, are lower MAE,
smaller parameter count, then lexical configuration ID. Target labels are unavailable
to preprocessing, tuning, model selection, and stopping.

### Fixed source-only hyperparameter grids

- Ridge alpha: 1, 10, 100.
- HistGradientBoosting: learning rate 0.05 or 0.10; leaves 15 or 31; L2 1 or 10;
  150 iterations.
- LightGBM: learning rate 0.05; leaves 15 or 31; min leaf 20 or 50; feature fraction
  0.8; bagging fraction 0.8; L1 0 or 1; L2 1 or 10; 300 rounds.
- kNN/ITENS: k-cell and k-drug each 8, 16, or 32; temperatures 0.05, 0.10, or 0.20;
  ESS shrinkage 0, 10, or 50; ITENS expert weights on a simplex grid in increments
  of 0.25.
- Matrix factorization: rank 16, 32, or 64; L2 0.1, 1, or 10; 30 ALS iterations.
- PCA dimensions remain 64 for comparability; every transform is fitted inside the
  corresponding training partition.

The complete attempted-configuration ledger is retained, including failures. The
chosen configuration is frozen before GDSCv1/NCI60 labels are read.

## Splits and random variation

The ten prespecified seeds are 104729, 130363, 155921, 181081, 205759, 231481,
257053, 282433, 308113, and 333667.

- Internal regimes: random pair, cold cell, cold drug, cold both.
- Cold drug additionally includes a Bemis-Murcko scaffold split when valid structures
  are available.
- External regimes: full feature-complete target, shared entities, and exact shared
  cell-drug pairs.
- Reciprocal exploratory matrix: each of CTRPv2, PRISM, GDSC2, and gCSI is a source
  and each other study a target. It estimates directionality but is not untouched.
- Confirmatory source: CTRPv2 only; targets: GDSCv1 primary and NCI60 secondary.

Deterministic models are refit on every seed's training split for internal evaluation.
For external prediction, stochastic training is repeated over all ten seeds; strictly
deterministic external predictions are reported once and marked deterministic rather
than duplicated to create artificial n.

## Outcomes and inference

Primary outcome: macro-mean per-drug Spearman.

Secondary outcomes: median per-drug Spearman, global Spearman, MAE/RMSE only within a
common response scale, concordance index, calibration slope/intercept where valid,
and subgroup estimates by tissue and response-density quartile.

- Model effects are paired by drug.
- Drug-level BCa bootstrap: 9,999 resamples, seed 20261004.
- A crossed cell-and-drug bootstrap is the sensitivity interval for shared-entity
  conclusions.
- Seed variability is reported separately and is not replaced by row-level bootstrap.
- Holm family 1: all candidate models versus the selected non-graph reference on the
  primary screen.
- Holm family 2: ITENS versus the strongest baseline on GDSCv1 and NCI60.
- Holm family 3: three explanation methods/endpoints.
- Raw and adjusted p-values appear together. Effects and confidence intervals remain
  primary; p-values do not replace them.

## Explanation validation

1. Exact reconstruction: maximum error <= 1e-8.
2. Fidelity: deletion and retention curves at k = 5, 10, 20, and 50, compared with
   random controls matched on edge count, source-cell degree, source-drug degree, and
   ITENS expert composition.
3. Stability: top-k Jaccard and attribution-rank correlation across the ten seeds and
   across external screens on exact matched contexts.
4. Biological validity: drug-target/pathway and DepMap CRISPR dependency evidence
   that was not used to form the response graph or similarity features. The tested
   universe, evidence release, direction, and multiple-testing family are frozen before
   evaluating enrichment.

If items 3 and 4 are unavailable or negative, the manuscript uses "exactly
decomposable" rather than "biologically explainable" and C3 is reported unsupported.

## Stop and reporting rules

- GDSCv1 and NCI60 labels are read only after mappings, model configuration, code,
  and checks are frozen.
- A failed confirmatory endpoint is reported as failed; no replacement target or
  metric is selected.
- Existing PRISM/GDSC2/gCSI estimates remain exploratory regardless of later results.
- All excluded entities, failed curves, unsupported mappings, and missing features are
  enumerated in a flow table.
- The final manuscript distinguishes benchmark conclusions, method conclusions, and
  biological interpretation explicitly.
