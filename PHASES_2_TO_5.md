# VGR research code V2 — implementation ledger (29 October 2026)

## Run from repository root

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python extended.py --phase 2 --dataset synthetic --epochs 30 --out outputs/phase2
python extended.py --phase 3 --dataset synthetic --epochs 30 --out outputs/phase3
python extended.py --phase 4 --dataset synthetic-node --epochs 30 --out outputs/phase4
python extended.py --phase 5 --dataset synthetic-node --epochs 30 --out outputs/phase5
```

All four extended entry points were executed as short CPU synthetic smoke tests. 15 unit tests passed; no real benchmark training was performed, and none of those smoke tests establishes model quality.

## Phase 2: predictive baselines (`vgr/phase2.py`)

- `FullGraphBackbone.forward`: full-graph dense GIN-like, GCN, GraphSAGE and no-edge MLP variants; same training loop and split.
- `majority_result`: majority-class score from training labels only.
- `degree_preserving_rewire`: double-edge swaps preserving degrees, number of edges and simple graph structure, without assuming labels stay correct.
- `compare_baselines`: trains four controls, evaluates a trained GIN on rewired test graphs, AND independently trains a GIN on rewired train/validation graphs. Separate this last control clearly from evaluation-time rewiring.
- These reference backbones are architecture *variants*, not exact reproduction of any particular PyG library defaults or baseline papers.

## Phase 3: explanation integrity (`vgr/phase3.py`)

- `group_removal_audit`: removes multiple edges simultaneously, calculates original-predicted-class confidence reduction and interaction vs the sum of individual removals; samples equal-budget random control groups.
- `concept_interventions`: manipulates the three *explicit structural concept inputs* to the classifier with the learned embedding fixed; an artificial concept-channel test, NOT a physically valid graph edit.
- `motif_gate`: selected-edge precision and recall against *construction-planted* motif edges plus equal-budget random expected precision. `pass_random_precision` is descriptive, not statistically significant.
- `shuffle_label_audit`: shuffles TRAIN and VALIDATION labels only, retrains the model, and evaluates on untouched original TEST labels; a single-run leakage warning, not proof of leakage absence.
- Existing `vgr/audit.py` contains PageRank, deletion, degree, random, saliency and selector rankings, sufficiency/necessity, and edge permutation test.

## Phase 4: node binary tasks (`vgr/node.py`)

- `NodeData`, `validate_node`: node-level single-graph data structure and checks for disjoint train/val/test node IDs.
- `load_node_npz`: load explicit NPZ with `x` (float N×F), `y` (binary N), `edges` (M×2) OR `edge_index` (2×M), optional `sensitive` (0/1/-1; -1 missing), optional `train_idx`,`val_idx`,`test_idx` (integer indices). If splits absent, make reproducible stratified 60/20/20 splits.
- `NodeGNN`: dense node-output message passing (GCN, GIN-like, SAGE, MLP) with no labels used as features automatically.
- `ego_graph`: k-hop neighbors of target, preserving original global node identifiers via a returned mapping. **Important:** ego-graph explanation must be evaluated with a node-target predictor, not by passing the ego graph into a graph-label GNN. Node-specific intervention explanation evaluation is still outstanding.
- `synthetic_node_data` only for smoke testing; **not** a fairness benchmark.
- Real Bail/Credit/German/Pokec adapters are **not implemented** because raw formats, canonical preprocessing and splits have not been inspected. Convert verified source into this NPZ schema without introducing leakage. PyG TU datasets require installation of `torch-geometric` for graph-level runs.
- Dense N×N adjacency is for small graphs. For Pokec-scale experiments, sparse PyG/scatter or neighborhood mini-batching is required (not implemented). Full-batch node training is transductive: val/test node *features and edges* are visible to the encoder (labels are not), as is standard for transductive evaluation; for inductive use, split disjoint graphs and a separate loader.

## Phase 5: conditional fairness (`vgr/fairness.py`)

- `fairness_gaps`: demographic parity (positive-rate gap), equal opportunity (TPR gap), equalized odds (max of TPR/FPR gaps), per-group accuracy, positive rates, TPR/FPR; undefined group/stratum returns None. Metrics are *descriptive disparities* not automatic evidence of unfairness.
- `dp_surrogate`, `eo_surrogate`: differentiable gap in average predicted positive probability; gradients flow to the node GNN. The soft EO term uses labeled positives in the TRAIN split only.
- `train_node`: optimizes supervised BCE and optional DP/EO surrogate. Selects checkpoint using val BCE, and a hard validation accuracy floor when provided. Returns `floor_feasible=False` and the unconstrained best model when floor cannot be met—failure never silently passes.
- `fairness_sweep`: trains a no-penalty baseline, defines validation utility floor from its validation accuracy minus user-specified drop, runs matched-seed penalty coefficients and reports both validation/test metrics. **Do not select penalty strength using TEST performance.**
- This does not implement equalized-odds *training* loss, adversarial-debiasing, fairness certificates, confidence intervals, Pareto optimality certificates, or causal counterfactual fairness. Only DP and EO train-time surrogates are present.

## Full model limitations and remaining work

1. No trained results on IMDb-BINARY, MUTAG, Bail, Credit, German or Pokec verified in this environment.
2. The V1 selector uses a biased straight-through estimator and can collapse. Phase 3 tests do not by themselves establish meaningful selections; require multi-seed, matched-budget, quantitative gates.
3. A learned selected-graph model is not yet directly connected to the node-specific predictor; Phase 4 supplies the correct data and node-head foundations, not the full node-level intrinsic selector.
4. No certified robustness, model-aware attack implementation, calibrated probabilities, structured causal guarantee or full V5 scientific result.
5. Graph-level fairness is available only for valid graph-level group metadata with the V1 coarse surrogate; node fairness is properly scoped to node labels in `vgr/fairness.py`.
6. No automatic fair-dataset download/preprocessing pipeline; external dataset-specific schemas must be verified first.
7. The structural concept tests do not prove human interpretability; test explanatory fidelity independently.

## Data-usage rules

- For IMDb-BINARY, use full-graph task and avoid inventing protected attributes.
- For Bail/Credit/German/Pokec, define the exact sensitive field, target and protocol in metadata; do not confuse a feature correlated with group membership with a pre-verified protected attribute.
- For molecular graphs, not every bond edit preserves the true label. Do not claim perturbations are label preserving by default.
- Preserve historical FRME A0-A5 labels. New experiments here are Phase 2–5 tooling; they do not substitute for V0–V5 experiments or their gates.
