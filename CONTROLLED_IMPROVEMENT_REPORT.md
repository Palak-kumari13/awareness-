# Controlled binary improvement experiment (29 October 2026)

## What was added
- `vgr/guardrails.py`: validation-only threshold selection; binary discriminative/constant-prediction checks; non-inferiority-constrained fairness candidate selection with baseline fallback. Candidate selection is never performed with test labels.
- `controlled_binary.py`: fixed candidate grid using one selected-node predictor, identical seeds/splits/epochs; fairness-only and fairness-plus-consistency variants.
- `tests/test_guardrails.py`: three new checks. Existing modules are untouched.

## Experiment executed
Dataset: `synthetic_node_data` (100 nodes, constructed edges, binary labels and synthetic groups). Seeds 0,1,2. 60 epochs per candidate; five candidates per seed. This dataset is not external validation and its sensitive labels are a synthetic attribute.

| Seed | Chosen variant | Baseline test balanced accuracy | Chosen test balanced accuracy | Baseline test DP gap | Chosen test DP gap |
|---|---:|---:|---:|---:|---:|
| 0 | baseline | 0.9000 | 0.9000 | 0.0808 | 0.0808 |
| 1 | baseline | 0.9444 | 0.9444 | 0.1212 | 0.1212 |
| 2 | baseline | 0.8990 | 0.8990 | 0.0330 | 0.0330 |

All three seeds selected the unregularized model on validation data. Thus no performance degradation relative to baseline in this comparison, **but no demonstrated improvement** in fairness, robustness, explanations, or transparency either. This is a control feature, not a new GNN advance. The test non-regression check is descriptive and is not guaranteed a priori from validation selection; on other data, test regression can occur.

## What still fails / is absent
1. Learned selector failed earlier binary planted-motif fidelity test. This update does not repair selector learning.
2. Fairness improvement is not demonstrated. Candidate comparison did not satisfy validation constraints.
3. Robustness training objective exists, but independent adversarial-defense gains were not established here.
4. Concept explanations are not shown to be faithful; the three concept features are not a complete transparent computation trace.
5. No real-world dataset or cross-seed confidence intervals; no claim of five-property completion.
6. Existing code is O(N²) dense for node tasks, so scalability work is required.

## Design constraints and next experiment
For a genuine advance, first train identical variants on planted binary motifs with independent selected-edge recovery and full/keep/drop matched controls; freeze splits, budgets and seed grid. Next run an isolated fairness grid on the *same* intrinsically selected node model, keeping a utility floor and calibrating decisions only from validation. Independently test robustness with a held-out attack family and fixed budget. Do not merge unrelated synthetic results into a claimed single-model five-property score.
