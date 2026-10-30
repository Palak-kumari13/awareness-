# Implementation audit — 2026-10-29

| Feature | Implementation | Tested here | Remaining |
|---|---|---|---|
| Binary graph predictor (GIN-like) | Yes | CPU smoke, 2 epochs | Real benchmarking / proper hyperparameter search |
| Hard edge selection used by predictor | Yes | Exact budget, gradient, edge permutation | Soft surrogate bias / tie handling, multiple budgets |
| Structural transparency | Three explicit consumed graph channels | Forward pass and tests | Concept interventions, deeper message attribution, concept supervision |
| Explanation fidelity | Keep/drop, removal, matched random, degree, saliency, PageRank | Small synthetic execution | Multi-neighbor interactions, statistical significance, planted recovery gate |
| Synthetic ground truth | Planted triangle vs path | Generator and split tests | More controlled balanced topology and failure diagnosis |
| Fairness training | Soft DP disparity for groups present | Code path available | Node-level datasets, utility-floor constraint, equitable metrics CI |
| Robustness consistency | Random edge flip training | Code path available | Budgeted attacks, edit validity oracle, adversarial training/certification |
| AngleFair | NO | NO | New principled embedding-space definition and ablation |
| Information bottleneck | NO | NO | Variational estimator and ablation only if supported |
| Model-aware adversarial attack | NO | NO | Implement valid edge-flip adversary and budgets |
| Node-level task adapter | NO | NO | Ego graph extraction, target-node head, correct fairness scope |
| Reference-distillation loss | NO | NO | Independent reference checkpoint and matched comparisons |
| Threshold calibration | Brier evaluation only | Yes | ECE, NLL, temperature scaling on validation only |
| MLP / rewired / majority baselines | NO | NO | Implement with frozen splits before claims |
| Label shuffle retraining audit | NO | NO | Separate negative control experiment |
| Across-seed CI | NO | NO | Paired seed aggregation, 5+ seed run |
| Production reproducibility | Config JSON, checkpoint | CPU smoke | Environment lock, complete dataset manifest |

**Observed smoke finding:** Two epochs on synthetic V1 produced test AUROC
~0.627 and constant positive predictions (accuracy 0.50, balanced accuracy
0.50, MCC 0). This explicitly FAILS the classification gate. This short
execution only checks wiring, not learning capability.

Do not represent this as a successful empirical research result. No GPU or
PyTorch Geometric training was completed here.


## Phase 6 added
Implemented: frozen-model undirected edge-flip greedy and random evasion attacks, graph-level and node-targeted, exact budgets, test-time audit, regression tests. Not implemented: poisoning, guaranteed valid edits for real graph labels, strong attack optimizers, certified robustness, adversarial training or benchmark superiority.
