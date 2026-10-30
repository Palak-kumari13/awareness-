# Decision-collapse safeguard, implemented 29 October 2026

The new `vgr/decision.py` supplies `choose_threshold`, `classification_audit`, `prediction_probabilities` and `node_prediction_probabilities`. `audit_decisions.py` is a separate executable for validation-selected thresholds. Existing graph Phase 1 `run.py` adds both default-threshold and validation-selected test audits to `metrics.json`. Phase 5 `fairness_sweep()` now records a decision audit for each weight and flags `fairness_success_claim_allowed` only if its existing validation utility floor is feasible and the test discrimination gate passes. These flags DO NOT prove fairness improvement.

## Honest scope of this fix

* **Fixed detection/reporting and some threshold selection:** All-positive/all-negative predictions are explicitly flagged; selecting a threshold uses validation labels exclusively. The test set is never searched for a good threshold. Validation threshold selection is only meaningful when both classes are present. The selected threshold is not a fix for a model with no score discrimination.
* **Not fixed:** a collapsed model's learned representations, explanation fidelity, structural selector collapse, fairness/robustness objective integration, data leakage or graph topology dependence. Improving these requires training changes and real benchmark experiments.
* `classification_audit` requires balanced accuracy >=0.55 and recall >=0.10 per class by default, and never passes a constant predictor. These thresholds are demonstration gates, not publication standards.
* Neither accuracy thresholds nor selection of hyperparameters based on test performance is appropriate. Hyperparameter selection must remain on training/validation only; test audit is an evaluation report, not a tune-loop.
* The existing `.metrics()` function and training optimizers are preserved; neither automatically learns calibrated probabilities nor introduces a full imbalance-aware training objective. Consider `pos_weight` estimated from **training labels only**, balanced minibatches, and early stopping using a prespecified validation metric after baseline comparisons.

## Commands

```sh
python -m unittest discover -s tests -v
python audit_decisions.py --task graph --dataset synthetic --epochs 40 --out outputs/graph_decision
python audit_decisions.py --task node --dataset synthetic-node --epochs 40 --out outputs/node_decision
python main.py --phase 1 --dataset synthetic --stage V1 --epochs 40 --out outputs/graph
python main.py --phase 5 --dataset synthetic-node --epochs 40 --out outputs/fairness
```

## What still needs implementation and validation for five properties

| Property | Implemented to date | Missing for full integrated GNN |
|---|---|---|
| Transparency | Three structural channels consumed by graph predictor, diagnostic interventions | Task-level interpretable concept bottleneck with audited semantics; message-level contribution inspection and validation of concept faithfulness on held-out graphs; node-level implementation |
| Explainability | Graph hard selector, keep/drop and matched controls, group removal, synthetic motif tests | Resolve selector failures; repeated seeds, meaningful motif gains, informative selection and combined node-level intrinsic selector; intervention effects for explanations and faithful-reference comparisons |
| Verifiability | Graph invariants, permutation, shuffled label, controls and decision gates | End-to-end versioned pipeline; split/leakage/edge identity contracts for each real dataset; statistical uncertainty, reproducible checkpoints and all-parameter tests; formal certificates only if separately developed |
| Fairness | Node-level DP/EO penalties and gap metrics with utility floor | Joint training with intrinsic selector and transparency; supported sensitive-group definitions on canonical benchmark adapters, Pareto and group-conditional performance, audit fairness at validation-selected decision thresholds, multiple seeds and evidence of improvement |
| Robustness | Structural flip attack evaluator and initial consistency penalty | Integrated adversarial training, restricted attack model and valid edit constraints, meaningful denominator for attack success, comparison of attacked/clean utility and explanation stability, attack-budget sweeps and more realistic graph-specific validity |

Graph-level IMDb/MUTAG lack established sensitive attributes: don't manufacture fairness ground truth. Use German initially for node-level fairness but it needs canonical preprocessing. Graph datasets need graph-level selected subgraphs; node datasets need node-level selectors over relevant computational ego graphs. Real datasets and PyG were not benchmarked as part of this audit.

## Smoke execution evidence

On the repository's *synthetic* data, 8-epoch seed-0 runs (not convergence studies): graph selected threshold 0.50365, default vs selected test balanced accuracy both 0.5625; predicted positive rate 93.75% vs 81.25%. Node selected threshold 0.44724, default vs selected test balanced accuracy 0.75 vs 0.90; predicted positive rate 22.22% vs 55.56%. These vary with training and split; neither outcome demonstrates real-world model quality or successful fairness/robustness.
