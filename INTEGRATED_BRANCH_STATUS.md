# Integrated research branch — measured results and limitations

This branch ADDS `vgr/integrated.py`, `integrated_experiment.py`, `integrated_diagnostics.py`, and `tests/test_integrated.py`. It does NOT replace the original Phase 1–6 pipeline. It is an experimental combined selector, predictor, fairness and stochastic-perturbation-training implementation.

## Verified execution (CPU, 29 October 2026)

* Automated suite: **37/37** passed (previous 33 + four new tests).
* Two independently planted labels: triangle and four-cycle on synthetic graphs. Multilabel uses two sigmoid outputs, independent per-label BCE, macro/micro F1, per-label AUROC and subset accuracy. Multilabel is NOT multi-class, and these are not real clinical/social outcomes.
* Multilabel holdout seed 0, 55 epochs, 160 samples: macro-F1 **0.7674**, subset accuracy **0.5938**; label AUROCs **0.7344 / 0.8164**.
* Same setup seed 1: macro-F1 **0.4990**, subset accuracy **0.4688**. Results are unstable; no statistical superiority established.
* Node selected-subgraph predictor, seed 0, 100 epochs, n=100: test balanced accuracy **0.90**, AUROC **0.99**, demographic parity gap **0.2828** (fairness weight 0.05, consistency 0.1).
* Matched node model without fairness/consistency: balanced accuracy **0.90**, AUROC **0.95**, DP gap **0.0808**. **Fairness intervention did not improve DP**, so a five-property success claim is not supported.
* Selector planted-motif precision on 40 graph holdouts (diagnostic setup): **0.6079** vs expected random **0.4255**. This is one run and NOT a significance test; labels may be recoverable by construction-specific shortcuts.
* Random 5% edge-deletion node probability drift, 8 perturbations: **0.01123** (no regularization) vs **0.00998** (fairness+consistency). Joint changes do not isolate the robustness effect. Not a model-aware adversarial robustness result.

## What was added

`planted_multilabel(n,seed)` creates two independently varying planted motifs, plus known planted-edge indicators. `MultiLabelGNN` reuses the graph selector/predictor but emits two logits. `train_graph_multilabel` applies independent BCE and an optional random-edit consistency penalty. `evaluate_multilabel` returns AUROC, positive rate, per-label F1, macro/micro F1 and subset accuracy. `SelectedNodeGNN` performs hard top-k selection inside the SAME node predictor. `train_selected_node` trains it with optional DP/EO fairness surrogate and random-edge consistency. `report_node` wraps existing metrics. `motif_audit` evaluates selected-edge precision against random expectation and `edit_stability` evaluates change in probabilities for random deletions. `integrated_experiment.py` exposes runnable entry points.

## Limitations that MUST NOT be hidden

1. Node fairness and stochastic edit robustness are jointly implemented, **not validated as effective**. Demographic-parity gap increased in the matched run.
2. The hard top-k mechanism uses a biased straight-through gradient; masks can remain unfaithful. Score spread, matched random/degree/saliency explanation comparisons and paired multi-seed confidence intervals remain important missing gates.
3. Real-world benchmark results: **none** in this branch. The current synthetic node generator is feature-driven, not a well-calibrated structural fairness test.
4. Training and validation losses are tracked; more rigorous model selection and threshold calibration are still required. Node threshold is 0.5 in this branch.
5. Model-aware greedy attacks exist in the original Phase 6, but are not yet exercised against this new unified node model in a multi-seed benchmark. Tested edit stability is from RANDOM deletions only.
6. Transparency remains limited to exposed concepts in the graph branch. This branch does not prove faithful message attribution or end-to-end concept sufficiency.
7. No certified robustness, semantic label-preservation oracle or real-world causal identification.
8. Dense O(N^2) node adjacency is not scalable to large networks.
9. The historical baseline Phase 1–6 scripts are unchanged. Their existing short-run and collapse limitations do not disappear by adding this branch.

## Run

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -q
python integrated_experiment.py --task multilabel --epochs 55 --graphs 160 --output integrated_multi.json
python integrated_experiment.py --task node --epochs 100 --output integrated_node.json
python integrated_diagnostics.py
```

## Required before paper claims

Train five paired seeds with frozen splits and baseline-matched hyperparameter budgets; test full graph vs selected graph on real MUTAG and IMDb-BINARY, and integrated node fairness on German after trusted preprocessing. Measure per-group utility, statistical uncertainty, robust edit/greedy attack success conditional on initially correct predictions, threshold calibration, random/degree/PageRank controls, and structural motif fidelity. Do not tune on test data.
