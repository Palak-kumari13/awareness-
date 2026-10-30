# Selector Repair — implementation and scientific status

## Implemented
- `vgr/selector_repair.py`: randomized motif positions, variable node counts/background edge probabilities, optional misleading triangles in negative graphs, deterministic held-out generator streams, Bernoulli teacher KL, three selector training variants, and same-budget random explanation comparisons.
- `selector_repair_experiment.py`: full-graph GNN teacher/reference trained on train data, joint selector baseline, alternating training, teacher-guided training, fixed split per seed; per-run test metrics and planted-edge precision. The class labels are binary. The hard selector remains a biased straight-through estimator.
- `tests/test_selector_repair.py`: five new functional checks. The existing source code and historical experiment outputs are preserved.

## Known limitations and important constraints
- The alternative methods are implementations to test, **not proven improvements**. They do not directly supervise motif masks. The randomized generator with hard-negative background triangles can be genuinely ambiguous: no GNN can identify an arbitrary planted triangle from identical triangles without additional signal. Ground-truth explanation recovery must be interpreted accordingly. 
- The existing GNN `edge_scores()` uses endpoint-projected features and node degrees. With constant features and symmetric topologies it can tie or nearly tie. The tiny score spread remains a limitation. This is not fixed by alternating optimization or distillation alone.
- The current synthetic graph labels come from the generator. A graph may contain a triangle in a negative class as a hard negative, so global presence-of-triangle is not a sufficient rule.
- The CLI does not silently search hyperparameters against the test set; provided settings are fixed before test evaluation. Use validation-only model selection for sweeps and preserve final untouched test sets. Five seeds and real-dataset tests still required.
- AngleFair/fairness and adversarial robustness are not enabled in this isolated experiment, as requested. Original modules remain available.
- `full_graph` is a full-graph comparator, not an intrinsically selected explainer. Teacher-guided training aligns probability distributions but does not validate motif recovery.

## Verified this session
- `python -m unittest discover -s tests -q`: **54 tests passed**, five newly added.
- Completed small smoke run: seed 0, 24 training graphs, 8 epochs, 40 held-out synthetic graphs. Joint AUROC 0.456, alternate 0.471, teacher-guided 0.456; balanced accuracy 0.5 for all; motif precision 0.161/0.183/0.161 versus random expectation 0.314. **The 88% gate fails**.
- A longer 90-training-graph, 30-epoch comparison was interrupted by runtime limit after seven of nine variant-seed combinations. Its partial log is included as `outputs/partial_long_run.log` but is not presented as a complete multi-seed evaluation.

## Commands
```
python -m unittest discover -s tests -q
python selector_repair_experiment.py --seeds 0 1 2 3 4 --epochs 100 --graphs 180 --hidden 64 --layers 2 --budget 0.5 --lr 0.001 --out outputs/full_comparison.json
```
The full run is CPU intensive. Any hypothesis about 88% classification or explanation precision is **unverified** until held-out test results support it.
