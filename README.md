# VerifiedGraphResearch — Unified Phases 1–5

This ZIP combines the **existing Phase 1 graph classifier** and **Phases 2–5 experimental modules**, in one runnable Python repository. It is **not** a completed or validated unified V5 architecture. Phase 6 adversarial attack integration and Phase 7 publication-grade experiments remain outstanding.

## Installation and tests

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## One entry point

```bash
python main.py --phase 1 --dataset synthetic --stage V2 --epochs 30 --out outputs/p1
python main.py --phase 2 --dataset synthetic --epochs 30 --out outputs/p2
python main.py --phase 3 --dataset synthetic --epochs 30 --out outputs/p3
python main.py --phase 4 --dataset synthetic-node --epochs 30 --out outputs/p4
python main.py --phase 5 --dataset synthetic-node --epochs 30 --out outputs/p5
```

Phase 1 retains the original `--stage V0/V1/V2/V3/V4/V5` switch, which is distinct from the numbered development phases 1–5. For real node datasets Phase 4/5 accept `--npz /path/to/dataset.npz`; inspect the schema in `PHASES_2_TO_5.md`. The TU graph dataset loader requires optional PyTorch Geometric.

## Folder map

- `main.py`: unified entry point, dispatches to existing runners without duplicating experiments
- `run.py`: original Phase 1 V0–V5 graph experiment runner
- `extended.py`: Phase 2–5 experiment runner
- `vgr/core.py`, `data.py`, `trainlib.py`, `audit.py`: Phase 1 model, data, training and intervention code
- `vgr/phase2.py`: prediction baselines and rewiring
- `vgr/phase3.py`: group removal, concept interventions, motif and leakage checks
- `vgr/node.py`: node classifier and NPZ adapter
- `vgr/fairness.py`: group metrics and fairness regularization
- `tests/`: original 15 invariant and phase tests
- `STATUS.md`, `PHASES_2_TO_5.md`: honest implementation notes and unimplemented work

## Research validity notice

Unit/smoke tests do not prove explainability, fairness, robustness, causality, or superior performance. Real dataset training, graph-level and node-level generalization, and V0–V5 empirical gates remain to be established.


## Phase 6 structural attacks
See `PHASE_6.md`. Use `python main.py --phase 6 --task graph|node ...`. No earlier phase modules changed.

## Experimental integrated branch (separate from historical phases)

See `INTEGRATED_BRANCH_STATUS.md`. Run `python integrated_experiment.py --task multilabel --epochs 55 --graphs 160` for independent two-label graph classification or `python integrated_experiment.py --task node --epochs 100` for a node-level hard-selector model with optional fairness and random-perturbation consistency. These additions are functional research code, **not verified fairness, robustness or transparency claims**. Actual results, including failing fairness observations, are documented in the status file.

## Controlled binary experiment (new)
Run `python controlled_binary.py --seeds 0 1 2 --epochs 60 --out controlled_binary_results.json` and inspect `CONTROLLED_IMPROVEMENT_REPORT.md`. The nonregression guardrails select among fixed candidates using validation data only. If no candidate improves the validation fairness metric without losing classification utility, the experiment retains the baseline. This does **not** mean fairness was improved or guaranteed on the test set.

## Optional AngleFair ablation
See `ANGLEFAIR_INTEGRATION_STATUS.md`; default `angle_weight=0` preserves prior training. The uploaded AngleFair ZIP contained historical CSVs only. The new mask-independent principal-angle loss is an experimental surrogate, not a verified fairness improvement.


## AngleFair / FRME compatibility integration
See `MERGE_AUDIT.md` for new fail-closed validation gates and frozen edge intervention functions. Original research modules are preserved, and historical failed mask methods are not re-enabled.

## Randomized synthetic selector-repair experiments

See `SELECTOR_REPAIR_STATUS.md`. Run `python selector_repair_experiment.py --seeds 0 1 2 3 4 --epochs 100 --graphs 180 --hidden 64 --budget .5 --out outputs/selector_comparison.json`. All earlier phases remain unchanged; this is an isolated diagnostic branch and does not constitute successful 88% classification.

## V6 identifiable-motif diagnostic (separate from V5 difficult task)

`python identifiable_motif_experiment.py --seeds 0 1 2 --epochs 18 --out identifiable_results.json`

See `DIAGNOSTIC_V6_RESULTS.md` for synthetic results, confounds, limitations and per-property evaluation. This purpose-built attribute-identified triangle task reaches 100% test accuracy in three seeds but must NOT be presented as 88% general graph classification accuracy or full five-property validation.
