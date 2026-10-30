# Stability upgrade: verified changes and remaining gaps

This revision preserves the existing Phases 1–6 APIs. Updates:

- `vgr.trainlib.train`: class weights computed ONLY on train labels, validation AUROC-first checkpoint selection (validation BCE ties), configurable fallback to BCE selection and early stopping. It does not force or fabricate discrimination.
- `vgr.phase3.motif_gate`: retains descriptive `pass_random_precision` but adds `research_success=false` until paired multi-seed evidence and non-collapsed prediction exist. Previously a 0.7 percentage point improvement could be misread as scientific success.
- `tests/test_stability_upgrade.py`: verifies the new checkpoint metadata and collapse checks.

**Not fixed or claimed:** the original hard selector has not been proven faithful; node fairness and graph selector remain separate models; Phase 6 attacks evaluate robustness but do not create a robust defense; full V5 joint model, formal verification, ground-truth real-world causality and benchmark superiority remain undone.

The original synthetic generator plants a triangle in positive graphs and an open path in negative graphs. This benchmark is easy to shortcut using global triangle counts and should not be used as sole evidence of explanation fidelity. IMDb/MUTAG/German are not bundled. Real benchmark validation requires compatible datasets and dependency installation.
