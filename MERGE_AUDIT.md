# AngleFair / FRME integration audit

## Provenance
- Base: local `VerifiedGraphResearch_AngleFair_Optional_Ablation.zip`, extracted intact.
- External references: https://github.com/Palak-kumari13/Angle-FairGNN and https://github.com/Palak-kumari13/FRME-GNN-fair-and-stable-
- External repository archives were not downloadable in the execution environment; no claim of verbatim imported GitHub modules.
- Existing `vgr/anglefair.py` is a separately defined class-conditional, mask-independent principal-angle surrogate; **not** a reconstruction of historical AngleFair loss.

## New code
- `vgr/research_gates.py`: adaptation of validation-only global fallback concept; rejects collapse/utility regressions, optional group and drift gates. Explanation certification is separate and requires matched controls.
- `vgr/frme_controls.py`: FRME-inspired frozen-model edge removal and pair-interaction measurement; not the original FRME graph reconstruction bridge.
- `tests/test_research_merge.py`: fail-closed gating and graph intervention invariants.

## Deliberately excluded from active training
- Historical collapsed soft edge masks; distillation as a default; unverified graph reconstruction objective; certified fairness/robustness/causality claims.
- Archived CSVs kept for historical reference only.

## Usage
```bash
python -m unittest discover -s tests -v
python anglefair_ablation.py --seeds 0 1 2 --epochs 60 --weights 0 0.001 0.01 --out anglefair_results.json
```

## Limitations
- Validation gate utility metrics are user-supplied validation records. It does not promise test-set nonregression.
- Principal-angle optimization may have unstable gradients under close singular values.
- Graph intervention controls inspect frozen model dependence, not physical causality.
- No new multi-seed benchmark improvement has been demonstrated by this merge.
