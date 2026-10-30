# AngleFair integration: evidence-led audit

## Source inspection
Uploaded `Angle_FairGNN(2).zip` contains only three results CSV files, no Python source for the original principal-angle loss. The attached FRME report says AngleFair was exploratory and masks were degenerate in earlier runs. **This addition is a new, explicitly defined mask-independent principal-angle group-alignment surrogate, not a reproduction of the original unpublished loss.**

## What changed
- New `vgr/anglefair.py`: `principal_angle_alignment()` compares per-class, per-sensitive-group embedding subspaces via squared cosines of principal angles. Minimizes mean squared sine of principal angles. Skips classes with insufficient group support. `anglefair_training_step()` adds the loss for train-index embeddings only.
- `vgr/integrated.py`: `SelectedNodeGNN.forward(..., return_embeddings=True)` exposes the representation without changing the standard forward output. `train_selected_node(..., angle_weight=0.0)` enables the optional regularizer; default remains exactly zero.
- `anglefair_ablation.py` compares fixed weights with the same synthetic splits/seeds; never selects weights using test scores.
- `tests/test_anglefair.py`: finite gradients, missing-group guard, model backward, no sensitive-group handling.
- Historical CSVs retained byte-for-byte under `historical_anglefair/`. No historical experiments relabeled as successful.

## Mathematical interpretation
With representation matrices for label y and protected groups s=0,1, take top r right singular vectors U_y0 and U_y1 from centered embeddings, then penalize `1 - mean(svdvals(U_y0.T @ U_y1)^2)` per eligible label. This encourages **within-label group subspace alignment**; it is NOT a proof of counterfactual fairness, demographic parity, or equal opportunity. Fairness gaps and classification utility must be evaluated independently.

## Executed checks
44 unit tests passed. Short 20-epoch synthetic pairwise comparison (seeds 0,1; no multi-seed confidence interval):

| Seed | Angle weight | Test balanced accuracy | Test demographic parity gap |
|---|---:|---:|---:|
| 0 | 0 | .9000 | .0808 |
| 0 | .01 | .9500 | .3939 |
| 1 | 0 | .8434 | .1212 |
| 1 | .01 | .8889 | .0303 |

The metric effects are **inconsistent**: seed 0 loses substantial demographic parity, so AngleFair does **not** pass an across-seed fairness nonregression gate. Default remains disabled; the ablation is included for research comparison only. No real dataset runs or claim of improvement.

## Excluded from main model
- Historical failed mask-space AngleFair and collapsed A1-A5 masks: no evidence of reliable edge attribution.
- Historical distillation and 90-run alternative-mask approaches: not validated as beneficial.
- GeoXGNN-inspired changes: historical report proposes them but does not establish implementation/effectiveness.
- Unverified claims of certified fairness/robustness/causality.

## Run
```
python -m unittest discover -s tests -q
python anglefair_ablation.py --seeds 0 1 2 --epochs 60 --weights 0 .001 .01 --out anglefair_results.json
```
Use `train_selected_node(data, angle_weight=0.01)` only as an explicit ablation. It is not active in default runs.
