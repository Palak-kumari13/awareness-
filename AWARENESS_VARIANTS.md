# V7 awareness experiments (V6 remains unchanged)

- `vgr/awareness.py`: topology-aware, motif-aware, spectral-aware and hybrid edge features. The code has no access to target labels or hidden social-community membership.
- `awareness_experiment.py`: shared two-layer message-passing GNN, hard selected-edge predictor, training, classification metrics and top-3-vs-random deletion drift.
- **Topology:** endpoint degree, clustering and common-neighbor features.
- **Motif:** adds local triangle support.
- **Spectral:** adds eigenvector-based spectral distance; sign-invariant squared distances, but degenerate eigenvalues can still cause instability; don't claim full permutation/eigenspace invariance without testing.
- **Hybrid:** combines topology, triangle and spectral features.

All explicit classifier concept channels are computed from the selected adjacency. Scoring is permitted to examine the original input adjacency, but no full-graph concept channel directly reaches the classifier. The straight-through selection estimator is biased and its edge deletion fidelity is NOT guaranteed.

## Run
```
python awareness_experiment.py --seeds 0 1 2 --epochs 12 --train-size 80 --val-size 30 --test-size 40 --out outputs/awareness_comparison.json
python -m unittest discover -s tests -q
```

## Verification limits
The 3-seed comparison is an experiment on simulated assortative/disassortative stochastic block models, **not** empirical fairness or robustness. Accuracy alone is not enough for five-property claims. Deletion drift is a diagnostic, not a causality measurement. The baseline V6 script and historical files remain unchanged. Dataset generation gives a structural statistical signal and is not equivalent to unmarked IMDb.
