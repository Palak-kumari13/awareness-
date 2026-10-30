# Phase 6 — Test-time structural robustness evaluation

This is **untargeted, model-aware greedy edge-flip evasion** versus equal-budget uniform random edge flips. The attacker maximizes supervised BCE against a **frozen trained model**, by exhaustively testing currently eligible single-pair flips and making the best choice at each step; this is a discrete greedy heuristic, not a globally optimal attack or Nettack reproduction. The random control samples eligible pairs uniformly without replacement.

- `vgr.phase6.attack_budget`: explicit number or edge-relative rate (rounded, may be zero).
- `edit_edges`: symmetric undirected addition/deletion; canonical unique edges.
- `graph_attack`: graph-label attack; selector is recomputed for every candidate.
- `node_attack`: fixed original target ego-node set; full graph predictor; target test label only for attacker objective.
- `summarize`: attack success among *initially correctly classified* cases, and prediction-probability drift.
- `phase6.py`: trains existing predictors using previous code then runs frozen-model attacks; includes random and greedy comparison.
- `tests/test_phase6.py`: exact flip budget, identity, determinism, model mode restoration, immutability and node-ID tests.

Assumptions: single graph, undirected simple graph, binary labels, full-batch dense operations. Attacks may be expensive for large graphs. There is **no** legitimate-label preservation oracle for real datasets, no claim of biological/social validity, no training-time poisoning, no adversarial training, no certified robustness and no claim of superior robust accuracy. The example attack is a *white-box-by-query* attack with true labels available to the benchmark evaluator; a real attacker might have weaker information.

Run:

```bash
python main.py --phase 6 --task graph --dataset synthetic --epochs 3 --examples 2 --flip-budget 1 --out outputs/p6graph
python main.py --phase 6 --task node --dataset synthetic-node --epochs 3 --examples 2 --flip-budget 1 --out outputs/p6node
```

The previous Phase 1-5 experiment modules are preserved unchanged. Only `main.py` is minimally extended to dispatch Phase 6.
