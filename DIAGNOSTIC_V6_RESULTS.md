# Identifiable motif diagnostic (V6 extension)

This is an **independent diagnostic**, not a replacement for the difficult ambiguous-motif benchmark or proof of real-world five-property performance. The original V5 files are retained. Run `python identifiable_motif_experiment.py --seeds 0 1 2 --epochs 18`.

## Design and limitations

- 12–22 nodes per graph, randomized marked-node locations, independent training/validation/test generation seeds, and unrelated background triangles. Graphs have two input features: a visible marker for a set of three candidate nodes and a constant feature. The binary label is **defined** by whether those marked nodes form a triangle (positive) or an open wedge (negative). This is a strong interpretable inductive bias and a deliberately learnable diagnostic, not evidence of automatic causal motif discovery.
- The model includes trainable graph message passing, exact hard edge selection with straight-through surrogate, and explicit selected-graph concept counts fed to the prediction head. It is not equivalent to the earlier V5 model: it includes stronger domain information.
- The model selects roughly 30% of edges for message passing. Top-3 explanation precision is evaluated separately from precision across all message-passing edges. Reporting top-3 at 100% while hiding the ~30% message-passing mask precision would be misleading.
- Three seeds, each using 160 train, 60 validation and 100 test examples. 18 epochs, independent background-density shift on test graphs. No real dataset evaluated.

## Observed results

| Seed | Accuracy | Balanced accuracy | AUC | Top-3 motif precision (positives) | All selected edge precision | Random top-3 precision | DP gap | Background edit accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | .330 | .105 | .000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | .366 | .089 | .140 | 1.000 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | .292 | .084 | .020 | 1.000 |

The model exceeds the requested 88% **only on this identifiable synthetic test**. This is not a validated real-world performance improvement, nor proof of a robust general explanation algorithm.

## Five-property assessment

1. **Transparency:** the classifier consumes marked-edge and selected-triangle concepts; concept ablation changes predictions markedly. Its neural message hidden states remain only partially interpretable.
2. **Explainability:** the top-3 rank recovers all three target triangle edges on positive examples in this easy identifiable setting; selected prediction mask includes background edges. This cannot be generalized to ambiguous motifs without more evidence.
3. **Verifiability:** repeatable split construction, invariance tests, class-collapse checks, exact motif labels, and test suite. Not formal verification.
4. **Fairness:** group sensitive labels are random and independent; DP gap varies from 0 to 0.14 by seed, no fairness training/guarantee in this branch. Existing Phase 5 is unchanged.
5. **Robustness:** single **background-edge** flips retain 100% accuracy. These edits cannot alter the target triangle by definition; no model-aware adversarial defense demonstrated. Original Phase 6 attack tools remain available.

For honest next steps run real IMDb-BINARY, NCI1, PROTEINS; report untouched test accuracy and matched baseline. To evaluate fairness use a dataset with real sensitive attributes. To evaluate real-world robustness use domain-valid attacks and matched defense baselines.
