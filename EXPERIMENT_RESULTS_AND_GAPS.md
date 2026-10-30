# Actual synthetic run audit — 29 October 2026

Commands were run on CPU with single-threaded BLAS/PyTorch for deterministic small tests. Data: repository's `synthetic` or `synthetic-node` generators; **NOT real IMDb/MUTAG/German benchmarks**. Most graph runs use seed 0 and 160 generated graphs. Small-sample estimates are noisy and not confidence intervals.

## Reproduced execution

- `python -m unittest discover -s tests -v`: **33/33 passing**.
- Phase 1 V0, 45 epochs: AUROC 0.668; balanced accuracy 0.625 at both default and validation-derived threshold; no single-class collapse.
- Phase 1 V1, 45 epochs: AUROC 0.621; balanced accuracy 0.469 at 0.5 versus 0.656 at validation-derived threshold; no collapse at selected threshold; still weaker ranking than V0.
- Phase 2, 35 epochs: GIN AUROC 0.619, balanced accuracy 0.594; feature-only MLP balanced accuracy 0.500. Rewired GIN AUROC approximately 0.619. Graph-specific benefit is not adequately established by this result.
- Phase 3, 35 epochs: planted-motif edge precision 0.270, random expected precision 0.282. **Explanation gate fails**. Other tests run but do not certify explanation.
- Phase 4 node task, 75 epochs: balanced accuracy 0.875 on an 18-node test split; AUROC 0.975. This is an easy, feature-driven synthetic fixture; it does not validate node explanations.
- Phase 5 fairness sweep, 75 epochs: balanced accuracy 0.875 for penalty strengths 0 / 0.03 / 0.1 / 0.3; test demographic parity gap 0.0779 and equal opportunity gap 0.1333 for all four. **No observed fairness benefit**.
- Phase 6 graph evasion, 35 training epochs, 2 flips per target: 7 initially correct among 10 sampled; random attack success 1/7; greedy attack success 2/7. Semantic label preservation of these graph edits is **not guaranteed**.

## Exact changes from Phases1–6 audited version

1. Training-class-only positive weighting (adjustable, default enabled) in `vgr/trainlib.py`.
2. Validation AUROC-first graph checkpoint selection with BCE tie-break and early-stopping patience, with `selection='bce'` fallback.
3. Original motif diagnostics preserved while adding explicit `research_success=false` to prevent interpreting tiny positive differences as proof of meaningful explanations.
4. Four additional regression/invariant tests. Existing APIs and remaining modules retained.

## What the upgrade **does not** fix

- The hard edge selector remains statistically unreliable on the synthetic motif task; it needs redesigned supervision/optimization and paired trials, not invented results.
- Fairness is node-model-specific, separate from graph-selected intrinsic prediction. A single end-to-end V5 fair/robust/transparent/explainable node model is **not implemented**.
- Phase 6 attacks assess vulnerabilities; they do not provide adversarial training or robustness certification.
- Larger real graph datasets have not been executed in this environment.
- No claim of formal causal validity, certified robustness, or state-of-the-art improvements is justified.

To evaluate this repository on a real benchmark install compatible PyTorch Geometric and run `python main.py --phase 1 --dataset IMDB-BINARY --stage V1 --epochs 100 --out outputs/imdb`. Be sure to check the command-line conventions in `main.py` and `README.md` for each phase; no real download is bundled.
