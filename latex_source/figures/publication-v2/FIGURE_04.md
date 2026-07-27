# Figure 4 — Predictor selection and data scale

Internal evidence note only; not report prose.

## Purpose

Compare predictor/input choices on the Tier-2 held-out test set, then separately document data-scale sensitivity in one full-context, seed-42 learning-curve ablation. Panel (b) does not identify the selected or deployed predictor.

## Data

- `scripts/report_figures/data/offline/tier2-matrix-summary.json`
- `scripts/report_figures/data/offline/tier2-learning-curve.json`
- `scripts/report_figures/data/offline/tier1-matrix-summary.json` is completeness-audited by the renderer but not plotted, preventing Tier-1 validation tau from being mixed with Tier-2 test tau.

## Why this experiment

The two panels answer different questions: panel (a) compares predictor/input candidates on held-out test data; panel (b) checks data-scale behavior for one fixed full-context seed-42 ablation. The ablation is not evidence that full-context was selected or deployed.

## How to read

Panel (a) dots are Tier-2 held-out test Kendall tau-b means. BERT whiskers are the observed seed minimum and maximum for seeds 17, 42, and 73; they are not confidence intervals. LightGBM has one seed (42), so it has no seed range. Panel (b) is a separate seed-42 full-context validation learning-curve ablation. Its x-axis uses nominal pool sizes 500/1000/2000/4000, and each tick discloses the effective n=499/999/1997/3997 after exclusions.

## Result

Tier-2 test means are 0.587 prompt-only, 0.630 prompt+schema, and 0.626 full-context; LightGBM test tau-b is 0.426799. Separately, the seed-42 full-context validation ablation is 0.605171, 0.610124, 0.637282, and 0.632051 at nominal pools 500, 1000, 2000, and 4000 (effective n=499, 999, 1997, and 3997), so its final step does not continue upward. This ablation does not establish deployment choice.

## Limitation

Panel (a) exposes only a three-seed min–max range for BERT and one LightGBM seed. Panel (b) is a single-seed learning curve and uses validation, not held-out test, tau-b.

## Reproducibility

Run `python scripts/report_figures/publication_v2/figures_04_06.py`. The renderer writes `fig4.pdf` and a 300 dpi `fig4.png` under this directory.

## Tomorrow's one-line explanation

Prompt plus schema has the highest Tier-2 test mean; separately, one full-context seed-42 ablation plateaus by its largest nominal pool and says nothing about which predictor was deployed.
