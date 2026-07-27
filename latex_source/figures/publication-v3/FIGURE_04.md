# Figure 4 — Predictor ranking and validation-only data-scale ablation

## Purpose

Separate two questions on one IEEE full-width canvas: panel (a) compares Tier-2 held-out test ranking across predictor/input variants; panel (b) shows a distinct single-seed validation learning curve for full context.

## Data

- `scripts/report_figures/data/offline/tier2-matrix-summary.json`
- `scripts/report_figures/data/offline/tier2-learning-curve.json`
- `scripts/report_figures/data/offline/tier1-matrix-summary.json` is completeness-audited by the reused loader but is not plotted.
- Renderer: `scripts/report_figures/publication_v3/figure_04.py`.

## How to read

Panel (a) is a horizontal point-range plot. Filled circles are BERT seed means; open circles are the three observed seeds and the whisker spans their observed minimum–maximum. LightGBM structural has one seed and is shown as a diamond. A broken x-axis gives the BERT ranges enough resolution without hiding the lower LightGBM result.

Panel (b) is explicitly labeled `validation only · single seed · not deployment evidence`. Its four major x ticks are only the nominal pool sizes 500/1,000/2,000/4,000. Effective analyzed counts appear once as a micro note: 499/999/1,997/3,997.

## Result

Held-out Tier-2 test means are 0.587 for prompt only, 0.630 for prompt + schema, 0.626 for full context, and 0.427 for the single-seed LightGBM structural baseline. Prompt + schema has the highest mean, while its observed seed range overlaps the full-context range.

The separate full-context seed-42 validation curve is 0.605, 0.610, 0.637, and 0.632, so the largest nominal pool does not continue upward.

## Limitation

BERT ranges are observed min–max across three seeds, not confidence intervals. LightGBM has one seed. Panel (b) is validation-only, single-seed evidence and does not identify a deployed predictor.

## Reproducibility

Run:

```bash
python -B scripts/report_figures/publication_v3/figure_04.py
```

The generator writes `fig4.svg`, `fig4.pdf`, and a 300 dpi `fig4.png` on a fixed 181.9 × 86.0 mm canvas. SVG text remains editable.
