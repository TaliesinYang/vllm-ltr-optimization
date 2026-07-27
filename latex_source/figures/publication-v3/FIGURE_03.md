# Figure 3 — Legacy scheduling trade-off

## Purpose

Show the measured baseline trade-off in one full-width, horizontal two-panel figure: shortest-first ordering lowers mean TTFT near saturation, while tail TPOT becomes worse at the highest tested load.

## Data

- `scripts/report_figures/data/offline/baseline-2026-06-22-RESULTS-summary.txt`
- Parsed by `scripts.plot_final_report_figures.parse_baseline_summary`.
- Rendered independently by `scripts/report_figures/publication_v3/figure_03.py`; publication-v2 files are not modified.

## How to read

Both panels show the same measured request-rate sweep and use logarithmic axes. Panel (a) marks 16 queries/s as the onset-of-saturation comparison: legacy shortest-first has 6,043.7 ms mean TTFT versus 17,274.2 ms for Stock FCFS, or 65.0% lower. Panel (b) uses the distinct 64 queries/s highest-tested-load comparison: legacy shortest-first has 1,400.29 ms p99 TPOT versus 171.19 ms for Stock FCFS, or 8.18× higher.

## Result

The sweep supports a queueing-latency benefit near saturation and a separate tail per-token cost at the highest tested load. The two annotations deliberately do not imply that 16 and 64 queries/s are the same operating point.

## Limitation

This is one legacy sweep with no repeated runs, so no repeat-level confidence interval is available. The only scope text retained inside the figure is `single sweep · no repeated runs`.

## Reproducibility

Run:

```bash
python -B scripts/report_figures/publication_v3/figure_03.py
```

The generator writes `fig3.svg`, `fig3.pdf`, and a 300 dpi `fig3.png` on a fixed 181.9 × 84.0 mm IEEE double-column canvas. SVG text remains editable.
