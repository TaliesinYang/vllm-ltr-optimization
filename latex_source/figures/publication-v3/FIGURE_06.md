# Figure 6 — One invalid-provenance BFCL subset

## Purpose

Report paired TTLT reductions on one common-complete BFCL subset without presenting invalid-provenance source summaries as general OOD evidence.

## Data

- `scripts/report_figures/data/rental-20260719T231309Z/matrix-ood/*.runs/*.samples.csv`
- `scripts/report_figures/data/rental-20260719T231309Z/matrix-ood/{StockFCFSShim,PureLTRScheduler,GatedHybridScheduler,TailSafeScheduler}.json`
- Renderer: `scripts/report_figures/publication_v3/figure_06.py`.

All four source summaries have `valid=false`. Seven error rows are excluded before constructing the common-complete subset.

## How to read

Both panels retain only the 119 request IDs completed by all four policies in each of three repeats. Panel (a) shows paired pooled-mean TTLT reduction relative to the matched FCFS shim; panel (b) shows the paired pooled-p99 reduction. Positive values mean lower TTLT. Filled circles are pooled paired estimates. Open circle/square/diamond markers expose the three repeat-matched estimates. Whiskers are paired hierarchical 95% intervals from 2,000 deterministic resamples with seed 1234: repeat and request-ID clusters are sampled jointly, with identical draws applied to a policy and the FCFS shim. The intervals therefore quantify percentage reduction directly; they are not absolute-latency intervals relabeled as paired effects.

## Result

On the common-complete subset, the three learned policies have 19.0–19.6% lower pooled mean TTLT and 26.4–26.9% lower pooled p99 TTLT than the FCFS shim.

## Limitation

This is one BFCL workload with invalid source-summary provenance, seven excluded error rows, and only three repeat clusters. No Prompt SJF control was run. The figure does not establish general OOD robustness or a learned-policy advantage over a non-learned prompt-length control.

## Reproducibility

Run:

```bash
python -B scripts/report_figures/publication_v3/figure_06.py
```

The generator writes `fig6.svg`, `fig6.pdf`, and a 300 dpi `fig6.png` on a fixed 181.9 × 78.0 mm canvas. SVG text remains editable.
