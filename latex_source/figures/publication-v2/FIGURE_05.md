# Figure 5 — Main mixed-workload serving result

Internal evidence note only; not report prose.

## Purpose

Show the measured TTLT distribution and pooled mean/p99 behavior for stock FCFS, the FCFS shim, four learned scheduling policies, and non-learned PromptLengthSJF.

## Data

- `scripts/report_figures/data/rental-20260719T231309Z/matrix/*.runs/*.samples.csv`
- Matching per-run JSON files provide repeat numbers.
- Seven policies each contain the same 150 unique request IDs in three repeats: 450 rows per policy and zero error rows.

## Why this experiment

This is the main live saturated MIXED comparison. It tests whether short-job scheduling improves end-to-end TTLT and whether any gain is specific to learned prediction.

## How to read

Panel (a) overlays empirical TTLT CCDFs. Its low-alpha bands use the same hierarchical repeat/request resampling as panel (b). Panel (b) bars are pooled mean and pooled p99 TTLT; the circle, square, and diamond markers expose the three repeat-level estimates. Whiskers are deterministic 95% intervals from 2000 paired hierarchical resamples (seed 1234): repeat clusters and request IDs are resampled, and the same draws are applied to every policy. P99 is always computed on the pooled bootstrap sample, never averaged across repeat p99 values.

## Result

Relative to real stock FCFS, the four learned policies improve mean TTLT by 14.8–15.1% and pooled p99 by 8.5–18.2%. PromptLengthSJF improves mean by 15.3% and pooled p99 by 19.8%. PromptLengthSJF is explicitly non-learned.

## Limitation

The experiment has only three repeat clusters. PromptLengthSJF matches or exceeds the learned policies here, so the gain supports short-job prioritization generally and cannot be attributed to learned prediction alone. Constant gateway overhead is shared across these policy arms.

**Integration warning:** Report captions and prose must call the whiskers paired hierarchical 95% intervals over repeat/request clusters. Do not revert to pooled request-bootstrap wording or imply 450 independent requests per policy.

## Reproducibility

Run `python scripts/report_figures/publication_v2/figures_04_06.py`. The renderer writes `fig5.pdf` and a 300 dpi `fig5.png` under this directory.

## Tomorrow's one-line explanation

Scheduling helps the mixed workload, but non-learned prompt-length SJF reaches the best pooled p99 improvement, so learned prediction has not won this comparison.
