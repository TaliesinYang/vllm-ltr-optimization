# Figure 5 — Paired live TTLT reduction on the mixed workload

## Purpose

Show whether short-job ordering lowers live time-to-last-token (TTLT), while separating that result from any unsupported claim that learned prediction beats a prompt-length control.

## Data

- `scripts/report_figures/data/rental-20260719T231309Z/matrix/*.runs/*.samples.csv`
- Matching per-run JSON files provide repeat identifiers.
- Renderer: `scripts/report_figures/publication_v3/figure_05.py`.

The balanced evidence contains the same 150 request IDs in each of three repeats for every policy, with zero error rows. These are repeated observations, not 450 independent requests per policy.

## How to read

Panel (a) shows empirical TTLT CCDFs for Stock FCFS, Gated hybrid as one representative learned policy, and Prompt SJF as the non-learned control. Gated hybrid is selected to keep the distribution panel readable; it is not labeled or interpreted as the best learned policy.

Panels (b) and (c) show pooled mean and pooled p99 TTLT reduction relative to the paired Stock FCFS baseline. Positive values mean lower TTLT. Filled circles are pooled paired estimates. Open circle/square/diamond markers expose the three matched repeat estimates. Whiskers are deterministic paired hierarchical 95% intervals from 2,000 resamples with seed 1234: repeat clusters and request IDs are resampled, and identical draws are applied to each policy and Stock FCFS. P99 is computed on each pooled bootstrap sample.

## Result

All four learned policies have lower pooled mean TTLT than Stock FCFS by 14.8–15.1%; their pooled p99 reductions range from 8.5% to 18.2%. Prompt SJF has 15.3% lower pooled mean TTLT and 19.8% lower pooled p99 TTLT. The measured comparison supports short-job ordering, while the learned policies do not beat the non-learned Prompt SJF control on either pooled point estimate.

## Limitation

There are only three repeat clusters. The paired hierarchical intervals preserve the repeat/request-ID structure but do not turn repeated rows into independent experimental replicates. The experiment tests one live saturated mixed workload, and Gated hybrid is only a representative learned-policy CCDF in panel (a).

## Reproducibility

Run:

```bash
python -B scripts/report_figures/publication_v3/figure_05.py
```

The generator writes `fig5.svg`, `fig5.pdf`, and a 300 dpi `fig5.png` on a fixed 181.9 × 92.0 mm canvas. SVG text remains editable.
