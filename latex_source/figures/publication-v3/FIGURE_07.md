# Figure 7 — Live wrong-hint ordering and simulated policy-bundle behavior

## Purpose

Keep the small live ordering probe separate from the simulated workload-mixture comparison, while preserving the evidence that the gated configuration is a policy bundle rather than an isolated gate intervention.

## Data

- Live probe: `scripts/report_figures/data/gateway_policy_probe/results_clite/gated_run_ungated.jsonl` and `gated_run_gated.jsonl`.
- Simulation: `scripts/report_figures/data/gateway_policy_probe/results_hybrid10/hybrid_matrix.csv`.
- Renderer: `scripts/report_figures/publication_v3/figure_07_08.py`.

## How to read

Panel (a) reports Kendall τ-b between prediction and first-token order for six chat and six tool requests in each mode. Points are recomputed estimates; whiskers are 2,000-resample request-bootstrap 95% intervals. This is an ordering probe, not latency evidence.

Panel (b) reports the simulated median per-run p99 wait ratio relative to FCFS across tool-traffic shares. Lines are policy medians; bands are 95% seed-clustered bootstrap intervals. Each share contains 10 seeds × 4 QPS settings. Lower ratios are better; 1.0 equals FCFS.

## Result

The gated tool-ordering point estimate is τ-b 0.0667 with a wide 95% interval from −0.7778 to 1.0000. In simulation, the gated policy bundle has a lower median p99 wait ratio than Pure LTR at every tested tool-traffic share.

## Limitation

The live n=6/class probe is too small for a categorical conclusion and does not measure latency. The second panel is simulation. The gated bundle also changes aging and tail-safe behavior, so it is not an isolated gate ablation or full-scale live latency proof.

## Reproducibility

Run:

```bash
python -B scripts/report_figures/publication_v3/figure_07_08.py
```

The generator writes editable `fig7.svg`/`fig7.pdf` and a 300 dpi `fig7.png` on a fixed 181.9 × 86.0 mm canvas.
