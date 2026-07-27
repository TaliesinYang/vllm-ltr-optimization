# Figure 7 — Wrong-hint ordering and simulated workload mixture

## Purpose

- Check whether the predefined tool-traffic fallback reduces dependence on wrong length hints.
- Separate live ordering evidence from simulated p99 behavior.

## Data

- Panel (a): `scripts/report_figures/data/gateway_policy_probe/results_clite/gated_run_ungated.jsonl` and `gated_run_gated.jsonl`.
- Panel (a) audit metadata: `scripts/report_figures/data/gateway_policy_probe/results_clite/gated_verdict.json`; its `verdict_superseded_heuristic` block is not used.
- Panel (b): `scripts/report_figures/data/gateway_policy_probe/results_hybrid10/hybrid_matrix.csv`.
- Live probe: opt-125m, 6 chat and 6 tool requests per mode; Kendall tau-b uses prediction versus first-token order.
- Simulation: 5 tool-traffic shares, 10 seeds, and 4 QPS settings per policy and share.

## Why this experiment

- Learned scheduling can amplify a wrong length hint by serving requests in the wrong order.
- The live probe checks the ordering mechanism; the simulation checks tail behavior across controlled workload mixtures.

## How to read

- Panel (a): dot = recomputed Kendall tau-b; whisker = request-bootstrap 95% interval from 2,000 resamples. Tau is ordering correlation, not latency.
- Panel (b): line = median per-run p99 wait ratio versus FCFS; band = 95% seed-clustered bootstrap interval. Lower is better; 1.0 equals FCFS.
- Horizontal axis in panel (b) is tool traffic share, where tool requests carry wrong hints in this simulation.

## Result

- Chat prediction/order correlation is 1.0 in both modes.
- Tool prediction/order correlation changes from 1.0 ungated to a 0.0667 point estimate gated; the gated interval is wide, `[-0.7778, 1.0000]`.
- The gated policy bundle has a lower simulated median per-run p99 ratio than Pure LTR at every tested share.
- The gated bundle remains above FCFS through 75% tool traffic: ratios 1.211, 1.410, 1.331, and 1.159 at 0%, 25%, 50%, and 75%.

## Limitation

- Panel (a) is a small live ordering probe, not a full-scale latency experiment; n=6/class cannot establish categorical decoupling.
- Panel (b) is simulation, not qwen3.5-9b live serving.
- `gated_hybrid` also changes aging and tail-safe policy. This is a gated policy-bundle comparison, not an isolated gate ablation.
- The fallback is predefined by workload class (`tool -> fallback`), not an evaluated online unreliability detector.

## Reproducibility

- Generator: `scripts/report_figures/publication_v2/figures_07_08.py::build_fig7`.
- Deterministic bootstrap and jitter seed: 1234; bootstrap resamples: 2,000.
- Regenerate: `python scripts/report_figures/publication_v2/figures_07_08.py`.
- Outputs: `latex_source/figures/publication-v2/fig7.pdf` and `fig7.png` (300 dpi).

## Tomorrow's one-line explanation

- The small live probe suggests the predefined fallback weakens wrong-hint ordering, while simulation favors the full gated policy bundle; neither is an isolated live gate-ablation proof.
