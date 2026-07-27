# Figure 8 — Observed path difference in one ordered replay

## Purpose

- Quantify the observed request-path difference between direct vLLM FCFS and the gateway path.
- Keep this cost measurement separate from scheduling-policy comparisons.

## Data

- `scripts/report_figures/data/rental-20260719T231309Z/gateway-overhead.json`.
- Same workload and arrival offsets are replayed once per path with FCFS in both arms.
- Pairing key: `request_id`; TTFT uses all 150 pairs.
- TTLT uses 118 pairs with matched `output_tokens`; 32 output-mismatched pairs are excluded.
- Execution order is fixed by `scheduler_benchmark/gateway_overhead.py::run_overhead_pair`: direct first, gateway second.

## Why this experiment

- A scheduling improvement inside the serving system may not produce an end-to-end gain when the decision path adds latency.
- Paired request deltas expose the distribution hidden by arm-level averages.

## How to read

- Each point is one paired `gateway - direct` delta; positive values mean the gateway path was slower for that request.
- Violin width shows request-level density; black line marks the median.
- White dot marks the observed mean difference; whisker is a 2,000-resample request-bootstrap 95% interval.

## Result

- TTFT, n=150: observed mean difference `+737 ms`; request-bootstrap 95% interval `[+708, +764] ms`.
- Matched-output TTLT, n=118: observed mean difference `+840 ms`; request-bootstrap 95% interval `[+676, +1,058] ms`.
- The observed TTFT difference is larger than the roughly 500 ms scheduling benefit documented for the current prototype.

## Limitation

- This is one ordered replay: direct first, gateway second. It is not counterbalanced across run order.
- Request bootstrap conditions on this replay and does not include run-level or run-order uncertainty.
- The displayed values are observed mean differences, not a causal gateway-overhead constant.
- Matched-output TTLT excludes 32/150 pairs. The all-request marginal p99 difference (`+9,482 ms`) is output-length confounded and is not plotted as overhead.

## Reproducibility

- Generator: `scripts/report_figures/publication_v2/figures_07_08.py::build_fig8`.
- Pair construction: `scripts/report_figures/fig8_overhead.py::load_overhead_pairs`.
- Deterministic bootstrap and jitter seed: 1234; bootstrap resamples: 2,000.
- Regenerate: `python scripts/report_figures/publication_v2/figures_07_08.py`.
- Outputs: `latex_source/figures/publication-v2/fig8.pdf` and `fig8.png` (300 dpi).

## Tomorrow's one-line explanation

- In one direct-first ordered replay, the gateway path was 737 ms slower in mean TTFT and 840 ms slower in matched-output mean TTLT; this is an observed paired difference without run-level uncertainty.
