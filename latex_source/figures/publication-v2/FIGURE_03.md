# Figure 3 — Legacy baseline reproduction

## Purpose

Explain why shortest-first scheduling is promising for queueing latency and why tail safety must be evaluated separately.

## Data

- Measured baseline table: `scripts/report_figures/data/offline/baseline-2026-06-22-RESULTS-summary.txt`.
- Integrity record: `scripts/report_figures/data/offline/SHA256SUMS.txt`.
- Parser and original geometry: `scripts/plot_final_report_figures.py` (`parse_baseline_summary`, `draw_fig3`).
- Legacy stack scope: `docs/REPRODUCE-BASELINE.md`, `docs/ENV-NOTES.md`, and `docs/SUMMARY.md` — Meta-Llama-3-8B-Instruct, LMSYS trace, and vLLM 0.4.1 fork.

## Why this experiment

The sweep reproduces the baseline scheduling trade-off before predictor and gate changes are evaluated: prioritizing predicted-short work can reduce queue delay while worsening per-token tail cost.

## How to read

Panel (a), “Mean TTFT,” plots mean time to first token; panel (b) plots p99 time per output token (TPOT). Both axes are logarithmic and lower is better. FCFS is neutral gray; legacy LTR is blue. The footer fixes the scope to Meta-Llama-3-8B-Instruct, LMSYS, the vLLM 0.4.1 fork, and one sweep.

## Result

At 16 queries/s, legacy LTR mean TTFT is 6,043.7 ms versus FCFS 17,274.2 ms, or 65.0% lower. At 64 queries/s, legacy LTR p99 TPOT is 1,400.29 ms versus FCFS 171.19 ms, or 8.18× p99 TPOT cost (LTR / FCFS).

## Limitation

This is one legacy Meta-Llama-3-8B-Instruct/LMSYS/vLLM 0.4.1-fork sweep with no repeated runs, so no repeat-level confidence interval is available. The figure does not establish that learned prediction beats PromptLengthSJF. The supported reduction wording is 65.0% lower mean TTFT; ratio-based “lower” wording would be ambiguous.

## Contribution boundary

Dazhi reproduced and analyzed the scheduling baseline. FCFS, LTR, and the underlying serving model are prior systems, not inventions owned by Dazhi.

## Reproducibility

Run `python -B scripts/report_figures/publication_v2/figures_01_03.py`. `build_fig3()` parses the committed table directly, plots every measured row without smoothing, and adds no synthetic uncertainty. Its PDF uses a fixed 3.5 in (252 pt) media box so 10 pt source text is not downscaled during IEEE single-column placement.

## Tomorrow's one-line explanation

Shortest-first cuts mean queueing delay under load, but this single sweep also shows an 8.18× p99 per-token tail cost, motivating explicit tail protection.
