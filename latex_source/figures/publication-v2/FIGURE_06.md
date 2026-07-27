# Figure 6 — One BFCL-only OOD workload

Internal evidence note only; not report prose.

## Purpose

Report behavior on one BFCL-only OOD workload using a paired common-complete sample, while keeping invalid-run provenance and exclusions visible.

## Data

- `scripts/report_figures/data/rental-20260719T231309Z/matrix-ood/*.runs/*.samples.csv`
- `scripts/report_figures/data/rental-20260719T231309Z/matrix-ood/{StockFCFSShim,PureLTRScheduler,GatedHybridScheduler,TailSafeScheduler}.json`
- All four source summaries have `valid=false`.
- The attempted matrix has 3 repeats × 120 requests × 4 policies. Seven rows report `stream response omitted completion_tokens usage`, all for `bfcl:irrelevance_130:0000` across different policy/repeat cells.

## Why this experiment

The experiment checks whether the observed scheduling behavior persists on a distribution-shifted workload without presenting one workload as general OOD robustness.

## How to read

Only `(repeat, request_id)` keys completed by all four policies are retained: 357 observations per policy, or 119 request IDs in each of three repeats. Bars are common-subset pooled mean and pooled p99 TTLT. Markers are repeat-level points. Whiskers are deterministic paired hierarchical 95% intervals from 2000 resamples with seed 1234.

## Result

On this common-complete subset, PureLTR, gated hybrid, and tail safe improve mean TTLT by 19.0–19.6% and pooled p99 by 26.4–26.9% versus the FCFS shim. These are common-subset estimates. The 17.4–18.6% mean and 8.2%/17.7% p99 values in the older per-policy-complete analysis use a different estimand and must not caption this common-complete plot.

## Limitation

This is one BFCL-only workload, all source summaries are `valid=false`, seven error rows were excluded, and only three repeat clusters are available. It does not establish general OOD robustness. Similar policy values here also do not demonstrate a gate-protection effect; that mechanism belongs to Fig.7.

**Integration warning:** Report captions and prose must use the common-complete estimand: n=357 per policy, mean improvement 19.0–19.6%, and pooled-p99 improvement 26.4–26.9%. Do not pair this figure with the older per-policy-complete n=358/359/358/358 or 17.4–18.6% / 8.2% / 17.7% values.

## Reproducibility

Run `python scripts/report_figures/publication_v2/figures_04_06.py`. The renderer writes `fig6.pdf` and a 300 dpi `fig6.png` under this directory.

## Tomorrow's one-line explanation

The scheduler advantage persists on this one BFCL-only common-complete subset, but invalid provenance, seven exclusions, and single-workload scope prevent a broad OOD claim.
