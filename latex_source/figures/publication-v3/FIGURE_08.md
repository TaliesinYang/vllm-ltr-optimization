# Figure 8 — Observed path difference in one ordered replay

## Purpose

Show the observed paired request differences between direct vLLM FCFS and the gateway path without presenting one ordered replay as a causal overhead constant.

## Data

- `scripts/report_figures/data/rental-20260719T231309Z/gateway-overhead.json`
- Pair construction: `scripts/report_figures/fig8_overhead.py::load_overhead_pairs`.
- Renderer: `scripts/report_figures/publication_v3/figure_07_08.py`.

TTFT uses all 150 request-ID pairs. TTLT uses 118 pairs with matched output-token counts; 32 output-mismatched pairs are excluded.

## How to read

Each blue point is one paired `gateway − direct` delta, so positive values mean the gateway path was slower for that request. Violin width shows request-level density and the short dark line marks the median. The white point is the observed mean; its black bar is a 2,000-resample request-bootstrap 95% interval. Panel (b) retains the observed 10.4 s TTLT outlier on one continuous axis.

## Result

TTFT has an observed mean difference of +0.737 s with a request-bootstrap 95% interval of [+0.708, +0.764] s. Matched-output TTLT has an observed mean difference of +0.840 s with an interval of [+0.676, +1.058] s.

## Limitation

This is one replay executed direct first and gateway second. The request bootstrap is conditional on that replay and excludes run-order and run-level uncertainty. The values are observed paired differences, not a causal or cross-run gateway-overhead constant.

## Reproducibility

Run:

```bash
python -B scripts/report_figures/publication_v3/figure_07_08.py
```

The generator writes editable `fig8.svg`/`fig8.pdf` and a 300 dpi `fig8.png` on a fixed 181.9 × 80.0 mm canvas.
