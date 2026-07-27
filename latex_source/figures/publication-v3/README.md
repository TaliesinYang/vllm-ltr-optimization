# Publication-v3 figure set

This directory is the isolated IEEE full-width figure set. Figures 1–2 are static Draw.io/SVG artifacts; Figures 3–8 are generated from committed evidence. Publication-v2 and presentation assets are outside this workflow.

## Figure index

| Figure | Main message | Source |
|---|---|---|
| 1 | One validated `vllm_xargs` metadata seam connects the predictor decision bundle to the Gateway; unreliable decisions omit `workflow_estimated_tokens`. | Static Draw.io/SVG; see `FIGURE_01.md` |
| 2 | Validation selects the checkpoint, while the empirical p10–p99 lookup comes from 6,000 output-length labels rather than held-out scores. | Static Draw.io/SVG; see `FIGURE_02.md` |
| 3 | Legacy shortest-first lowers mean TTFT near saturation but raises tail TPOT at the highest tested load. | `figure_03.py` |
| 4 | Held-out predictor ranking and the single-seed validation data-scale curve answer separate questions. | `figure_04.py` |
| 5 | Short-job ordering lowers live mixed-workload TTLT, but learned policies do not beat Prompt SJF on the reported point estimates. | `figure_05.py` |
| 6 | One common-complete BFCL subset shows paired reductions, without promoting invalid-provenance summaries to general OOD evidence. | `figure_06.py` |
| 7 | The small live ordering probe is separate from the simulation, where the gated configuration is a policy bundle. | `figure_07_08.py` |
| 8 | One ordered replay reports observed gateway-minus-direct differences, not a causal overhead constant. | `figure_07_08.py` |

## Generate and validate

From the repository root:

```bash
python3 scripts/report_figures/publication_v3/render_all.py
```

The command first requires and XML-parses these static sources:

- `fig1.drawio`, `fig1.drawio.svg`, `fig1.svg`
- `fig2.drawio`, `fig2.drawio.svg`, `fig2.svg`

It then calls the generators in numeric order: `figure_03`, `figure_04`, `figure_05`, `figure_06`, then `figure_07_08` for Figures 7 and 8. Generated SVG/PDF/PNG files are written to this directory.

For an isolated output directory while retaining the committed static sources:

```bash
python3 scripts/report_figures/publication_v3/render_all.py \
  --output-dir /tmp/publication-v3
```

## Draw.io headless fallback

The canonical editable sources for Figures 1–2 are the native `.drawio` files. Their `.drawio.svg` companions contain synchronized embedded `mxfile` XML, and the plain SVG files retain editable groups and text.

If the Draw.io Electron CLI is unavailable or fails to complete in a bounded headless attempt, retain the native `.drawio` source and synchronized embedded XML. Use the grouped SVG as the visible source, render PDF/PNG with an installed SVG renderer such as `rsvg-convert`, and set/check 300 dpi PNG metadata. Record the fallback in the corresponding `FIGURE_0N.md`; do not imply a successful Draw.io CLI export.

## Review gate

`render_all.py` provides mechanical generation and XML validation only. It is not a publication approval signal. A reviewer separate from the implementer must inspect code/data lineage, labels, edge direction, clipping/collisions, canvas size, raster density, and consistency between native Draw.io, embedded XML, and visible SVG before recording any review result.
