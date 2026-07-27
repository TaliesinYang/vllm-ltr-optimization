# Figure 2 — Frozen evidence and artifact lineage

## Intended message

The frozen Tier-2 evidence is split before analysis, validation selects the predictor run, and the empirical percentile lookup is computed from 6,000 `output_length` labels. Held-out predictor performance is reported only as evaluation and is not an input to the lookup.

## Panel structure

- **(a) Frozen evidence:** short source revisions plus raw and analyzed Train/Validation/Test counts.
- **(b) Selection and lineage:** the 10-run matrix selects `bert-prompt_schema-tier2-seed17` by validation Kendall τ. That run yields the checkpoint. Separately, 6,000 output-length labels yield the empirical p10–p99 lookup. The held-out τ tag has no outgoing edge.

## Frozen evidence

| Evidence | Train | Validation | Test |
|---|---:|---:|---:|
| Raw split | 4,000 | 1,000 | 1,000 |
| Analyzed | 3,997 | 998 | 999 |

Source pins shown in the figure:

- ToolACE revision `6bda777`
- Qwen tokenizer revision `c202236`
- BERT backbone revision `86b5e09`

## Code and artifact lineage

- `configs/training_sources.json`: full repository and revision declarations behind the short source pins.
- `scripts/report_figures/data/offline/tier2-sample-manifest.json`: frozen 6,000-row sample, seed 42, and raw 4,000/1,000/1,000 split.
- `scripts/report_figures/data/offline/tier2-matrix-summary.json`: 10/10 completed runs, analyzed counts, and the validation-selected run. The selected run has validation τ = 0.6571718653 and held-out test τ = 0.6423292768; the figure rounds both to three decimals.
- `scheduler_benchmark/rank_quantiles.py`: loads `output_length` from the label records and builds the nearest-rank empirical p10–p99 lookup. It does not build the lookup from held-out predictor scores.
- `scripts/build_rank_quantiles.py`: requires the labels path and checkpoint path and emits the replay sidecar plus quantile manifest.

## Files and editability

- `fig2.drawio`: native uncompressed draw.io XML source.
- `fig2.drawio.svg`: grouped SVG with the synchronized native draw.io XML embedded in the root `content` attribute.
- `fig2.svg`: grouped, text-editable publication SVG.
- `fig2.pdf`: one-page vector export.
- `fig2.png`: 300 dpi raster export.

## Export note

The draw.io Electron CLI did not complete within the bounded export attempt. The native `.drawio` source was retained, the synchronized grouped SVG was authored directly, and installed `rsvg-convert` produced PDF/PNG; `sips` set explicit 300 dpi PNG metadata. No v2 figure or presentation file was modified.

## Mechanical checks

- Native draw.io XML and both SVG files parse with `xmllint`.
- The embedded `mxfile` content matches the retained native source after XML attribute whitespace normalization; visible node geometry and labels are synchronized with that source.
- All 3 lineage edges use expanded `mxCell` elements with `mxGeometry relative="1"`.
- No edge originates from the held-out evaluation tag, and no selected-run edge enters the empirical lookup.
- The publication canvas is 181.9 mm wide; PDF page size is 515.622 × 223.370 pt.
- PNG dimensions and metadata are 2149 × 931 px at 300 dpi.
- Raster inspection was used to remove label overflow and label/edge collisions.

Independent review remains separate from implementation.
