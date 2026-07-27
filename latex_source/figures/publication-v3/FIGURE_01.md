# Figure 1 — Runtime path and decision seam

## Intended message

One runtime metadata seam connects the learned predictor to the serving path. The decision bundle returns to Gateway validation, and the Gateway stamps `vllm_xargs` before forwarding to vLLM. Unreliable or invalid decisions carry `prediction_reliable=0` and omit `workflow_estimated_tokens`. Reliability remains an explicit placeholder boundary.

## Panel structure

- **(a) Runtime request path:** `Client → Gateway → vLLM`. The Gateway validates the returned decision bundle, stamps metadata, and forwards it through the labeled `vllm_xargs` seam. The orange state shows the exact fallback contract.
- **(b) Decision seam:** `prompt + tool schema → BERT prediction → reliability decision`. Only the reliable branch applies the empirical rank-to-length mapping and returns `estimated_tokens`; the fallback branch returns no estimate. Both branches form the decision bundle that returns to panel (a).

## Code lineage

- `scheduler_benchmark/predictor.py`: renders prompt plus tool schema for the BERT sequence classifier and emits a sigmoid rank score. The current BERT predictor reports placeholder `confidence = 0.9` and `ood = false`.
- `scheduler_benchmark/rank_quantiles.py`: maps the rank score through the empirical training-length distribution; this is an approximation, not a calibrated interval.
- `scheduler_benchmark/decision_service.py`: emits `estimated_tokens` only when `prediction_reliable` is true.
- `scheduler_benchmark/gateway_transport.py`: validates the decision bundle and preserves the no-estimate fallback contract.
- `/Users/alex/develop/VeloxMesh/internal/ltr/decision.go`: validates the external decision verdict, stamps `prediction_reliable`, and stamps `workflow_estimated_tokens` only for a reliable decision.

## Files and editability

- `fig1.drawio`: native uncompressed draw.io XML source retained for editing.
- `fig1.drawio.svg`: grouped SVG with the synchronized native draw.io XML embedded in the root `content` attribute.
- `fig1.svg`: grouped, text-editable publication SVG.
- `fig1.pdf`: one-page vector export.
- `fig1.png`: 2149 × 931 px raster export with 300 × 300 dpi metadata.

## Export note

The Homebrew `drawio` wrapper initially pointed to a missing `/Applications/draw.io.app`. After the registered app became present, Electron CLI `--version`, `--help`, and export calls still failed to complete within 30 seconds; the hanging processes were terminated. The native `.drawio` source was therefore retained, the grouped SVG was authored directly, and PDF/PNG were rendered with installed `rsvg-convert`; `sips` set explicit 300 dpi PNG metadata. No v2 figure or presentation file was modified.

## Mechanical checks

- Native draw.io XML and both SVG files parse with `xmllint`.
- The embedded `mxfile` content matches the retained native source after XML attribute whitespace normalization; visible node geometry and labels are synchronized with that source.
- All 11 draw.io edges use expanded `mxCell` elements with `mxGeometry relative="1"`.
- The publication canvas is 181.9 mm wide; PDF page size is 515.622 × 223.370 pt.
- PNG dimensions and metadata are 2149 × 931 px at 300 dpi.
- Raster inspection was used to remove label overflow and label/edge collisions.

Independent review remains separate from implementation.
