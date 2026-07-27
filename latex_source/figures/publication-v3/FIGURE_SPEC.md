# Publication-v3 Figure Spec

## Target

- IEEE double-column canvas: 181.9 mm wide; prefer 65–100 mm height.
- Vector-first: editable SVG/PDF. Fig.1–2 retain Draw.io-embedded SVG exports.
- Raster preview: 300 dpi PNG.
- Three-second test: one dominant message per figure; figure must remain understandable without its caption.

## Type

- Font family: Arial / Helvetica, regular and bold only.
- Figure/panel headline: 10 pt bold.
- Body, axes, legend, annotations: 8 pt.
- Micro text: 7 pt, only for sample counts or scope notes.
- Text color: `#1A1A1A`.

## Palette

| Role | Color | Meaning |
|---|---|---|
| Learned predictor / decision control | `#0072B2` | model score, learned ordering, control request |
| Fallback / non-learned control | `#E69F00` | fallback, prompt-length heuristic, control branch |
| Neutral system / baseline | `#4A4A4A` | request path, FCFS, infrastructure, default text lines |
| Limitation / excluded / invalid | `#D55E00` | warnings only; never decorative |
| Background | `#FFFFFF`, `#F2F2F2` | white canvas and neutral grouping |

No gradients, shadows, 3D, decorative icons, or color without semantic meaning.

## Geometry

- Panel labels: bold lowercase `(a)`, `(b)`, `(c)` at the upper-left, outside plot content.
- Connector stroke: 0.75–1.0 pt; minimum visible connector length 10 mm.
- Arrow labels never sit on lines or box boundaries.
- Boxes sized to content; avoid equal-sized generic pipeline cards.
- Schematics use 2D nesting and explicit system boundaries rather than one flat row.
- Data plots use shared legends and aligned baselines; do not repeat legends across panels.

## Statistical language

- Prefer `lower` or a downward arrow for latency improvement; never use a bare plus sign.
- Show paired change against the named baseline when the interval is paired.
- Distinguish validation selection, held-out evaluation, simulation, and live measurements visually.
- `Observed`, `suggestive`, and `supported` are not interchangeable with causal claims.

## Locked figure messages

- Fig.1: one real metadata seam connects predictor decisions to scheduling; reliability remains a placeholder boundary.
- Fig.2: validation selects the checkpoint; the empirical rank lookup comes from 6,000 output-length labels, not held-out predictor scores.
- Fig.3: shortest-first improves mean TTFT near saturation but creates a tail TPOT trade-off at the highest tested load.
- Fig.4: schema-aware text models rank well; prompt+schema and full-context seed ranges overlap, and the learning curve is a separate single-seed validation ablation.
- Fig.5: short-job ordering lowers live TTLT, but the learned predictor does not beat the prompt-length control.
- Fig.6–8: messages remain evidence-limited; final wording follows their independent reviewer report.

## Review gate

- Implementer may not approve its own figure.
- Every figure requires a different agent to issue explicit PASS.
- Any FAIL returns to implementation; completion requires all eight PASS plus cross-figure consistency PASS.
