# Figure version history

Each round of figure work is archived here before the next round overwrites
`figs/*.pdf`, so versions can be compared side by side.

| Version | Date | What changed | Why |
|---|---|---|---|
| v1-initial | 2026-07-26 | first generated set | baseline |
| v2-semantic-palette | 2026-07-27 | accent=proposed / grey=baselines (was a rainbow); withheld strata as hatched empty slots (were full-height bars that read as maximal values) | colour now encodes ownership; removed a misleading encoding |
| v9-figure-craft | 2026-07-27 | both schematics redesigned per the figure-craft skill: 2D nesting (scheduler nested inside the engine, matching the code), a real captured request traced through with measured numbers (147kB schema, S4 conf 0.62, 37.9ms, Rule C floors), queue glyph with trusted/pinned slots, numbered circles removed | Phase 0 code-truth caught the old fig1 drawing a Decision->QueueOrder->vLLM path that does not exist: the verdict rides the forwarded request (decision.go:218) and is read back inside the engine (vllm_scheduler.py:87) |
| v8-10pt | 2026-07-27 | all TikZ figure text raised from 8/7pt to 10pt; request flow drawn heavier than decision flow; boundary title moved outside the box | the schematics had been violating the course 10pt floor since they were drawn, and figure_contract.py never saw them because they are not PDF files |
| v7-orthogonal | 2026-07-27 | fig2 branch rerouted orthogonally from a single split point; warn labels merged into one node clear of the lines | diagonal edges read as hand-drawn next to top-venue schematics, and the two separate orange labels landed on the Fallback box and across the vertical line |
| v6-tikz | 2026-07-27 | architecture and decision-path figures rewritten as native TikZ compiled inside the document (styles in systems-figures.tikzstyles); matplotlib versions deleted | hand-written patch code was acting as a bad vector editor -- moving one component meant editing five coordinates -- and could never match the document's own type; TikZ inherits it and derives boundaries with `fit` |
| v5-forest | 2026-07-27 | ranking and cold-start redrawn as dot-and-interval (forest) plots; withheld strata moved to a non-numeric status column reading WITHHELD n<100; added scripts/figure_contract.py to check effective font size, placement scale and font embedding | estimates with intervals are not quantities accumulated from zero, so bars were the wrong encoding; a hatched slot on the quantitative axis still invited reading as a value |
| v4-true-scale | 2026-07-27 | wide figures promoted to two-column floats; architecture redrawn at 7.16in | five figures were being scaled to ~49% in LaTeX, collapsing 10pt text to 4.9pt — the actual cause of the "cheap" look, and a violation of the >=10pt rule |
| v3-architecture | 2026-07-27 | arch + decision redrawn: system boundary, numbered request lifecycle, contributed components highlighted, measured facts as annotations rather than boxes, degraded path in warning colour | first two figures read as parts lists rather than as a system |

## Compare two versions

```bash
cd figs/versions
for f in arch decision ranking coldstart gate workload; do
  pdftoppm -r 150 -png -singlefile v1-initial/$f.pdf /tmp/${f}_v1
  pdftoppm -r 150 -png -singlefile ../$f.pdf        /tmp/${f}_now
done   # then open /tmp/*_v1.png and /tmp/*_now.png
```
