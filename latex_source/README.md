# latex_source

The LaTeX source of the submitted report, and the checks that keep its numbers
honest.

Build with `tectonic -X compile 00.tc_main.tex`. The compiled result is
`00.tc_main.pdf`.

| Path | What it is |
|---|---|
| `00.tc_main.tex` | driver; `\input`s the numbered sections in order |
| `01…09.*.tex` | one file per report section |
| `reference.bib` | bibliography, audited against Crossref and OpenAlex |
| `figs/` | the figure PDFs the report includes |
| `scripts/` | the checks described below |
| `FIGURE-SPEC.md` | the figure standard the plates are built against |
| `EVIDENCE-MAP.md` | generated; every printed number and the artifact it came from |

The figure *generators* are not here — they live in
`scripts/report_figures/paper_v1/` at the repository root, next to the run data
they read. Every number in every plate is read from a committed artifact at
build time rather than typed into the script.

## Checks

- `scripts/build_evidence_map.py` re-derives each quantitative claim from the
  artifact it names and exits non-zero if the artifact disagrees with the number
  printed in the report. `EVIDENCE-MAP.md` is its output, not a hand-written
  document.
- `scripts/figure_contract.py` reports each figure's native size, its placed
  size, and the smallest type it contains, and fails if a plate would render
  text below the legibility floor.
- `scripts/overlap_check.py` finds glyphs printed on top of other glyphs in the
  compiled PDF. It reads the text layer only, so it cannot see a label sitting
  on a rule — the figures still need an eye on them.

`_superseded-planning-draft/` holds the earlier outline-stage draft
(`main.tex` and its `sections/`), kept for history. It is not the submitted
report and does not compile against the current figures.
