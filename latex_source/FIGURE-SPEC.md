# Figure specification — locked

One document, written once, so figure work stops being a sequence of patches
that each break something else. Every rule below either comes from a source
that outranks preference (the assignment, the advisor's stated standard) or
from a measurement of the paper this set is benchmarked against. Where a rule
is a judgement call, it says so and gives the reason.

Anything not listed here is not a constraint. Anything listed here is not
negotiable at implementation time — if a rule turns out to be wrong, change
this file first, then the figures.

## 0. Sources of authority

1. **The assignment** (`_inbox/FinalReport_Summer.docx`): matplotlib only for
   graphs; Evaluation carries at least six of them; schematic diagrams must be
   architecture-like with arrows; figure text must be legible at print size.
2. **The advisor's standard** (Ryoo, recorded 2026-06-08, encoded in the
   `figure-craft` skill): a reviewer judges from the figures before the prose,
   so *easy-to-read-fast is criterion #1*. "If it takes too much time I'm not
   going to read it." The most common failure named is **too much text in the
   figure**. Also: do not ship one giant figure — split into panels, and
   prefer **wide-and-short** over tall.
3. **EXION** (HPCA'25, arXiv 2501.05680), the craft benchmark, measured rather
   than remembered:
   - 19 figures over 14 pages, and **every page carrying figures still carries
     762–984 words of text**. Their figures are roughly 1.5–2 in tall.
   - Palette sampled from the published artwork: two sequential family ramps
     (blue `#5ABAD1 #3984B6 #264992 #161F63`; magenta `#F4AEA3 #E8638B
     #A73B8F #61208D #3C1357`), one light tone for the outside baseline
     (`#B7DFCB`), greys for structure (`#E8E8E8 #DFDFDF #D1D1D1 #7F7F7F`).
   - Conventions: framed header strips, boxed legend bands, framed result
     callouts carrying factors, per-element numeric labels, compact sans.
   - **EXION has no warning colour.** Its warm tones are a data family, not an
     alert. Our use of vermillion is our own convention (see §4).

## 1. Size — the rule this set kept breaking

**Hard caps.** Double-column plate: **≤ 2.8 in tall**. Single-column:
**≤ 2.6 in**. The architecture schematics may reach 3.2 in because they carry
2D nesting rather than axes.

This is the rule that was missing and it is why the set ballooned to 4.3–5.3 in
per figure. Each agent was told never to shrink type, obeyed, and grew the
canvas instead; six locally-correct decisions produced a set that eats half a
page per figure. Both constraints hold together, and when they collide the
resolution order is:

1. cut content (a row, a redundant annotation, a decorative band),
2. reflow (legend into the header line, footnote into the caption, panels
   side-by-side instead of stacked),
3. tighten padding and inter-panel gaps,
4. only then reconsider the cap — by editing this file, with a reason.

Never resolve it by reducing type size.

**Test:** a page holding one double-column figure must still hold ≥ 600 words
of body text. That is the EXION density, relaxed by about a third.

## 2. Type

- One family, **DejaVu Sans**, for text, bold, italic and math. Not Helvetica:
  it ships no bold, oblique or math faces here, so a Helvetica figure silently
  renders in four families at once.
- Three sizes only: **10 pt** axis labels, **9 pt** annotations, header strips
  and tick labels, **8 pt** dense in-panel micro text. Never 8.5, never 9.5.
- Math subscripts render at 0.7×, so `$\tau_b$` at 8 pt puts its subscript at
  5.6 pt. Where a symbol appears in 8 pt text, write it plain (`tau_b`) or
  promote the text to 10 pt. Subscripts are permitted only at 10 pt.

## 3. Text budget

- Panel headers are **labels of at most four words, with no digits**:
  "(a) Paired effects", "(b) Absolute TTLT". A header carrying a finding is a
  sentence in a strip; move the finding to a callout.
- Findings live in **framed callouts**: `boxstyle='square,pad=0.25'`,
  face `#E8E8E8` or white, edge `#7F7F7F`, linewidth 0.6.
- Captions: **at most two sentences**. The caption states what the figure
  shows and the one qualifier that stops a reader over-reading it. Panel-by-
  panel enumeration belongs in the header strips, not the caption.
- No sentence set inside a box as a banner. That reads as filler.

## 4. Colour

- Data: the EXION family ramp, light→dark by **how much of the ordering
  decision the arm's ranker owns** — PolicyFCFS, PromptLengthSJF, PureLTR,
  GatedRuleC. Ramp position is meaning, not order of appearance.
- The engine we did not build (stock scheduler, external baseline): the mint
  `#B7DFCB`.
- Structure only (frames, header strips, gridlines, footnotes): the greys.
- **Vermillion `#D55E00` is reserved for degraded semantics** — fail-open,
  overstatement, a non-independence caveat. This is a deliberate departure
  from EXION, which has no alert colour, and it is kept because in this paper
  failure modes are a result rather than a nuisance: the gate's overstatement
  band and the fail-open path are things the reader must not mistake for data.
  It appears on at most one element per figure and never as a series.
- Nothing outside those lists. Anti-aliasing intermediates do not count.

## 5. Statistical integrity

These override any craft rule they collide with.

- An interval stays an interval. Never redraw a CI as a bar.
- Print the point estimate; the whisker already carries the interval. Do not
  print both endpoints per row.
- Never label raw samples, and never label every step of an ECDF.
- Do not claim separation from overlapping marginal intervals; if the paper
  claims a difference, the figure must show the paired comparison that
  supports it.
- Every number in the artwork is read from a committed artifact at build time.
  No literals. If it cannot be derived, it is not printed.

## 6. Anti-template (de-AI)

- 2D nesting over a single row of equal boxes.
- Size by importance; the contribution is not the same size as the substrate.
- Real domain detail — actual arm names, actual n, the real decision rule.
  This is the strongest signal a template cannot fake.
- Considered layout, not pixel-even spacing.
- No gradients, shadows, 3D, or decorative icons.

## 7. Build-time enforcement

A figure script must fail loudly rather than emit a violation. Each generator
carries guards that assert its own layout: header word count, no digits in
headers, glyph size floor, slot widths, overlap sweeps. **Do not weaken a
guard to make a build pass.** If a guard's threshold is wrong, change it and
say why in the commit.

Repository-level checks: `scripts/figure_contract.py` (scale and font floor),
`scripts/overlap_check.py` (glyph collisions in the compiled PDF),
`scripts/build_evidence_map.py` (every printed number against its artifact).
