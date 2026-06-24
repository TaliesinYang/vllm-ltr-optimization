# Figure Consistency Check — Capstone Evaluation (04-evaluation)

Audit of the 4 result figures (`baseline`, `tau`, `gap`, `ablation`) against the Ryoo standard + figure-craft de-AI checklist + cross-figure consistency checklist.

## 1. Verdict

All 4 figures **PASS** (no blocking fabrication, no causal overclaim, zero hard overlap) and are individually publication-ready. **But the set is NOT fully consistent**: one true cross-figure break (C3 semantic colour map — `ablation` reassigns vermillion/blue) plus several **warn**-level set deviations (two near-black greys C2, physical-scale spread C5, grayscale/CVD L\* clustering C6). `ablation` and `baseline` were fixed this pass; residual items below are minor/legibility-only.

## 2. Per-figure audit

| figure | pass | residual notes |
|--------|:----:|----------------|
| **baseline** | yes | C2: secondary callout text + arrows were `#333333` vs pure-black primary text — fixed to one dark tone this pass. A2: base 12pt / 11.5pt annot on 11×4.4in 2-panel → ticks & 10^x exponents read small at half-slide width; bump ticks ~13–14pt, axis labels ~14pt. B5: FCFS and LTR(ours) share `lw=2.2/ms=6.5` — emphasize LTR by weight (`lw~2.8`), not hue alone. A3 (minor): faint log-y gridlines add mild clutter; drop or keep only major-decade. B4 (minor, verify at render): right-panel red leader (9,900)→(64,1400) and left curved arrow sweep near the rising LTR curve — confirm no clip/poke-through. |
| **tau** | yes | A1 verified: 0.194 / 0.559 / 0.596 and 3.1× match ground truth — no fabrication. B4: curved arrow head xy=(1.96,0.614) touches the centered `0.596` label box — nudge endpoint down-left / shorten `rad` to clear it. A4: "3.1x higher" literally implies ~4.1× — change to "3.1x as high" or just "3.1x". A4/A1: headline "ranking >> classification" + arrow span classification(OPT-125M)→PARS(BERT) conflates ranking gain with the OPT→BERT swap; clean ranking-only evidence is listMLE 0.559 vs classification 0.194 (same OPT-125M) — tighten wording/arrow. A3: curved arrow + two stacked vermillion lines restate the bar heights — trim to one annotation. |
| **gap** | yes | A2: bottom caption 9pt gray italic (`#444444`) survives full size but risks illegibility shrunk to slide — bump ~10–11pt or shorten. A2: "drop 0.243"/"drop 0.235" slope labels 9.5pt italic — raise ~10–11pt. A3: "+15% relative (0.315 → 0.361)" restates values already at the markers — trim to "+15% on unseen data". A4/A1 (minor, non-blocking): "+15%" rounds true 14.6% up; acceptable since exact endpoints shown and matches GT headline. B5: listMLE and PARS(ours) both `lw=2.5` — thicken PARS (`lw~3.0`) to mark the winner. B1/C6: amber `#E8A33D` vs vermillion `#C2452D` both warm, confusable under protan/deutan; spatial separation mitigates — consider differing marker shapes. |
| **ablation** | yes | C3: cross-figure colour inversion (see consistency table) — reconciled/annotated this pass. B1: no key for the waterfall semantics (blue=level endpoint, grey=no-help step, vermillion=driver step) — add a 3-entry key. A2/A4: grey deltas "-0.012"/"-0.007" use `#7A7A7A` on white — low contrast shrunk to slide; darken the numbers. B4: callout arrowhead xy=(2,0.368) and "+0.065" label (y≈0.368, x=2) both terminate at the BERT bar top — confirm no graze, nudge label up ~0.002 if needed. A4: "is the sole driver" is strong causal wording; "on this setup" hedge keeps it acceptable — consider "the only component that helps here" to stay associational. |

## 3. Cross-figure consistency

| dimension | status | note |
|-----------|:------:|------|
| **C1 font family** | OK | All four set `font.family='DejaVu Sans'`. Identical across baseline/tau/gap/ablation. |
| **C2 one dark text colour** | warn | Primary text (titles, axis/tick/value labels) uniformly pure black across all 4. But secondary grey differs: baseline `#333333` (TTFT callout + double-arrow) vs gap/ablation `#444444`. Two near-blacks; baseline is the minority — unify to `#444444`. |
| **C3 semantic colour → meaning** | **fail** | baseline/tau/gap mutually consistent: FCFS/weak=blue `#2E5E8C`, listMLE/mid=amber `#E8A33D`, LTR/PARS/ours=vermillion `#C2452D`, neutral=grey `#7A7A7A`. `ablation` REMAPS the same hexes to a waterfall: base-listMLE and PARS-full bars are BLUE (not amber / not vermillion), and VERMILLION denotes the +BERT-backbone Δ-step, NOT "ours". Only grey=no-help is preserved. Net: the winner (PARS) prints blue while a single Δ-step prints vermillion — a reader who learned "vermillion = ours" misreads ablation. The one true set break. |
| **C4 type-size tiers** | warn | Core tiers identical: title 13 bold, axis/tick/body 12, annotation ~11–11.5 across all 4. Two minor deviations: gap is the ONLY figure with a 9–9.5pt micro tier (italic "drop" labels + footnote); baseline adds a ~14.4pt suptitle tier (justified — it is the only 2-panel composite). ablation bumps the emphasized "+0.065" to 13pt (intentional, B5-OK). |
| **C5 physical text scale** | warn | All dpi=300, all sizes absolute pt, but figwidth spread is large: tau 6.0in, gap 7.2in, ablation 8.5in, baseline 11in → effective pt/in = 2.00 / 1.67 / 1.41 / 1.09. Scaled to a common column width, tau prints ~1.83× larger than baseline; baseline (widest, 2-panel) prints SMALLEST — the exact "make it larger so I can see" risk (A2). Bump baseline base font or trim width toward the ~1.4–1.7 pt/in band. |
| **C6 grayscale + colourblind** | warn | Okabe-Ito-derived: blue+vermillion+amber separate well under deutan/protan. But L\* clustering hurts pure grayscale: blue 38.7, vermillion 46.9, grey 51.2 within ~12.5 L\* (amber 72.0 is the only clearly separable one). vermillion-vs-grey (tau bars; ablation Δ-steps) and blue-vs-vermillion (baseline lines, same `o` marker) are marginal in B/W — legible here only because position + value labels disambiguate. Set-wide limitation, not a single-figure fix. |
| **C7 zero overlap** | OK | No text-on-text, text-on-marker, or arrow-through-text in any of the 4 rendered PNGs. Near-passes only: tau's curved arrow grazes the listMLE bar top (label stays clear); gap's italic "drop" labels hug the slope lines (still readable). Acceptable. |
| **A1/A4 number + claim integrity** | OK | Shared numbers agree exactly across figures: listMLE in-dist 0.559 and PARS in-dist 0.596 match tau↔gap; cross-dist 0.315/0.361 match gap↔ablation; ablation deltas sum (0.315−0.012=0.303, +0.065=0.368, −0.007=0.361). Claims honestly hedged: ablation "sole driver … on this setup", gap "overfitting persists … does not eliminate it", baseline keeps the honest p99-TPOT cost panel. |

## 4. Fixed this pass

Two figures were edited this pass (`figures_to_fix` = `ablation`, `baseline`):

- **ablation** — addressed the C3 colour-map break: reconciled / explicitly annotated the waterfall colour semantics so a reader carrying "vermillion = ours" from tau/gap is not misled (blue = absolute level endpoint, grey = no-help step, vermillion = driver step). Internal coherence preserved; the cross-set inconsistency is now flagged rather than silent.
- **baseline** — addressed the C2 near-black break: unified the `#333333` secondary callout text and its two leader arrows to a single dark tone matching the pure-black primary text, so the figure uses one dark text colour.

Residual items in §2 for these two are all **minor/legibility-only** (slide-scale font bumps, optional gridline trim, arrow-clearance verify) — none blocking. `tau` and `gap` were not edited this pass; their residuals are likewise non-blocking.

## 5. Ryoo-standard note

- **Numbers verified against ground truth.** All headline values were checked, not trusted: tau bars 0.194 / 0.559 / 0.596 and the 3.1× ratio match GT; shared cross-figure numbers agree exactly (listMLE 0.559, PARS 0.596, cross-dist 0.315→0.361); ablation deltas sum correctly to the printed endpoints. No fabricated data or results.
- **Claims are associational, not causal.** Wording is hedged where a mechanism could be over-read: ablation "sole driver … on this setup", gap "overfitting persists … does not eliminate it". The remaining nits (ablation "sole driver" phrasing, tau "3.1x higher", "+15%" rounding 14.6%) are flagged for tightening but are non-blocking because exact endpoints are shown alongside every rounded/emphatic label.

---
File: `/Volumes/T7 Shield/obsidian/4-Resources/Courses/VPL/FDUClasses/26VU_CSCI_6806_V1 Computer Sci Gr Capstone Proj/deliverables/04-evaluation/figures/FIGURE_CONSISTENCY_CHECK.md`
