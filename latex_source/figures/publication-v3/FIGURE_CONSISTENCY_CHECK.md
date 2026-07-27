# Publication-v3 Figure Consistency Check

Date: 2026-07-22  
Scope: Figures 1–8 in `latex_source/figures/publication-v3/`  
Overall verdict: **PASS — no remaining cross-figure blocker**

## Review basis

- Inspected a full-resolution two-column montage and every 300 dpi PNG at full size.
- Parsed all eight publication SVGs and PDFs, the Figure 1–2 native `.drawio` files, and the XML embedded in `fig1.drawio.svg` and `fig2.drawio.svg`.
- Cross-checked figure wording against `FIGURE_01.md` through `FIGURE_08.md`, the publication-figure design spec, generator code, and the underlying evidence loaders.
- Ran `python -m pytest -q tests/test_publication_v3_figures.py`: **17 passed**.

The active publication-v3 adaptation uses the requested 10/8/7 pt hierarchy and reserves vermillion for warnings. The older publication-v2 design-spec sentences requiring all text at least 10 pt and assigning vermillion to the gated policy are therefore treated as superseded style rules; its evidence and statistical-scope rules remain enforced.

## Consistency checklist

### Typography and text color — PASS

- All eight SVGs use Arial with Helvetica/sans-serif fallback; all PDFs embed Arial-family fonts.
- The shared hierarchy is 10 pt for figure claims and panel labels, 8 pt for panel titles, axes, legends, and primary annotations, and 7 pt for micro/scope text.
- Figures 1–2 encode the same physical hierarchy on a 1,800-unit-wide SVG: 35 units ≈ 10 pt, 28 units ≈ 8 pt, and 24.5 units ≈ 7 pt; 31-unit node titles provide an intermediate ≈9 pt node role.
- Primary text is consistently near-black `#1A1A1A`; secondary structural text uses neutral gray `#4A4A4A`.

### Semantic palette — PASS

- Blue `#0072B2`: learned predictors/schedulers, reliable paths, primary estimates, and measured distributions.
- Neutral gray `#4A4A4A`: FCFS/reference paths, structural elements, and zero/reference lines.
- Amber `#E69F00`: fallback state, simple non-learned Prompt SJF, and the single-seed LightGBM comparator.
- Vermillion `#D55E00`: provenance, validation-only, and scope warnings; it is not reused as a learned-policy encoding.
- Low-alpha blue fills consistently denote uncertainty or density rather than a separate policy.

### Panel labels, strokes, legends, and scope notes — PASS

- Panel labels use bold `(a)`, `(b)`, and `(c)` consistently; panel order matches each Markdown explanation.
- Data strokes are visually dominant over grids; zero/reference lines are neutral and dashed; uncertainty whiskers/bands are distinct from point estimates.
- Legends do not cover decision-driving data and consistently explain line/marker/repeat/CI encodings.
- Scope treatments match evidentiary severity without changing their meaning: placeholder note in Fig.1, evaluation-only tag in Fig.2, compact footer in Fig.3, warning callout in Fig.4, clustering footer in Fig.5, invalid-provenance line in Fig.6, and limitation strips in Figs.7–8.

### Policy and artifact naming — PASS

- `Stock FCFS`, `FCFS shim`, `Pure LTR`, `Gated hybrid`, `Tail safe`, `LTR aging`, and `Prompt SJF (non-learned)` are stable across Figs.5–6.
- `Legacy shortest-first` is confined to the legacy sweep in Fig.3.
- `Gated policy bundle` is used in Fig.7 because that simulation changes gate, aging, and tail-safe behavior; the distinct `Gated hybrid` label remains confined to the live-serving matrix in Figs.5–6.
- Figure 2 keeps the checkpoint lineage and the 6,000-label empirical lookup lineage separate.

### Fixed canvas and vector editability — PASS

| Figure | SVG canvas | PDF page | Vector/editability evidence |
|---|---|---|---|
| 1 | 181.9 × 78.8 mm | 515.622 × 223.370 pt | Native draw.io + embedded XML + editable SVG text; no raster image |
| 2 | 181.9 × 78.8 mm | 515.622 × 223.370 pt | Native draw.io + embedded XML + editable SVG text; no raster image |
| 3 | 181.9 × 84.0 mm | 515.622 × 238.110 pt | Editable Matplotlib SVG text; no raster image |
| 4 | 181.9 × 86.0 mm | 515.622 × 243.780 pt | Editable Matplotlib SVG text; no raster image |
| 5 | 181.9 × 92.0 mm | 515.622 × 260.787 pt | Editable Matplotlib SVG text; no raster image |
| 6 | 181.9 × 78.0 mm | 515.622 × 221.102 pt | Editable Matplotlib SVG text; no raster image |
| 7 | 181.9 × 86.0 mm | 515.622 × 243.780 pt | Editable Matplotlib SVG text; no raster image |
| 8 | 181.9 × 80.0 mm | 515.622 × 226.772 pt | Editable Matplotlib SVG text; no raster image |

- Every PDF is one page and contains zero embedded raster images.
- Figure 1 has 11 expanded draw.io edges and Figure 2 has 3; every edge contains `mxGeometry relative="1"`.
- For Figures 1–2, embedded draw.io XML matches the retained native source after whitespace normalization, and the visible `*.drawio.svg` trees match the publication SVG trees.

### Statistical and evidentiary language — PASS

- Fig.1 calls confidence/OOD a placeholder and shows the exact reliable/no-estimate transport contract.
- Fig.2 uses validation for run selection, labels held-out tau as evaluation only, and has no selected-run or held-out edge into the empirical lookup.
- Fig.3 separates the 16 qps mean-TTFT comparison from the 64 qps p99-TPOT comparison and states that it is a single sweep without repeated runs.
- Fig.4 calls BERT whiskers observed seed min–max rather than confidence intervals and labels the learning curve validation-only and single-seed.
- Fig.5 identifies the CCDF as representative, labels Prompt SJF non-learned, exposes three matched repeats, and describes 150 recurring IDs rather than 450 independent requests.
- Fig.6 limits the claim to one invalid-provenance BFCL common-complete subset, discloses 7 excluded error rows, `valid=false`, and the absence of a prompt control.
- Fig.7 separates the small live ordering probe from simulation and states that the gated configuration is a policy bundle, not an isolated gate ablation.
- Fig.8 keeps the 10.4 s outlier on one continuous axis and describes request-bootstrap intervals as conditional on one direct-first, gateway-second replay, not causal overhead.

## Per-figure verdicts

| Figure | Verdict | Cross-figure evidence |
|---|---|---|
| Fig.1 | **PASS** | Typography/palette match; runtime and decision seam are synchronized across draw.io/SVG and preserve the placeholder reliability boundary. |
| Fig.2 | **PASS** | Typography/palette match; validation selection, checkpoint, held-out evaluation, and label-derived lookup remain separate and synchronized. |
| Fig.3 | **PASS** | Shared plot style, policy legend, operating-point annotations, log axes, and single-sweep scope are consistent. |
| Fig.4 | **PASS** | Shared plot style; observed seed ranges and validation-only learning curve use distinct, correctly scoped encodings. |
| Fig.5 | **PASS** | Shared learned/control palette; representative CCDF, paired reductions, repeats, and clustered intervals remain legible and honest. |
| Fig.6 | **PASS** | Shared learned palette; invalid-provenance warning and common-complete accounting are visually dominant and statistically scoped. |
| Fig.7 | **PASS** | Shared typography and blue learned-policy encoding; live/simulation split and policy-bundle confound remain explicit. |
| Fig.8 | **PASS** | Shared typography and distribution encoding; mean/CI key, continuous outlier axis, and ordered-replay limitation remain explicit. |
