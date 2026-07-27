# Publication-v2 figure set

This directory preserves the original report figures and adds an audited set for the capstone presentation/report workflow.

## Contents

| Figure | Purpose | Evidence note |
|---|---|---|
| Fig.1 | Implemented request/control architecture and ownership boundary | `FIGURE_01.md` |
| Fig.2 | Offline data-to-artifact lineage | `FIGURE_02.md` |
| Fig.3 | Legacy shortest-first mechanism baseline and tail cost | `FIGURE_03.md` |
| Fig.4 | Tier-2 predictor comparison and single-seed learning curve | `FIGURE_04.md` |
| Fig.5 | Main MIXED live-serving result and heuristic control | `FIGURE_05.md` |
| Fig.6 | One BFCL-only common-complete workload-shift result | `FIGURE_06.md` |
| Fig.7 | Small live reliability probe plus simulated workload mixture | `FIGURE_07.md` |
| Fig.8 | Observed paired path difference in one ordered replay | `FIGURE_08.md` |

Every figure has vector PDF and 300 dpi PNG output. The Markdown notes record purpose, source data, experiment rationale, reading guide, result, limitation, reproducibility, and a one-line oral explanation.

## Regenerate

From the repository root:

```bash
python -B scripts/report_figures/publication_v2/render_all.py
```

## Scope

- Figures organize measured evidence; they do not create new measurements.
- Fig.3 is a legacy single sweep, not the current Qwen stack.
- Fig.6 uses a common-complete estimand and must not be captioned with the older per-policy-complete values.
- Fig.7 does not isolate a gate-only causal effect.
- Fig.8 reports one non-counterbalanced ordered replay, not a universal causal overhead constant.
- Final-report prose and captions remain student-authored under course rules.
