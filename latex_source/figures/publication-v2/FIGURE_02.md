# Figure 2 — Offline artifact lineage

## Purpose

Show how pinned source declarations become a frozen sample, validation-selected training result, and deployment artifacts without presenting stored provenance as live-service evidence.

## Data

- Source and model pins: `configs/training_sources.json`.
- Frozen sample and split metadata: `scripts/report_figures/data/offline/tier2-sample-manifest.json`.
- Recorded training matrix: `scripts/report_figures/data/offline/tier2-matrix-summary.json`.
- Recorded learning curve: `scripts/report_figures/data/offline/tier2-learning-curve.json`.
- Offline evidence ledger: `scripts/report_figures/data/offline/SHA256SUMS.txt`.
- Static lineage relationships: `scripts/plot_final_report_figures.py` (`FIG2_COMPONENTS`, `FIG2_EDGES`).

## Why this experiment

This lineage makes the offline evidence reproducible and separates committed training artifacts from the runtime serving path.

## How to read

Read left to right across the top: pinned declarations → frozen raw split and effective analyzed counts → recorded training summaries. Model selection uses validation Kendall τ; the separately labeled held-out test τ is evaluation, not the selection criterion. The two lower branches are artifacts produced from the offline run.

## Result

The committed manifest records a raw seed-42 split of 4,000/1,000/1,000 train/validation/test rows. After recorded censor/failure exclusions, the analyzed counts are 3,997/998/999. The complete 10/10-run Tier-2 matrix selects `bert-prompt_schema-tier2-seed17` by validation τ = 0.657; its held-out test τ is 0.642. Short hashes shown in the figure are parsed or computed from current files at render time.

## Limitation

Figure 2 is offline artifact lineage only. Raw split counts are not analyzed counts, and held-out test τ is not a selection score. The figure does not load the runtime rank manifest or claim that any live gateway, decision service, checkpoint copy, or vLLM process is currently deployed.

## Contribution boundary

Dazhi implemented and evaluated the predictor/ranker and scheduling thread. ToolACE, Qwen, BERT, and the dataset/model artifacts are external inputs rather than Dazhi's inventions. Gateway infrastructure and reusable evaluation ownership remain with Mingye and Yibo respectively.

## Reproducibility

Run `python -B scripts/report_figures/publication_v2/figures_01_03.py`. `build_fig2()` reads the paths above on every render; no displayed provenance hash is copied as a hard-coded live-state claim.

## Tomorrow's one-line explanation

The checkpoint is selected by validation evidence, evaluated once on the held-out test split, and traced through frozen offline artifacts; live deployment status stays out of scope.
