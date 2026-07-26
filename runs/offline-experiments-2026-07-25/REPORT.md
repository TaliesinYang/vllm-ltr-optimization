# Offline experiments E3 / E1 / E2 — 2026-07-25 evening

Assembled by the session lead from the executing agent's structured reports (the agent's
harness blocked it from writing this file). Every number below is also in the committed
JSON artifacts in this directory; on any discrepancy the JSON wins.

Commits: `d0ca541` (E3), `064f9ed` (E1), `f3067ec` (E2). Total compute ≈ 7 minutes on Mac CPU.

## Data provenance

- Sample: `/Volumes/T7 Shield/vllm-ltr-results/rebuild/tier2-toolace-sample-6000.jsonl`,
  sha256 `ee5a5889…` — matches the recorded value; verified programmatically on every run.
- Splits are **pre-assigned per row** (`tier2_split` column): train 3997 / val 998 / test 999
  after censor/failure filtering. Seed does NOT control the split — it only sets
  `random.seed`/`torch.manual_seed` and pair shuffling. One fixed split for all runs.
- BERT rescoring reproduced recorded per-seed test τ to delta 0.0 (prompt_schema) /
  ≤2.6e-06 (prompt_only); checkpoint SHAs match `runs/offline-evidence-r1/scoring-report.json`.

## E3 — LightGBM same-recipe, seeds 17/42/73

| Model | s17 | s42 | s73 | mean ± std |
|---|---:|---:|---:|---|
| LightGBM structural (scalar) | 0.4268 | 0.4268 | 0.4268 | 0.4268 ± 0.0000 |
| BERT prompt_only | 0.5922 | 0.5758 | 0.5916 | 0.5865 ± 0.0093 |
| BERT prompt_schema | 0.6423 | 0.6252 | 0.6231 | 0.6302 ± 0.0105 |
| BERT full_context | 0.6170 | 0.6365 | 0.6248 | 0.6261 ± 0.0098 |

- The 0.0000 std is **correct, not a bug**: the recipe (`tier2_training.train_lightgbm_tier2`,
  fixed hyperparameters — the source of the recorded 0.4268, NOT `run_lightgbm_grid`) uses
  `subsample=1.0`, `colsample_bytree=1.0`, so nothing consumes `random_state`. Write-up
  framing: "the baseline is deterministic w.r.t. seed, verified by rerun" — NOT "we found
  low variance". The single-seed objection is void by construction.
- Real uncertainty band: test τ-b 0.4268, 95% CI [0.3907, 0.4611] (session-clustered
  bootstrap, 1000 iters, 809 sessions).
- τ computed via both `scipy.stats.kendalltau` and `offline_statistics.kendall_tau_b`;
  agreement < 1e-12.
- Open item: grid-searched LightGBM (40 configs) not yet run; ~1 min; closes the
  "under-tuned baseline" objection.

## E1 — schema identity (hash) baseline

| Model | test τ-b (3 seeds) |
|---|---|
| LightGBM scalar (ref) | 0.4268 ± 0.0000 |
| E1a: + fingerprint as native categorical | 0.4268 ± 0.0000 (feature never split on; importance 0) |
| E1a-relaxed: categorical guardrails off | 0.4131 ± 0.0000 (fingerprint becomes top-gain feature → WORSE: memorisation) |
| E1b: + per-hash train-label lookup (out-of-fold, K=5) | 0.4348 ± 0.0069 |
| BERT prompt_schema | 0.6302 ± 0.0105 |

- Identity buys +0.0080 — **not distinguishable from noise** (scalar CI is ±0.035 wide).
  Correct statement: "identity does not measurably help", NOT "helps a little".
- Mechanism: only **45/999 (4.5%)** test rows share a tool-set fingerprint with train;
  train has 3162 unique fingerprints over 3997 rows (≈1.26 rows each). A lookup table has
  almost nothing to look up — schema text generalises because it is compositional.
- Identity key = SHA-256 over sorted top-level tool names, via a parser handling ≥6
  ToolACE system-prompt templates (JSON, YAML-ish, markdown, XML, HTML table, LaTeX);
  5/5994 rows legitimately advertise no tools.
- **Do not hash the raw `tool_schema` string**: 2383 rows embed a per-row timestamp
  ("The current time is …"), which would make nearly every row unique and silently fake a
  cold-start result. Methods-section material.
- Encoding: target encoding (out-of-fold, K=5; unseen → global train mean/median,
  count 0, seen-flag 0) — chosen because it IS the deployed per-client lookup table under
  test; native categorical reported alongside.

## E2 — cold-start evaluation

Subsets of the fixed test split (n=999): seen_combination 45 · unseen_combination 954 ·
strict unseen_tools 333 (no individual tool name appears in any train row; nested in
unseen_combination; 20 no-tool rows excluded). Train vocabulary: 3162 fingerprints,
7690 tool names.

Test τ-b, mean ± std over 3 seeds, [95% CI, session-clustered bootstrap]:

| Model | all (999) | seen_comb (45) | unseen_comb (954) | unseen_tools (333) |
|---|---|---|---|---|
| BERT prompt_schema | 0.6302 ±.0105 [.613,.668] | 0.6012 ±.0357 [.483,.777] | 0.6274 ±.0071 [.605,.664] | 0.6393 ±.0081 [.582,.675] |
| BERT prompt_only | 0.5865 ±.0093 [.561,.624] | 0.2385 ±.0576 [.062,.481] | 0.6129 ±.0060 [.587,.646] | 0.6230 ±.0085 [.587,.675] |
| LightGBM scalar | 0.4268 [.391,.461] | 0.6533 [.457,.805] | 0.4041 [.367,.442] | 0.4842 [.431,.538] |
| schema-hash lookup | 0.4348 ±.0069 | 0.5823 ±.0231 | 0.4158 ±.0072 (0 lookup hits) | 0.4872 ±.0026 (0 hits) |

Findings:

1. **Pre-registered survival criterion PASSED**: on unseen strata, text vs hash/scalar CIs
   are fully separated (+0.212 within unseen_comb, +0.152 within unseen_tools). The
   lookup takes zero hits on both unseen subsets and degrades to its fallback exactly as
   designed.
2. **79/21 decomposition** (the control finding): LightGBM→prompt_only = +0.1597 (79%),
   prompt_only→prompt_schema = +0.0437 (21%). On cold-start strata schema text adds only
   ~+0.015. Caveat: the first step bundles encoder-vs-trees with text-vs-scalars — state
   this, don't overclaim precision.
3. **Schema earns its keep on repeated schemas**: seen_combination — prompt_only 0.2385
   vs prompt_schema 0.6012. n=45, wide CIs, treat as suggestive; but the separation
   survives the CI. This is the regime the OpenCode probe shows real deployments live in.
4. **Do not compare τ across subsets** (different intrinsic difficulty — LightGBM *rises*
   to 0.4842 on the strict subset). Only within-subset, between-model comparisons are valid.
5. LightGBM 0.6533 vs prompt_schema 0.6012 on seen_comb is NOT a reversal (n=45, CIs
   [.457,.805] vs [.483,.777] overlap heavily). Do not report it as one.

## Artifacts

`e3-lightgbm-seeds.json` · `e1-schema-hash.json` · `e1-categorical-sensitivity.json` ·
`e2-cold-start.json` · `e2.log` · scripts `common.py`, `e3_lightgbm_seeds.py`,
`e1_schema_hash.py`, `e1_categorical_sensitivity.py`, `e2_cold_start.py`.
Per-row predictions (`*.jsonl`) gitignored; regenerable in <7 min. Per-row BERT scores
(`e2-bert-test-scores.jsonl`, 6 checkpoints) make S1–S4 re-stratification a few seconds of
CPU with zero new inference.

Note: the venv `.worktrees/final-training-artifacts/.venv` gained scipy, lightgbm,
scikit-learn (installed 07-25).
