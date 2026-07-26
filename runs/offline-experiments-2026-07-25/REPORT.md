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

---

# T1 — S1–S4 re-stratification + grid-searched baseline (2026-07-26)

Ticket: issue #5. Spec: issue #4. Script `t1_strata.py`, artifact `t1-strata.json`,
log `t1.log`. Wall clock 36.7 s. Zero new BERT inference — model scores are read back
from `e2-bert-test-scores.jsonl`; only the grid-searched LightGBM was trained here.

## Stratum sizes (reported first, per the tracer bullet)

Strata follow the CONTEXT.md *Cold-Start Transfer* glossary entry. S1 is defined by
fingerprint alone; S2–S4 partition the remainder by tool-name novelty.

| Stratum | Definition | n |
|---|---|---:|
| S1 | seen-combination (fingerprint appears in train) | **45** |
| S2 | new combination, every tool name seen in train | **78** |
| S3 | partial-new tools (some seen, some not) | **543** |
| S4 | all-new tools (no tool name seen in train) | **333** |
| all | — | 999 |

Partition verified exhaustive and disjoint: 45 + 78 + 543 + 333 = 999, with zero rows
left unstratified. 20 of the 45 S1 rows advertise no tools at all; their empty-tool-list
fingerprint does appear in train, so S1 is where they legitimately land. Identity key is
SHA-256 over the sorted top-level tool-name list via the E1 multi-template parser — the
raw `tool_schema` string is never hashed, because 2383 rows embed a per-row timestamp
that would make almost every row its own unique identity.

**S1 (45) and S2 (78) are both below the ticket's n<100 bar, so their τ is withheld.**
Only S3 and S4 carry reportable τ. See the open question at the end — this collides with
one of the pre-registered criteria.

## The table — test Kendall τ-b, mean ± std over seeds 17/42/73, [95% CI]

CIs are session-clustered bootstrap, 1000 iterations, computed on the seed-17 predictions.

| Model | S1 (45) | S2 (78) | S3 (543) | S4 (333) | all (999) |
|---|---|---|---|---|---|
| **BERT prompt_schema** (schema TEXT) | withheld | withheld | **0.6468** ±.0109 [.619,.692] | **0.6393** ±.0081 [.582,.675] | **0.6302** ±.0105 [.613,.668] |
| BERT prompt_only (control) | withheld | withheld | 0.6247 ±.0118 [.587,.662] | 0.6230 ±.0085 [.587,.675] | 0.5865 ±.0093 [.561,.624] |
| **LightGBM grid-searched** (baseline of record) | withheld | withheld | 0.3987 ±.0000 [.352,.443] | 0.5008 ±.0000 [.444,.554] | **0.4395** ±.0000 [.407,.472] |
| LightGBM fixed (E3) | withheld | withheld | 0.3854 ±.0000 [.331,.436] | 0.4842 ±.0000 [.431,.538] | 0.4268 ±.0000 [.391,.461] |
| schema-hash lookup (E1b) | withheld | withheld | 0.3965 ±.0109 [.335,.434] | 0.4872 ±.0026 [.431,.539] | 0.4348 ±.0069 [.394,.462] |
| schema-hash categorical (E1a) | withheld | withheld | 0.3854 ±.0000 [.331,.436] | 0.4842 ±.0000 [.431,.538] | 0.4268 ±.0000 [.391,.461] |

Do **not** compare τ across strata — they differ in intrinsic difficulty and label
distribution. Only within-stratum, between-model comparisons are sound.

## Grid-searched LightGBM

`offline_baselines.run_lightgbm_grid`, same five scalar features, same fixed split,
selection on validation, one test evaluation per seed.

- **The grid is 20 configs, not the 40 stated in the ticket.** `lightgbm_grid()` yields
  2 × 2 × 2 × 2 = 16 combinations plus 4 hand-added ones. The ticket's "40 configs" is
  wrong; 20 is what exists and what ran.
- All three seeds selected the identical best config — `max_depth=3, num_leaves=7,
  learning_rate=0.1, n_estimators=300` — with validation τ-b 0.4586 and test τ-b 0.4395.
  Std is again exactly 0.0000, for the same structural reason as E3: no stochastic
  sampling consumes `random_state`.
- Grid (0.4395) beats fixed (0.4268) by +0.0127, so **the grid-searched model is the
  scalar baseline of record**. The "under-tuned baseline" objection is closed.

### Consequence for the headline number

| Comparison | Δτ |
|---|---:|
| prompt_schema − LightGBM fixed | +0.2034 |
| **prompt_schema − LightGBM grid (baseline of record)** | **+0.1907** |

The ratified spine claim says "+0.20 τ". Against the tuned baseline it is **+0.19**.
That is a wording fix, not a survival problem — but the sentence should say +0.19, or
say "+0.20 against the deployed fixed-hyperparameter baseline, +0.19 against a tuned one".

The 79/21 decomposition also shifts slightly against the tuned baseline:

| Step | vs fixed | vs grid |
|---|---:|---:|
| encoder (LightGBM → BERT prompt_only) | +0.1597 (79%) | +0.1470 (77%) |
| schema text (prompt_only → prompt_schema) | +0.0437 (21%) | +0.0437 (23%) |

So the honest decomposition is now **~77% encoder / ~23% schema text**.

## Pre-registered criteria, re-evaluated verbatim

Criteria as frozen in `docs/DIRECTION-DECISION-2026-07-25.md`, claim model =
BERT prompt_schema, baseline = LightGBM grid-searched:

| Stratum | n | Claim τ | Baseline τ | Δτ | Primary: CIs separated | Secondary: Δτ ≥ 0.05 |
|---|---:|---:|---:|---:|:--:|:--:|
| S1 | 45 | — | — | — | not evaluated (n<100) | not evaluated |
| S2 | 78 | — | — | — | not evaluated (n<100) | not evaluated |
| S3 | 543 | 0.6468 | 0.3987 | **+0.2481** | **PASS** [.619,.692] vs [.352,.443] | **PASS** |
| S4 | 333 | 0.6393 | 0.5008 | **+0.1386** | **PASS** [.582,.675] vs [.444,.554] | **PASS** |

**Both reportable unseen strata pass both criteria against the tuned baseline.** The
primary criterion (CI separation on unseen strata) is met with a wide margin on S3 and a
clear one on S4.

## Open question for the session lead — two ratified rules collide

The pre-registered **seen-stratum tie bar** (`Δτ < 0.02` on the seen stratum) cannot be
evaluated, because the ticket's **n<100 rule** withholds τ on S1 (n=45) — and S1 *is* the
seen stratum. Both rules are ratified; they are mutually exclusive here.

This is left unresolved rather than silently decided. The options:

1. Keep the n<100 rule; record the S1 tie bar as **not evaluable** at this sample size.
   Costs nothing — the tie bar was pre-registered as "does not kill" either way.
2. Grant S1 an explicit exemption and report its τ with a loud suggestive-only caveat.
   The E2 numbers already exist (prompt_schema 0.6012, prompt_only 0.2385, scalar 0.6533,
   CIs ±0.15 wide and heavily overlapping).

Recommendation: **option 1**. The tie bar was declared non-killing in advance, n=45 gives
CIs too wide to adjudicate a 0.02 threshold, and the existing "seen-combination n=45 is
suggestive-only" honesty caveat already covers the ground. Option 2 reports a number that
cannot support the test being asked of it.

Note also that S2 (n=78) falls below the bar, so the "new combination, all tools seen"
regime — arguably the most deployment-relevant cold-start case — has **no reportable τ**
in this test split. If that stratum matters to the story, it needs a larger test split,
not a re-analysis.

## Reproduction

```bash
V=/Users/alex/develop/vllm-ltr-optimization/.worktrees/final-training-artifacts/.venv/bin/python
cd /Users/alex/develop/vllm-ltr-optimization/runs/offline-experiments-2026-07-25
$V t1_strata.py            # 37 s, writes t1-strata.json
```

Implementation note: `t1_strata.py` imports `lightgbm` **before** anything that pulls in
torch. Importing it afterwards loads a second OpenMP runtime and segfaults (SIGSEGV,
exit 139) on the first `fit`. E3 only worked because its import order happened to be
correct; this is now a load-bearing comment at the top of the file.
