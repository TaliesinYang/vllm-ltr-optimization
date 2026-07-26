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

---

# E4 — cached two-tower Ranker (2026-07-26)

Ticket: issue #8. Scripts `e4_embed.py`, `e4_fusion.py`, `e4_latency.py`; artifacts
`e4-embeddings-meta.json`, `e4-fusion.json`, `e4-latency.json`; logs `e4_embed.log`,
`e4_fusion.log`, `e4_latency.log`. Compute: 24 min embedding precompute, 15 s fusion
training, 89 s latency measurement.

## Architecture

Two towers over one **frozen** encoder (the fine-tuned `bert-prompt_schema-tier2-seed17`
checkpoint, keeping the encoder in-family with the single-tower baseline):

- **prompt tower** — `[USER]\n{prompt}`, per request, 512 cap.
- **schema tower** — `[TOOLS]\n{tool_schema}`, precomputed per schema body hash.
- **fusion** — MLP (256 hidden) over `[prompt_emb, schema_emb]`, trained with the *same*
  pairwise margin loss and same-generator pair construction as the single-tower ranker
  (`train_ranker.build_pair_indices` / `pairwise_margin_loss`), same fixed split,
  seeds 17/42/73.

Only the fusion head is trained. This is the "frozen towers + MLP fusion on cached
embeddings" option the brief pre-authorised; full end-to-end fine-tuning on CPU was not
attempted and is not needed to answer either question.

## Schema length — why the truncation question is worth asking

| Statistic | Value |
|---|---:|
| unique schema bodies | 4732 (over 5994 rows) |
| median schema tokens | **691** |
| p95 / max | 1382 / 2954 |
| **schemas exceeding the 512 cap** | **3375 of 4732 (71%)** |
| mean 512-token windows per schema | 1.94 |

So the single-tower model really is discarding most of the schema on most requests. The
hypothesis was well-motivated. It is still wrong — see below.

## Question (ii): does un-truncating the schema recover τ? **No.**

The controlled contrast — one encoder, one fusion recipe, one split, truncation the only
thing that moves:

| Stratum | n | full | trunc512 | Δτ (full − trunc) | CIs overlap |
|---|---:|---:|---:|---:|:--:|
| S1 | 45 | — | — | — | withheld (n<100) |
| S2 | 78 | — | — | — | withheld (n<100) |
| S3 | 543 | 0.6291 | 0.6283 | **+0.0008** | yes |
| S4 | 333 | 0.6193 | 0.6306 | **−0.0112** | yes |
| all | 999 | 0.6079 | 0.6145 | **−0.0067** | yes |

**Reading the whole schema does not help, and is very slightly worse on S4 and overall.**
Every interval overlaps; the effect is indistinguishable from zero in both directions.

This kills the truncation-cap hypothesis in the ratified spine claim, which currently
says the schema's contribution "is capped by the current 512-token truncation, a
hypothesis E4's cached two-tower encoder tests." E4 tested it. The cap is not the
binding constraint. That sentence needs to change.

Plausible reading (not tested here): mean-pooling 1.94 windows dilutes the signal about
as much as truncation discards it, so the tail of a long tool list carries little
length-relevant information beyond what the first 512 tokens already convey. Testing
that would need a different pooling rule, not a different cap.

## τ table — two-tower vs the T1 baselines

| Model | S3 (543) | S4 (333) | all (999) |
|---|---|---|---|
| BERT prompt_schema single-tower (fine-tuned) | 0.6468 ±.0109 | 0.6393 ±.0081 | **0.6302** ±.0105 |
| BERT prompt_only single-tower (fine-tuned) | 0.6247 ±.0118 | 0.6230 ±.0085 | 0.5865 ±.0093 |
| two-tower trunc512 (frozen towers) | 0.6283 ±.0055 | 0.6306 ±.0045 | 0.6145 ±.0037 |
| two-tower full (frozen towers) | 0.6291 ±.0079 | 0.6193 ±.0109 | 0.6079 ±.0058 |
| LightGBM grid (baseline of record) | 0.3987 ±.0000 | 0.5008 ±.0000 | 0.4395 ±.0000 |

**The two-tower rows are NOT a controlled comparison against the single-tower rows.**
Those towers are frozen; the single-tower baselines were fine-tuned end to end. A
two-tower deficit (−0.016 on all) mixes frozen-vs-fine-tuned with
two-tower-vs-single-tower and must not be read as a truncation or architecture result.
The only clean claim from this experiment is the trunc-vs-full contrast above.

Worth noting anyway: two-tower with frozen towers lands within ~0.016 τ of the fine-tuned
single tower overall, and *above* `prompt_only` — so the architecture is not obviously
lossy, it simply was not given a fine-tuning budget.

## Question (i): per-request latency at the `/v1/decision` contract

Real HTTP through the unchanged `DecisionApplication`; no contract change. Protocol
mirrors `scripts/server/measure_decision_latency.sh` — 20 warm-up calls discarded, then
200 measured samples. Mac CPU, torch intra-op threads = 2 (the deployed default), same
machine for every row.

| Configuration | conc. | p50 ms | p95 ms | p99 ms | × 15 ms contract | × 7 ms JITServe QRF |
|---|---:|---:|---:|---:|---:|---:|
| single-tower (deployed shape) | 8 | 628.8 | 698.3 | 722.0 | **48.1×** | 103.1× |
| two-tower, schema cached | 8 | 222.1 | 353.3 | 377.2 | **25.1×** | 53.9× |
| two-tower, cold cache | 8 | 249.1 | 1650.3 | 1950.1 | 130.0× | 278.6× |
| single-tower | 1 | 114.5 | 117.7 | 119.6 | 8.0× | 17.1× |
| two-tower, schema cached | 1 | 37.4 | 57.2 | **64.1** | **4.3×** | 9.2× |

Cache accounting: the cached run took 220 hits / 0 misses; the cold run 156 / 64 (64
distinct schemas, each encoded once). Throughput rose from 12.8 to 33.8 rps at
concurrency 8.

**Caching the schema tower buys a consistent ~1.9× at p99** (1.91× at concurrency 8,
1.87× serial). It does not buy an order of magnitude.

### Honest verdict against the pre-registered kill condition

The kill condition for the deployability leg is "per-request cost still ≥ heuristic
budget by orders of magnitude (JITServe's 7 ms QRF is the published bar)".

Under the protocol this ticket specified (concurrency 8): **25× the contract and 54× the
QRF bar — the kill condition is met.** E4 does not rescue deployability on CPU.

The serial diagnostic complicates that, and should be stated rather than buried: at
concurrency 1 the cached two-tower is 64 ms p99, **4.3×** the contract — within one order
of magnitude. Most of the concurrency-8 gap is CPU contention on this machine (8 handlers
× 2 torch threads on a laptop), not model cost. So the honest summary is: *the cached
two-tower is roughly 4× over budget in per-request compute, and the measured 25× is
hardware contention on top of that.* Whether a dedicated box or GPU closes the remaining
4× is untested and should not be assumed.

Either way, no pass/fail theater: on the evidence specified, the artifact does not meet
the 15 ms contract, and it is nowhere near JITServe's 7 ms.

## Thread-safety note

HuggingFace fast tokenizers are not thread-safe — `enable_truncation` mutates shared Rust
state and concurrent callers can hit `RuntimeError: Already borrowed`. The two-tower
predictor reproduced this reliably (it tokenizes twice per request), so it uses a
per-thread tokenizer.

The shipped `BertPredictor` **survived** an 8-way × 24-call probe at this contract. That
is evidence it does not race easily at this load — **not** evidence that it is
thread-safe, and it is not being reported as a defect. The single-tower row is measured
with a `ThreadSafeBertPredictor` subclass for run stability; it differs from the shipped
class only in tokenizer ownership, not in the model or the scoring path.

## Caveat on the caching premise

In this dataset the schema cache is nearly useless: 4732 unique schema bodies across 5994
rows, i.e. **1.27 rows per encode**. ToolACE is multi-tenant by construction. The caching
argument rests entirely on the separate probe finding that schema is byte-identical
across turns within one real deployment — not on anything measured here. The latency rows
above assume a warm cache because that is the deployed steady state the probe supports;
the cold-cache row shows what a miss costs.

## Reproduction

```bash
V=/Users/alex/develop/vllm-ltr-optimization/.worktrees/final-training-artifacts/.venv/bin/python
cd /Users/alex/develop/vllm-ltr-optimization/runs/offline-experiments-2026-07-25
$V e4_embed.py      # ~24 min, writes e4-embeddings.pt (gitignored)
$V e4_fusion.py     # ~15 s
$V e4_latency.py    # ~90 s
```

---

# T5 — evidence-based Reliability Gate confidence (2026-07-26)

Ticket: issue #9. Scripts `t5_score_validation.py`, `t5_gate.py`; artifacts
`t5-gate.json`, `t5-validation-scoring.json`. Compute: 241 s validation scoring + 3 s.
`/v1/decision` contract unchanged — only the value written to the existing
`Prediction.confidence` field changes.

## Design

Request-time signal, available at admission with no generation:

```
request -> tool-set fingerprint + tool names  ->  stratum (S1..S4)  ->  confidence
              vs the Ranker's training vocabulary
```

Cross-Workload Transfer is not handled here; it remains a Fallback trigger, as specified.

**Fit and evaluation are on different splits.** T1's per-stratum τ is measured on *test*.
Deriving confidence from those numbers and then validating on test would be circular —
the table would agree by construction and demonstrate nothing. So confidence is fit on
**validation** and evaluated on **test**. This required scoring the validation split with
the three prompt_schema checkpoints (241 s), which did not previously exist.

**Confidence rule**: `max(0, lower bound of the 95% session-clustered bootstrap CI of
validation τ)`. The lower bound rather than the point estimate, because a gate that
overstates its own reliability is precisely the defect being replaced.

## The stratum that breaks the obvious rule

| Stratum | val n | test n | realized test τ |
|---|---:|---:|---:|
| S1 seen-combination | 74 | 45 | 0.6012 |
| **S2 new combination, all tools seen** | 82 | 78 | **0.4392** |
| S3 partial-new tools | 472 | 543 | 0.6468 |
| S4 all-new tools | 370 | 333 | 0.6393 |

**S2 is by far the hardest stratum for the Ranker — not S4.** Requests whose tools were
all individually seen but whose *combination* is new score 0.44, roughly 0.20 below every
other stratum. Novel tools are handled well; novel *compositions* of familiar tools are
not. That is the opposite of the intuition the S1→S4 ordering suggests.

Caveat, stated plainly: S2's τ rests on 78 test rows and is formally withheld under the
n<100 rule. It is reported here because it is a **design input** — the gate cannot be
designed responsibly while ignoring the one stratum where it would fail. This is the same
rule-collision as T1's S1 tie bar and needs the same kind of ruling.

## Rule comparison — every candidate scored on test

`overstates_by = assigned − realized`; positive means the gate claimed more reliability
than it delivered, which is the failure mode that matters.

| Rule | S1 | S2 | S3 | S4 | never overstates | worst |
|---|---:|---:|---:|---:|:--:|---:|
| **placeholder 0.9** (ships today) | +0.299 | **+0.461** | +0.253 | +0.261 | no | **+0.461** |
| A: pooled small-stratum value (0.659) | +0.058 | **+0.220** | −0.068 | −0.016 | no | +0.220 |
| B: floor to lowest measured (0.579) | −0.023 | **+0.140** | −0.068 | −0.016 | no | +0.140 |
| global control, no stratification (0.630) | +0.028 | **+0.190** | −0.017 | −0.010 | no | +0.190 |
| **C: abstain on unmeasurable strata (0.0)** | −0.601 | −0.439 | −0.068 | −0.016 | **yes** | **−0.016** |

**Rule C is selected.** It is the only rule that never overstates on the held-out split.
S1 and S2 — the strata too small to estimate — receive confidence 0, which routes them to
the existing Fallback path rather than having the gate vouch for them.

Assigned confidences under Rule C: S1 0.0, S2 0.0, S3 0.5787, S4 0.6233.

## What actually does the work — and what does not

Two honest observations that should travel with this result:

1. **The hardcoded 0.9 is wrong in every stratum**, overstating realized ranking quality
   by +0.25 to +0.46. Replacing it with any measured value is a clear improvement. This
   is the strongest and least contestable part of the ticket.

2. **The graded part of stratification earns very little.** Among the two strata large
   enough to estimate, realized τ differs by only 0.0075 (0.6468 vs 0.6393), and the
   global no-stratification control (a single measured 0.630) is *also* conservative on
   S1, S3 and S4. Stratification's entire advantage over one global number is that it
   **abstains where it cannot measure** — and that matters only because S2 exists.

So the defensible claim is: *the gate's value comes from knowing where it does not know,
not from finely grading where it does.* Writing it up as "confidence tracks cold-start
stratum" would overstate what the numbers support — realized τ is flat across S1/S3/S4
and only S2 breaks ranks.

## Limits

- Confidence is a **measured ranking-quality floor, not a calibrated probability**. It is
  a τ lower bound in [0,1]; it is not P(correct). The `Prediction.confidence` docstring's
  "each predictor documents whether it is calibrated" obligation is met by saying so.
- Fit on 998 validation rows from one workload family (ToolACE tier-2). Nothing here
  licenses a claim about other workloads; that is Cross-Workload Transfer's job.
- No serving-benchmark claim is made, per the ticket.
- The reliability threshold that turns confidence into `prediction_reliable` is unchanged
  and not tuned here. Note that under Rule C, S1/S2 traffic will fail any positive
  threshold — that is intended, but it changes the fraction of requests taking the
  Fallback path and should be sized before deployment.

## Reproduction

```bash
V=/Users/alex/develop/vllm-ltr-optimization/.worktrees/final-training-artifacts/.venv/bin/python
cd /Users/alex/develop/vllm-ltr-optimization/runs/offline-experiments-2026-07-25
$V t5_score_validation.py   # ~4 min, writes t5-bert-validation-scores.jsonl (gitignored)
$V t5_gate.py               # ~3 s, writes t5-gate.json
```

---

# T6 — Rule C wired into the Decision Service (2026-07-26)

Ticket: issue #10. `scheduler_benchmark/predictor.py` no longer carries
`PLACEHOLDER_CONFIDENCE = 0.9`; `BertPredictor` now reports the confidence measured for
the request's Cold-Start stratum. `/v1/decision` contract unchanged — same fields, same
shapes, only the value in `confidence` differs.

## What changed

| Piece | Role |
|---|---|
| `scheduler_benchmark/tool_vocabulary.py` | schema parser, fingerprint, stratum rule, `GateVocabulary` |
| `scheduler_benchmark/artifacts/gate_confidence.json` | committed values + training vocabulary (345 KB) |
| `runs/offline-experiments-2026-07-25/build_gate_artifact.py` | derives that artifact from `t5-gate.json` |
| `scheduler_benchmark/predictor.py` | loads the artifact, classifies at request time |

Served values, read from the artifact rather than written in code: **S1 0.0, S2 0.0,
S3 0.5786788998431738, S4 0.6232874397453674**, and `unknown` 0.0 for requests whose tool
set cannot be read. `unknown` is computed as the minimum of the four, so it can never
exceed what any stratum earned even if the values are later refit.

The `LTR_CONSTANT_CONFIDENCE` env override is retained as an escape hatch (read once at
construction) for A/B runs and gate-disabled baselines.

## Two deliberate design choices

**The parser moved into the library.** Stratum classification needs the same
multi-template tool-name parser the offline experiments use. Duplicating it would let the
serving path and the evidence drift apart silently — the gate would then vouch using a
different definition of "seen" than the one it was measured under. It now lives in
`scheduler_benchmark/tool_vocabulary.py`. `build_gate_artifact.py` asserts the library
parser and the experiment parser agree on all **5994 rows** before writing the artifact,
and that guard passed.

**Unreadable tool sets abstain rather than default.** An empty tool list or a schema the
parser cannot read returns `unknown_confidence`, not a stratum guess. Under Rule C that is
0.0 either way, but the code does not rely on that coincidence.

## Tests

TDD: the first test asserted a novel-tools request must not report 0.9 and that
`PLACEHOLDER_CONFIDENCE` no longer exists. It went red as required (12 failures across the
new and existing predictor tests), then green after the implementation.

Coverage added: all four strata via a synthetic vocabulary fixture, empty tool list,
unparseable schema, blank schema text (still rejected upstream), the env escape hatch, and
a test that the committed artifact's values equal `t5-gate.json`'s — so a hand-edited
artifact fails CI.

```
tests/test_predictor.py            19 passed
tests/test_decision_service*.py    31 passed
full suite                        259 passed, 1 skipped, 1 failed
```

The single failure is `test_final_report_figures.py::test_latex_review_contracts_are_recorded`,
which asserts `tier2-learning-curve.json` appears in the LaTeX Evaluation section. It is
**pre-existing and unrelated** to this ticket — verified by running it in a clean worktree
at HEAD, where it fails identically. It belongs to the report track, not the serving path.

## Limits

- Confidence remains a measured τ lower bound, not a calibrated probability.
- The vocabulary is the ToolACE tier-2 training split. A deployment on other traffic would
  classify most requests S3/S4 by construction; that is correct behaviour for this
  artifact, but the values themselves are only evidenced for this workload family.
- No serving-benchmark claim is made. Under Rule C, S1+S2 traffic fails any positive
  reliability threshold and takes the Fallback path (~12% of the offline test split);
  Fallback load should be sized before deployment.
