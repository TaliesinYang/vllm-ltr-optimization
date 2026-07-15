# Server Training, Data Storage, and VeloxMesh Integration Design

Date: 2026-07-15  
Status: Pending user review  
Primary repository: `TaliesinYang/vllm-ltr-optimization`  
Execution location: rented single-GPU server only  

## 1. Decision and Relationship to the Existing Design

This document makes the approved workflow-aware optimizer design executable on a rented server. It adds the previously missing training-data contract, deterministic split policy, storage and retention policy, target-label generation, training matrix, service API, and VeloxMesh deployment path.

It supersedes only the execution order and the "VeloxMesh later" wording in `2026-07-14-workflow-aware-serving-optimizer-design.md`. The following earlier decisions remain unchanged:

- `vllm-ltr-optimization` is the primary implementation and evidence repository.
- Predictor, scheduler, and cache policies are independently ablatable.
- The served 7B/8B LLM is inference-only; only the small predictor is trained.
- Prediction must have an auditable fallback.
- Simulator values are not real serving latency.
- Multi-GPU routing and cluster scheduling are out of scope.

VeloxMesh is now the only public serving entry point. It forwards a request snapshot to the optimizer service, applies the returned scheduling/cache hints, and routes only to a tool-capable backend. Training logic and policy logic remain in this repository rather than being duplicated in Go.

## 2. Rental Boundary

No dataset download, model download, target generation, predictor training, or GPU benchmark runs on the local Mac. Local work is limited to source authoring and dependency-free fixture/unit tests.

The rental gate is:

- one NVIDIA GPU with at least 48 GB VRAM;
- at least 200 GB persistent storage;
- Docker with NVIDIA Container Toolkit and Compose v2;
- outbound HTTPS access to GitHub, Hugging Face, and the configured artifact backup endpoint;
- SSH access; and
- a persistent root selected through `GPP_ROOT`, defaulting to `/hy-tmp/gateway-policy-probe` on Hengyuan Cloud.

The server is not considered ready merely because CUDA is visible. `server_pipeline.sh preflight` must also verify free disk, container GPU access, pinned source revisions, and backup credentials before downloading or training.

## 3. Pinned Sources

Every source is downloaded by immutable revision. The downloader rejects a moving `main` revision.

| Role | Source | Immutable revision | License/use boundary |
|---|---|---|---|
| Primary tool-aware training | `Team-ACE/ToolACE` | `6bda777c88d21e5a204703c1ee45597a8fa4f734` | Apache-2.0; API/domain grouped split |
| Workflow/history/KV replay | `DiscoPosse/lmcache-agentic-traces` | `9e1de874521be873b2c92621049ecb836b536257` | CC-BY-4.0; session grouped split |
| External coding-agent test | `Inferact/codex_swebenchpro_traces` | `0d52ae8c75738117be9e58c7071bd9a5b43ff78f` | MIT; repository/trial grouped split |
| Cache-locality stress trace | `semianalysisai/cc-traces-weka-no-subagents-051226` | `0ae681ae27a0e3e716b344cb21f1b01bb1313d52` | Apache-2.0; no text-predictor training |
| Tool-call OOD test | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` | Evaluation only; never predictor training |
| Optional long-horizon OOD test | `hkust-nlp/Toolathlon-Trajectories` | `6194034105bc27fa438447172be0e7b4e35396e4` | CC-BY-4.0; evaluation only |
| Target served model | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Inference only; ungated model source |
| Predictor backbone | `google-bert/bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` | Four feature variants, three seeds |
| Gateway | `zardonc/VeloxMesh` | `fc20873bd0a822d6110abf18291c360a4daa12d5` | Existing four-patch stack plus optimizer adapter |
| Serving engine | `vllm-project/vllm` | `v0.24.0` commit `ee0da84ab9e04ac7610e28580af62c365e898389` | Pinned before the just-released 0.25 line; custom request/scheduler/KV overlay |

`APIGen-MT-5k`, xLAM-60k, tau-bench trajectories, LMSYS, and ToolBench are not part of the primary new-training mixture. This avoids non-commercial restrictions, benchmark contamination, repeated chat training, and unnecessary scope. Existing LMSYS-trained checkpoints remain frozen baselines.

## 4. Canonical Record Contract

Normalizers emit one logical LLM turn per record. Canonical records are stored as Parquet with JSON-encoded variable structures to avoid source-specific nested schemas.

Required fields:

```text
record_id                 SHA-256 stable identity
source                    canonical dataset name
source_revision           immutable source revision
source_native_id          original record or turn identity
group_id                  split-isolation identity
session_id                pseudonymous session identity when present
turn_index                zero-based turn number
prompt_text               current user/request text
messages_json             ordered prior/current OpenAI-style messages
tools_json                normalized OpenAI tool definitions
workflow_json             derived workflow features and missing-value masks
source_output_text        source response when redistributable
source_output_tokens      tokens under the pinned target tokenizer when available
pre_gap_ms                observed tool/user gap or null
prefix_tokens             target-tokenizer prefix length or null
cached_tokens             source-reported cached tokens or null
license_id                SPDX-style source license label
split                     train, validation, test, or cache_only
```

`record_id` is computed from `source_revision`, `source_native_id`, and `turn_index`. Source-native secrets and credentials are removed before canonical output. Decision logs never contain prompt text, tool arguments, or tool results.

## 5. Deterministic Split and Leakage Control

Split seed is `6806`. The split bucket is:

```text
bucket = uint32(sha256("6806:<source>:<group_id>")[0:8]) mod 100
train       = 0..79
validation  = 80..89
test        = 90..99
```

Group identities are source-specific:

- ToolACE: SHA-256 of the sorted normalized tool-name set, keeping the same API family in one split.
- LMCache agentic traces: session identifier.
- Inferact: repository identifier when present, otherwise trial identifier.
- SemiAnalysis: trace identifier and `cache_only` role.
- BFCL and Toolathlon: forced to `test` and excluded from training manifests.

Before splitting, the pipeline runs two contamination checks against BFCL:

1. exact SHA-256 over normalized prompt plus normalized tool schema; and
2. 128-permutation MinHash over lowercase character 5-grams, with Jaccard threshold `0.85`.

Any matching ToolACE record is excluded from training and written to `reports/bfcl_overlap_exclusions.parquet`. Split validation fails if any `group_id` appears in more than one split or any BFCL record appears in a training manifest.

## 6. Target-Model Label Generation

Source output lengths are retained for provenance but are not treated as target-backend truth. A pinned Qwen2.5-7B-Instruct server generates target labels once, before predictor training.

Generation configuration:

```text
model revision: a09a35458c702b33eeacc393d103063234e8bc28
vLLM: v0.24.0
tool parser: hermes
enable auto tool choice: true
temperature: 0.0
top_p: 1.0
seed: 42
max_model_len: 32768
max_tokens: 2048
```

The generator processes ToolACE, LMCache, Inferact, and BFCL records. Each result records `completion_tokens`, raw finish reason, tool-call parse validity, truncation status, latency, decoding configuration hash, model revision, and request checksum. Invalid or absent tool calls remain in the latency dataset with their actual output length and a quality flag; they are not silently discarded.

Inputs exceeding the model context preserve system/tool definitions and the most recent messages, drop the oldest history, and record the exact dropped-token count. Generation failures are retried twice, then written to a failure manifest. The pipeline refuses to train until every requested record is either labelled or explicitly present in that failure manifest.

## 7. Predictor Inputs and Fair Ablation

Four independently trained variants use identical records, labels, splits, hyperparameters, and seeds:

1. `prompt`: current prompt only;
2. `prompt_schema`: prompt plus tool definitions;
3. `prompt_schema_history`: previous messages/tool results plus the first two fields;
4. `prompt_schema_history_workflow`: all prior fields plus turn index, previous-tool count, prior gap, prefix size, cache-read indicator, and missing-value masks.

All variants use a 512-token BERT input. To avoid the old schema/truncation confound, segment budgets are fixed across variants:

```text
special tokens and separators: 8
current prompt: 192
tool schema: 160
recent history: 120
workflow serialization: 32
```

Absent segments are represented by a typed missing token; unused capacity is not reassigned to the prompt in the primary ablation. A separate diagnostic may allow dynamic capacity, but its result is not substituted for the primary table.

## 8. Training Objective and Matrix

Only the BERT predictor is trained. Qwen2.5-7B-Instruct remains frozen and serves only as label generator and inference backend.

Each predictor emits a scalar `predicted_log_tokens`. The deployable token estimate is `round(expm1(predicted_log_tokens))`, clamped to `[1, 2048]`.

Training loss:

```text
regression = SmoothL1(predicted_log_tokens, log1p(target_tokens), beta=0.5)
ranking    = softplus(-sign(y_i-y_j) * (p_i-p_j))
loss       = regression + 0.5 * ranking
```

Ranking pairs are formed within a batch only when relative target-length difference is at least `0.20`, matching the existing PARS filtering scale without claiming PARS as a new contribution.

Fixed hyperparameters:

```text
backbone: google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594
max_length: 512
epochs: 3
train_batch_size: 32
eval_batch_size: 64
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.10
mixed_precision: bf16 when supported, otherwise fp16
gradient_clip_norm: 1.0
seeds: 17, 42, 73
checkpoint selection: highest validation Kendall tau, tie-broken by lower validation MAE
```

The full matrix contains 12 runs: four feature variants times three seeds. Training writes one final selected checkpoint per run; intermediate optimizer checkpoints are deleted only after the selected checkpoint and resume state have been backed up.

The three seed models for a feature variant form an ensemble. Ensemble mean supplies `estimated_tokens`; normalized ensemble disagreement and validation embedding distance feed an isotonic calibrator for the probability that relative token error is at most `0.50`. The Gateway gate consumes this calibrated probability and the OOD score, never a training-set tau constant.

## 9. Evaluation Separation

Training, predictor evaluation, and live serving are separate pipeline stages.

Predictor evaluation reports:

- MAE and median absolute error in tokens;
- RMSE on `log1p(tokens)`;
- Kendall tau-b and Spearman correlation;
- sampled pairwise rank accuracy with a fixed pair manifest;
- calibration ECE and Brier score for the reliability event;
- tool-call parse-validity rate; and
- results by source, workload class, length bucket, and seen/unseen API group.

Gateway live evaluation loads frozen checkpoints and performs no online gradient updates. Online tool-latency quantiles and cache telemetry may update, but they are reset at the start of each repeated run. BFCL canonical-answer lengths remain a separately labelled proxy result; target-Qwen outputs are the real backend labels.

## 10. Persistent Storage Layout

The root is configurable and never assumed to live in the Git checkout:

```text
$GPP_ROOT/
  cache/
    huggingface/hub/
    huggingface/datasets/
    pip/
  source/
    vllm-ltr-optimization/
    VeloxMesh/
    vllm/
  data/
    raw/<source>/<revision>/
    canonical/v1/<source>/*.parquet
    splits/split-6806/{train,validation,test,cache_only}.parquet
    labels/qwen2.5-7b/<decoding_hash>/*.parquet
    manifests/
    reports/
  checkpoints/
    <feature_variant>/seed-<seed>/
  services/
    logs/
    decision-audit/
  runs/
    <run_id>/config/
    <run_id>/raw/
    <run_id>/metrics/
    <run_id>/figures/
    <run_id>/status/
  export/
    <run_id>/
```

Raw downloads are immutable. Canonical data is rebuilt into a new schema-version directory rather than modified in place. Labels are keyed by record ID and decoding hash, so changing a model or decoding option cannot overwrite earlier labels.

Every stage writes `manifest.json` containing source revisions, command, container image ID/digest, Python/package versions, row counts, exclusions, input/output SHA-256 values, start/end times, and exit status. JSON evidence copied into the course vault must use `git add -f` because the vault globally ignores `*.json`.

## 11. Retention, Backup, and Shutdown Gate

Minimum persistent disk is 200 GB. The preflight stops when free space is below 150 GB. Storage budget reserves approximately 50 GB for images/build caches, 30 GB for model/tokenizer assets, 40 GB for dataset/cache expansion, 20 GB for predictor checkpoints, 20 GB for run artifacts, and 40 GB safety margin.

Keep:

- immutable raw downloads and source-revision metadata;
- canonical and split manifests;
- all target labels and failure manifests;
- each selected checkpoint plus tokenizer/config/calibrator;
- raw live metrics, decision audit, and exact run configuration; and
- the final export checksums.

Delete only after successful backup verification:

- superseded intermediate checkpoints;
- transient container build layers; and
- temporary uncompressed conversion files.

After each expensive stage, create a `tar.zst` export and SHA-256 file, upload both to the configured OSS/object-storage path, download or remotely verify the checksum, and only then mark the stage complete. The server shutdown command is unreachable until `server_pipeline.sh check --require-backup` passes.

## 12. VeloxMesh and Service Contract

Deployment contains four services:

```text
client -> VeloxMesh -> optimizer-service -> VeloxMesh -> vLLM
                                      \-> audit/metrics
```

Only VeloxMesh binds to `0.0.0.0`. Optimizer and vLLM bind to the private Compose network.

VeloxMesh sends `POST /v1/decision` with:

```text
request_id, model, messages, tools, tool_choice,
conversation_id, workflow_id, arrival_time_ms,
previous_tool_gap_ms, queue_summary, cache_pressure
```

The optimizer service uses the same feature serializer as training and returns:

```text
estimated_tokens, confidence, ood_score, reliable,
policy_hint, fallback_reason, priority_score,
session_id, kv_ttl_ms, kv_priority, reuse_probability,
predictor_revision, feature_variant
```

VeloxMesh applies a 20 ms decision timeout. Timeout, malformed response, unavailable predictor, or unsupported backend hint triggers the documented category/aging fallback and emits an audit header. It never blocks a request indefinitely.

The modern vLLM overlay accepts `est_tokens`, `workflow_id`, `session_id`, `kv_ttl_ms`, and `kv_priority`. Scheduling consumes `est_tokens` only when `reliable=true`; cache hints are advisory and capability-checked. Prefix caching on/off is the first real baseline. TTL/idleness/reuse policies use the same backend hook and are reported as project implementations or literature-inspired proxies unless original paper code is actually executed.

## 13. Server Pipeline and Resume Semantics

One entry point exposes independent stages:

```text
server_pipeline.sh preflight
server_pipeline.sh bootstrap
server_pipeline.sh download
server_pipeline.sh normalize
server_pipeline.sh label
server_pipeline.sh train
server_pipeline.sh evaluate
server_pipeline.sh deploy
server_pipeline.sh benchmark
server_pipeline.sh check
server_pipeline.sh export
server_pipeline.sh all
```

`all` calls the same stage implementations in order. A stage may resume only when its inputs match the recorded manifest checksums. A changed config, revision, split seed, or decoding hash invalidates downstream completion markers instead of silently reusing stale results.

## 14. Ready-to-Rent and Completion Gates

`READY_TO_RENT.md` may say `READY` only when the repository contains:

- pinned dataset/model/backend registry;
- normalizers and schema validation;
- split and contamination checker;
- target-label generator and failure manifest checker;
- four-variant, three-seed training config;
- evaluator and calibration builder;
- optimizer service and contract tests;
- VeloxMesh adapter patch and tests;
- modern vLLM request/scheduler/cache overlay and checks;
- Compose/bootstrap/pipeline scripts;
- result and backup checkers; and
- commands that produce all expected artifact directories.

Server execution is complete only after:

- all requested records are labelled or explicitly failed;
- 12 selected predictor checkpoints exist with manifests;
- train/validation/test group disjointness passes;
- predictor and calibration reports pass schema checks;
- VeloxMesh routes tool requests through the optimizer to a real tool-capable vLLM backend;
- FCFS, pure prediction, safe fallback, gated, prefix-cache, and KV-policy ablations complete nonzero requests;
- raw and summarized results preserve free-preemption, relative-unit, sawtooth, proxy-label, and PARS-origin caveats; and
- the final export checksum is verified outside the ephemeral runtime.

## 15. Explicit Non-Claims

- ToolACE or source-model output lengths are not Qwen target labels until replayed.
- A schema-compatible request is not proof of successful tool calling.
- Prefix caching is not a learned agent-aware KV policy.
- A PBKV-, MORI-, KVFlow-, or Continuum-inspired policy is not a reproduction of that system.
- Simulator latency is not milliseconds until calibrated on the rented GPU.
- The current simulator's free preemption remains an optimistic limitation.
- The tool-workload sawtooth remains unstable.
- Existing PARS code is a port, not an algorithm introduced by this project.
- The previously reported gap shrink is 3.3%, not 15%.
