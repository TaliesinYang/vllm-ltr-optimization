# Server Training, Data Storage, and VeloxMesh Integration Design

Date: 2026-07-15  
Status: Approved after three-domain independent review
Primary repository: `TaliesinYang/vllm-ltr-optimization`  
Execution location: rented single-GPU server only  

## 1. Decision and Relationship to the Existing Design

This document makes the approved workflow-aware optimizer design executable on a rented server. It adds the previously missing training-data contract, deterministic split policy, storage and retention policy, target-label generation, training matrix, service API, and VeloxMesh deployment path.

It supersedes the earlier document's execution order, backend-integration order, VeloxMesh ownership wording, and unconditional KV/completion gates. Where the documents conflict on server data, vLLM v0.24, deployment, or readiness, this revision controls. The following earlier decisions remain unchanged:

- `vllm-ltr-optimization` is the primary implementation and evidence repository.
- Predictor, scheduler, and cache policies are independently ablatable.
- The served 7B/8B LLM is inference-only; only the small predictor is trained.
- Prediction must have an auditable fallback.
- Simulator values are not real serving latency.
- Multi-GPU routing and cluster scheduling are out of scope.

VeloxMesh is the only public serving entry point, but it is not the live scheduler. It owns authentication, tool-capability routing, request identity propagation, and one request-time prediction RPC. The optimizer service returns a `PredictionBundle`. A vLLM v0.24 adapter and custom scheduler, shipped from this repository and loaded with `--scheduler-cls`, own the live `SystemSnapshot` and execute scheduling/cache decisions inside the engine on every scheduling iteration. Queue and KV state never originate in VeloxMesh.

Training, prediction, scheduling, and cache-policy logic remain in this repository rather than being duplicated in Go. The old VeloxMesh category/tail-risk patches and the vLLM 0.4.1 `est_tokens` patch are legacy evidence only; they are not the v0.24 deployment stack.

## 2. Rental Boundary

No dataset download, model download, target generation, predictor training, or GPU benchmark runs on the local Mac. Local work is limited to source authoring and dependency-free fixture/unit tests.

The rental gate is:

- one NVIDIA GPU with at least 48 GB VRAM;
- at least 300 GiB persistent storage, with at least 250 GiB free before the first download;
- Docker with NVIDIA Container Toolkit and Compose v2;
- outbound HTTPS access to GitHub, Hugging Face, and the configured artifact backup endpoint;
- SSH access; and
- a persistent root selected through `GPP_ROOT`, defaulting to `/hy-tmp/gateway-policy-probe` on Hengyuan Cloud.

The 300/250 GiB values are conservative provisioning gates, not measured usage claims. After each pilot, preflight recomputes `required_free_gib = measured_stage_peak + largest_export_staging + safety_margin` and uses the larger of that value and the conservative gate. The server is not considered ready merely because CUDA is visible. `server_pipeline.sh preflight` must also verify the persistent mount and inodes, container GPU access, pinned source revisions and license locks, digest-pinned images, dependency locks, backup credentials, and secret handling before downloading or training.

The default gateway bind is loopback or a private/Tailscale address. `GATEWAY_BIND=0.0.0.0` is allowed for LAN review only when TLS, bearer authentication, host firewall rules, and an explicit LAN CIDR allowlist are enabled. Optimizer, vLLM, metrics, and tracing endpoints never publish a public port.

## 3. Pinned Sources

Every source is downloaded by immutable revision. The downloader rejects a moving `main` revision. A machine-readable source lock also pins dataset config, selected split/file globs, file SHA-256 or LFS OIDs, loader version, tokenizer revision, license-file SHA-256, allowed use, attribution path, redistribution flag, and export policy. The first verified snapshot freezes discovered row/group counts; later downloads must match that lock.

| Role | Source | Immutable revision | License/use boundary |
|---|---|---|---|
| Primary tool-aware training | `Team-ACE/ToolACE` | `6bda777c88d21e5a204703c1ee45597a8fa4f734` | Apache-2.0; API/domain grouped split |
| Workflow/history/KV replay | `DiscoPosse/lmcache-agentic-traces` | `9e1de874521be873b2c92621049ecb836b536257` | CC-BY-4.0; session grouped split |
| External coding-agent test | `Inferact/codex_swebenchpro_traces` | `0d52ae8c75738117be9e58c7071bd9a5b43ff78f` | MIT; external test only |
| Cache-locality stress trace | `semianalysisai/cc-traces-weka-no-subagents-051226` | `0ae681ae27a0e3e716b344cb21f1b01bb1313d52` | Apache-2.0; no text-predictor training |
| Tool-call OOD test | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` | Evaluation only; never predictor training |
| Optional long-horizon OOD test | `hkust-nlp/Toolathlon-Trajectories` | `6194034105bc27fa438447172be0e7b4e35396e4` | CC-BY-4.0; evaluation only |
| Target served model | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Inference only; ungated model source |
| Predictor backbone | `google-bert/bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` | Four feature variants, three seeds |
| Gateway | `zardonc/VeloxMesh` | `fc20873bd0a822d6110abf18291c360a4daa12d5` | Transport-only optimizer adapter; four-patch stack is legacy evidence |
| Serving engine | `vllm-project/vllm` | `v0.24.0` commit `ee0da84ab9e04ac7610e28580af62c365e898389` | Request adapter/custom scheduler; Phase-2 KV overlay is capability-gated |

`APIGen-MT-5k`, xLAM-60k, tau-bench trajectories, LMSYS, and ToolBench are not part of the primary new-training mixture. This avoids non-commercial restrictions, benchmark contamination, repeated chat training, and unnecessary scope. Existing LMSYS-trained checkpoints remain frozen baselines.

## 4. Canonical Record Contract

Normalizers emit one logical LLM turn per record. Canonical records are stored as Parquet with JSON-encoded variable structures to avoid source-specific nested schemas. `request_messages_json` ends immediately before the assistant output being labelled; the current source assistant output is never part of the predictor input.

Required fields:

```text
record_id                 SHA-256 logical source-turn identity
canonical_schema_version  canonical schema revision
normalizer_revision       Git/code revision of the source normalizer
source                    canonical dataset name
source_revision           immutable source revision
source_config             pinned dataset configuration
source_native_id          original record or turn identity
group_id                  split-isolation identity
dataset_role              predictor_fit, internal_eval, external_eval, or cache_replay
session_id                pseudonymous session identity when present
turn_index                zero-based turn number
prompt_text               current user/request text
request_messages_json     ordered request messages ending before target output
tools_json                normalized OpenAI tool definitions
workflow_json             derived workflow features and missing-value masks
generation_controls_json  max tokens, stop, temperature, top-p, format, tool choice
request_checksum          SHA-256 of the exact rendered target request
source_output_text        source response when redistributable
source_output_tokens      tokens under the pinned target tokenizer when available
pre_gap_ms                observed tool/user gap or null
prefix_tokens             target-tokenizer prefix length or null
cached_tokens             source-reported cached tokens or null
license_id                SPDX-style source license label
split                     train, validation_select, validation_calibration,
                          test, external_test, or null
```

`record_id = sha256(source, source_revision, source_config, source_native_id, turn_index)`. A label uses `label_id = sha256(record_id, request_checksum, model_revision, tokenizer_revision, serving_image_digest, decoding_hash)`, so a normalizer, request, model, tokenizer, image, or decoding change cannot reuse stale labels.

Canonicalization removes source-native secrets and credentials. Raw traces are secret-scanned, encrypted at rest, access-controlled, and excluded from public exports by default. Decision logs use an explicit allowlist and never contain request bodies, headers, prompt text, tool arguments, tool results, credentials, or cache salts.

## 5. Deterministic Split and Leakage Control

Split seed is `6806`. The split bucket is:

```text
key = utf8("6806:<component_id>")
bucket = int.from_bytes(sha256(key).digest()[0:4], "big") mod 100
train                  = 0..79
validation_select      = 80..84
validation_calibration = 85..89
test                   = 90..99
```

Group identities are source-specific:

- ToolACE: explicit normalized provider/domain/API-family identity augmented by connected components over tool-schema fingerprints; tool names alone are insufficient.
- LMCache agentic traces: session identifier.
- Inferact: repository plus benchmark task/issue identity, with trial nested beneath that group.
- SemiAnalysis: trace identifier with `dataset_role=cache_replay`; replay partition is independent of statistical split.
- Inferact, BFCL, and Toolathlon: `dataset_role=external_eval`, forced to `external_test`, and excluded from fitting, checkpoint selection, threshold decisions, and calibration.

Before split assignment, the pipeline builds connected components across all records—not only external evaluation—from repository/task/issue identifiers and two request-fingerprint checks:

1. exact SHA-256 over normalized prompt plus normalized tool schema; and
2. 128-permutation MinHash over lowercase character 5-grams, with Jaccard threshold `0.85`.

Identifier-, exact-, and near-duplicate-linked components are assigned atomically to one internal split or `external_test`. Any component intersecting external evaluation is excluded in full from predictor fitting and written to a source-specific overlap report. Split validation fails if any component appears in more than one split or any external-evaluation record appears in a fitting, selection, or calibration manifest.

The split checker emits per-source row/group proportions. A source is eligible for fitting only when it has at least 20 independent components; for eligible sources, each non-training internal split must contain at least one component and observed row share must be within five percentage points of its target. Otherwise the source is reclassified as evaluation-only rather than silently producing an unstable split.

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

The generator processes ToolACE, LMCache, Inferact, and BFCL records, while preserving their fit/evaluation roles. The experimental predictor contract uses the fixed decoding profile above. `tool_choice` and nullable stop/response-format values are still serialized into `generation_controls_json`; a live request that does not match the supported profile bypasses optimizer fields and uses `fallback_native` rather than receiving an unreliable prediction.

Each result records `label_id`, `completion_tokens`, raw finish reason, tool-call parse validity, censoring status, latency, decoding configuration hash, model/tokenizer/image revisions, and request checksum. Invalid or absent tool calls remain with their actual output length and a quality flag; they are not silently discarded. Generation-pass latency is provenance/diagnostic data, not a predictor target or live-serving measurement.

The pinned chat template is rendered before the context check. Inputs must satisfy `prompt_tokens <= 32768 - 2048`; the normalizer preserves system/tool definitions and the most recent messages, drops the oldest history, and records the exact dropped-token count. If system/tool content alone exceeds the input budget, the record receives a typed failure.

`finish_reason=length` at 2048 tokens is a right-censored label. It contributes `softplus(log1p(2048) - prediction)` instead of ordinary regression and is excluded from ordinary MAE/RMSE. Generation failures are retried twice and then written to a typed failure manifest. Training is blocked unless every record is accounted for, overall generation failure is at most 1%, per-source failure is at most 3%, and the failure report shows no concentration by group or length bucket above the same per-source limit.

Before full generation, a deterministic source/length-stratified pilot of up to 512 records per source measures label throughput, peak VRAM, and disk growth. A one-epoch BERT pilot measures training cost. The rental projection sums scaled full-label cost, all 12 primary runs, three required one-seed diagnostic runs, deploy/load testing, every planned repeated GPU benchmark, export staging, and retry margin. A checked-in rental-budget schema supplies maximum GPU-hours, storage, and cost; `server_pipeline.sh all` refuses to begin expensive stages when the total projection exceeds any operator-provided bound. Label generation and predictor training are mutually exclusive GPU stages.

## 7. Predictor Inputs and Fair Ablation

Four independently trained variants use identical records, labels, splits, common generation controls, sampling weights, hyperparameters, and seeds:

1. `prompt`: current prompt only;
2. `prompt_schema`: prompt plus tool definitions;
3. `prompt_schema_history`: previous messages/tool results plus the first two fields;
4. `prompt_schema_history_workflow`: all prior fields plus turn index, previous-tool count, prior gap, prefix size, and availability masks.

All variants use one deterministic serializer and a 512-token BERT input. To avoid the old schema/truncation confound, segment budgets and per-segment head/tail truncation rules are fixed across variants:

```text
special tokens and separators: 8
current prompt: 184
tool schema: 152
recent history: 112
workflow serialization: 32
generation controls: 24
```

Every variant receives the same availability indicators. Ablated values are replaced by typed mask tokens without exposing variant-specific source identity, and unused capacity is not reassigned in the primary ablation. Primary reporting includes the full paired set and a complete-case paired subset. The diagnostic matrix has three variants—masks-only, permuted-workflow, and dynamic-capacity prompt-only—each run with the declared exploratory seed `42`. Their manifests and reports are required but are not counted among the 12 primary checkpoints. Cache-read status is excluded unless a target-backend cache probe proves it is observable before scheduling.

## 8. Training Objective and Matrix

Only the BERT predictor is trained. Qwen2.5-7B-Instruct remains frozen and serves only as label generator and inference backend.

Each predictor emits a scalar `predicted_log_tokens`. The deployable token estimate is `round(expm1(predicted_log_tokens))`, clamped to `[1, 2048]`.

Training loss:

```text
point           = SmoothL1(predicted_log_tokens, log1p(target_tokens), beta=0.5)
censored        = softplus(log1p(2048) - predicted_log_tokens)
regression_or_censored = mean(where(is_censored, censored, point))
eligible(i,j) = abs(y_i-y_j) / max(y_i,y_j,1) >= 0.20
ranking       = mean(softplus(-sign(y_i-y_j) * (p_i-p_j)))
loss          = regression_or_censored + 0.5 * ranking
```

Each unordered eligible pair appears at most once; zero eligible pairs produce zero ranking loss. The pair-sampling RNG and maximum pairs per batch are pinned in the run config. A fixed source-balanced and group-balanced sampler prevents long sessions or large sources from dominating. The `0.20` filter matches the existing PARS filtering scale without claiming PARS as a new contribution.

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
checkpoint selection: highest uncensored validation_select Kendall tau,
                      tie-broken by lower uncensored validation_select MAE
```

The full matrix contains 12 runs: four feature variants times three seeds. Training writes one final selected checkpoint per run; intermediate optimizer checkpoints are deleted only after the selected checkpoint and resume state have been backed up.

The three seed models for a feature variant form an offline ensemble. Convert each seed from log space to a clamped token estimate first; their token-space arithmetic mean supplies the offline ensemble estimate, and `std(token_estimates) / max(mean(token_estimates), 1)` is normalized disagreement.

OOD distance uses L2-normalized penultimate-layer CLS embeddings and cosine distance to the nearest of `min(32, number_of_training_groups)` k-means centroids fitted on training-group embeddings. Median/IQR normalization is fitted on `validation_calibration` only. Five-fold group cross-fitting on `validation_calibration` fits a logistic raw reliability score, then a one-dimensional isotonic map from that score to `P(abs(pred-target)/max(target,1) <= 0.50)`. Each seed's deployable calibrator uses OOD distance, predicted log length, and input truncation fraction; the offline-ensemble calibrator adds normalized ensemble disagreement. Final logistic models are refitted on all calibration groups; censored and failed labels are excluded. `prediction_reliable` requires calibrated probability at least `0.80` and OOD distance no larger than the training 99th percentile.

In live single-GPU deployment, vLLM owns the GPU and the optimizer uses the single selected calibrated seed on CPU unless a co-serving pilot proves the three-model ensemble meets the decision-latency gate without harming vLLM. The three seeds always remain independent-replication and offline-ensemble evidence.

## 9. Evaluation Separation

Training, predictor evaluation, and live serving are separate pipeline stages.

Predictor evaluation reports per-seed and offline-ensemble results separately:

- MAE and median absolute error in tokens;
- RMSE on `log1p(tokens)`;
- Kendall tau-b and Spearman correlation;
- sampled pairwise rank accuracy with a fixed pair manifest;
- calibration ECE and Brier score for the reliability event;
- reliability coverage, false-reliable rate, selective MAE/tau, and risk-coverage AUC;
- label coverage, failure, censoring, and context-drop rates;
- source-macro and pooled results by source, workload class, length bucket, and seen/unseen API group; and
- group-bootstrap 95% confidence intervals plus per-seed mean and standard deviation.

Pairwise accuracy uses one pinned manifest stratified by source and length-gap bucket. Tool-call parse validity belongs to the label-quality report; predictor metrics are stratified by parse-valid/invalid records instead of treating validity as predictor accuracy.

Standard Kendall tau, Spearman, pairwise accuracy, MAE, median error, and RMSE exclude right-censored rows. Censored rows receive a censor-aware concordance metric and one-sided violation rate. These are reported separately and do not replace uncensored checkpoint-selection metrics.

Tool latency is an online quantile estimator, while next-step and reuse probability use a transition model rather than extra BERT heads. Transition/reuse statistics fit on training groups only, thresholds tune on calibration groups only, and evaluation starts with empty online state. Their configs define targets, smoothing, minimum-support fallback, and held-out-session metrics.

Gateway live evaluation loads frozen checkpoints and performs no online gradient updates. Online tool-latency quantiles and cache telemetry may update, but they reset at the start of each repeated run. BFCL canonical-answer lengths remain a separately labelled proxy result; target-Qwen outputs are the real backend labels. Each live ablation uses a pinned matched-request manifest, warmup, at least three repeated runs at declared offered loads, at least 99% completed requests, at most 1% errors, and reports TTFT, TPOT, workflow JCT, throughput, p95/p99, optimizer overhead, and confidence intervals.

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
    canonical/<canonical_fingerprint>/<source>/*.parquet
    splits/<split_fingerprint>/{train,validation_select,validation_calibration,test,external_test}.parquet
    replay/<replay_fingerprint>/*.parquet
    labels/<label_fingerprint>/*.parquet
    manifests/
    reports/
  checkpoints/
    <training_fingerprint>/<feature_variant>/seed-<seed>/
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

Raw downloads and all fingerprint directories are immutable. A changed normalizer, split, request, model, decoding profile, or training config creates a new lineage rather than modifying earlier evidence. Reports and service audits are namespaced by run and stage fingerprint.

Every stage writes a versioned manifest containing source revisions, redacted command, clean primary/VeloxMesh/vLLM repository commits, normalizer code hash, dependency-lock digest, container image digest, Python/package versions, canonical config, ordered input/output SHA-256 values, row counts, exclusions, start/end times, and exit status.

`stage_fingerprint = sha256(manifest_schema_version || ordered_repository_commits || container_digest || dependency_lock_digest || canonical_stage_config || ordered_input_artifact_digests)`. GPU stages require clean primary, VeloxMesh, and vLLM worktrees; preflight fails on tracked modifications or untracked source/config files. Dirty GPU execution has no override. The explicit DAG is `download -> normalize -> split -> pilot-label -> label -> train -> calibrate -> evaluate -> deploy-smoke -> benchmark -> export`. Label shards and each primary or diagnostic training run have independent manifests. A stage writes to a temporary fingerprint directory, validates outputs, atomically renames it, and only then creates a completion marker. Mismatches create new lineages and invalidate descendants without deleting old outputs.

Course-vault evidence is a schema-validated, redacted summary under one narrow allowlisted path. Do not blanket force-add generated JSON; if the vault's global `*.json` rule applies, use `git add -f` only for the exact reviewed evidence file.

## 11. Retention, Backup, and Shutdown Gate

The rental provision is at least 300 GiB with 250 GiB initially free. The previous approximate components—50 GiB images/build caches, 30 GiB model/tokenizer assets, 40 GiB dataset/cache expansion, 20 GiB predictor checkpoints, 20 GiB run artifacts, and 40 GiB safety—are planning estimates, not measured facts, and omit export staging. Pilot sizing records measured peaks. Preflight uses `max(250 GiB, measured_stage_peak + largest_export_staging + safety_margin)` until audited measurements justify a lower gate.

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

After each expensive stage, create a deterministic `tar.zst`, signed inventory, and SHA-256 file. The inventory records path, byte size, digest, producing fingerprint, license/export decision, and required/optional status. Upload them to the configured object-store path, then read the object back or use a provider SHA-256 API. Upload exit status and ETag alone are insufficient.

A verified backup receipt is bound to run ID, local SHA-256, remote URI, object version ID, byte size, and verification time. `server_pipeline.sh verify-export` restores into an empty directory, validates inventory/manifests, and performs a CPU-only report and checkpoint-loading smoke test. Auto-shutdown is disabled by default; it requires explicit `--shutdown`, a valid receipt, and `server_pipeline.sh check --require-backup`. A mismatch exits nonzero without shutdown. The old `WORKFLOW.md` and `hy_*` upload/shutdown scripts are marked legacy baseline-only and are removed from the default start path.

Public exports are deny-by-default for raw datasets, request text, traces, model weights, Hugging Face caches, secrets, and provider credentials. They contain pinned download recipes, source locks, redacted manifests, split IDs/checksums, redistributable checkpoints when their license permits, metrics, and `THIRD_PARTY_LICENSES`/`NOTICE`. Public export is blocked until the project-owned code has a reviewed top-level `LICENSE` or explicit all-rights-reserved notice and ownership statement. Unknown or incompatible license metadata fails preflight. Private backups remain encrypted and access-controlled under each source's license.

Preflight resolves `realpath(GPP_ROOT)`, rejects a root inside any Git worktree, verifies the expected persistent mount/device, free blocks and inodes, creates directories with `umask 077`, and validates non-root container UID/GID read-write access. Secrets enter containers only through Compose secrets or read-only files and are redacted from commands, environment captures, logs, and manifests.

## 12. VeloxMesh, Prediction, and Engine Contract

Deployment ownership is:

```text
client -> VeloxMesh -> vLLM
             |-> optimizer-service (request-time prediction only)
vLLM custom scheduler -> in-process workflow_optimizer policy
all components -> private audit/metrics sinks
```

The project-owned Compose file has mutually exclusive `prepare`, `train`, and `deploy` profiles. During `deploy`, vLLM exclusively owns the GPU and optimizer-service is CPU-only unless a measured reservation proves safe. Deploy services are `gateway`, `optimizer`, `vllm`, and optional internal observability services. Checkpoints/config are read-only; audit/run volumes are writable. Gateway readiness requires optimizer warmup, vLLM health, and a real two-turn tool-call probe.

### 12.1 Prediction API

VeloxMesh sends `POST /v1/decision`. Required and optional fields are schema-versioned, typed, and bounded:

```text
schema_version:       literal "1.0"
request_id:           required UUID/string, 1..128 bytes
decision_id:          required UUID/string, 1..128 bytes; generated by VeloxMesh
model_id:             required string, 1..256 bytes
workflow_id:          optional string, <=128 bytes
step_id:              optional string, <=128 bytes
conversation_id:      optional string, <=128 bytes
request_age_ms:       required monotonic duration, integer >=0
messages:             required OpenAI message array
tools:                optional OpenAI tool array
tool_choice:          optional OpenAI tool-choice value
generation_controls:  required supported decoding profile
previous_tool_gap_ms: optional integer >=0
```

The body limit defaults to 2 MiB and is configurable downward. `queue_summary` and `cache_pressure` are absent because only the engine owns those values. The optimizer uses the exact training serializer and returns a `PredictionBundle`:

```text
schema_version:          literal "1.0"
decision_id:             required UUID/string
estimated_tokens:        optional integer in [1, 2048]
reliability_probability: required float in [0, 1]
ood_score:               required finite float >=0
prediction_reliable:     required boolean
predictor_revision:      required string
feature_variant:         prompt | prompt_schema | prompt_schema_history |
                         prompt_schema_history_workflow
reason_code:             prediction_reliable | low_reliability |
                         ood_rejected | missing_optional_features
reuse_probability:       optional float in [0, 1]
```

VeloxMesh generates `decision_id` before the RPC and the optimizer echoes it unchanged. Reason precedence is `ood_rejected`, `low_reliability`, `missing_optional_features`, then `prediction_reliable`. A non-200 response uses `{schema_version, error_code, retryable}` with `error_code` in `{invalid_schema, body_too_large, invalid_request, unsupported_controls, rate_limited, not_ready, internal_error}`. HTTP 200 covers reliable and model-produced unreliable predictions. HTTP 400/413/422/429/503 is never retried on the request path and maps to `fallback_native` while retaining the error code and preallocated decision ID.

`DECISION_TIMEOUT_MS` is configurable. The request path has no retries, uses pooled connections, bounded concurrency, cancellation propagation, and a circuit breaker. Timeout, unavailable service, unsupported decoding controls, or malformed response omits all optimizer metadata and selects `fallback_native`; unsupported individual fields are dropped as `capability_fallback`. No Go category/tail-risk/aging policy is executed as a second owner.

Readiness requires loaded checkpoints/calibrator and warmup. A CPU-side load test must show warmed p99 decision latency below 80% of the configured timeout at target concurrency and report timeout/fallback rate. Otherwise the optimizer remains not ready.

### 12.2 vLLM Transport and Scheduler

VeloxMesh transports reliable prediction metadata through namespaced `vllm_xargs` keys such as `workflow_estimated_tokens`, `prediction_reliable`, `workflow_id`, `step_id`, and `decision_id`. When `prediction_reliable=false`, it omits `workflow_estimated_tokens`. An explicit v0.24 adapter copies validated namespaced values into typed internal request metadata; stock vLLM accepting or ignoring an unknown JSON field is not evidence that the policy works.

The pinned custom scheduler, loaded with `--scheduler-cls`, constructs `SystemSnapshot` from live waiting/running requests and KV telemetry on every `schedule()` call, executes the repository-owned policy, and records proposed versus applied ordering. A minimal native comparison may map a clamped signed `int64 priority` to vLLM's top-level `priority` and start with `--scheduling-policy priority`, where lower values run earlier. SJF/joint-policy claims require concurrency tests that demonstrate changed queue order; HTTP 200 is insufficient. The vLLM 0.4.1 `est_tokens` patch remains legacy evidence only.

### 12.3 KV Capability Phases

Phase 1 supports only `prefix_cache={off,on}` and reports real prefix-hit telemetry. It makes no claim of TTL, priority, preserve/swap, prefetch, or learned eviction.

Phase 2 is enabled only after the v0.24 overlay implements typed workflow metadata, block-level metadata, shared-prefix merge rules, expiry-on-lookup, an eviction comparator, live `ref_cnt` safety, capability bits, events, metrics, and GPU tests. Expiry may remove future hash hits but must never free a live referenced block. Until this hook passes those gates, `kv_ttl_ms`, `kv_priority`, reuse-guided eviction, preserve/swap/prefetch actions, and their ablations are `UNSUPPORTED_NOT_RUN` and are not completion requirements. Implemented policies are reported as project implementations or literature-inspired proxies unless original paper code is executed.

Workflow/session identity is never reused as `cache_salt`. `cache_salt` is a separate unpredictable tenant-isolation secret, is capability-checked, and is never written to audit logs.

### 12.4 Tool-Call and Audit Proof

The VeloxMesh adapter preserves `role=tool`, `tool_call_id`, assistant `tool_calls`, `tools`, and `tool_choice` in streaming and non-streaming paths. Capability is configuration/probe-derived rather than hard-coded. vLLM starts with `--enable-auto-tool-choice --tool-call-parser hermes`. Completion requires a two-turn E2E test: assistant tool call, tool-result message with matching `tool_call_id`, and final assistant response with parsed calls asserted.

Every request emits one redacted decision record keyed by `request_id`, preallocated `decision_id`, and `trace_id`, including timeout, circuit-open, malformed-response, and other fallback paths: contract/predictor/policy/backend revisions, RPC latency, timeout/circuit state, confidence/OOD, proposed fields, applied/dropped fields with reason, fallback source, vLLM ordering outcome, prefix hit, supported KV action/outcome, TTFT, TPOT, and workflow JCT. Response headers are debugging aids only. Prometheus counters/histograms cover decision status, latency, fallback, and capability drops; trace context propagates across VeloxMesh, optimizer, and vLLM.

## 13. Server Pipeline and Resume Semantics

The implementation defines exact project paths: `docker/Dockerfile.*`, `deploy/compose.yaml`, digest-pinned image and hashed dependency locks, `configs/sources.lock.yaml`, and `scripts/server_pipeline.sh`. Base and service images are pinned by digest; Python dependencies are hash-locked. The environment lock records supported Linux architecture, CUDA, PyTorch, minimum NVIDIA driver, and compute capability. Compose declares GPU reservations, health checks, private networks, non-root UID/GID, Compose secrets, and explicit `$GPP_ROOT` bind mounts.

One entry point exposes independent stages:

```text
server_pipeline.sh preflight
server_pipeline.sh bootstrap
server_pipeline.sh download
server_pipeline.sh normalize
server_pipeline.sh split
server_pipeline.sh pilot-label
server_pipeline.sh label
server_pipeline.sh train
server_pipeline.sh calibrate
server_pipeline.sh evaluate
server_pipeline.sh deploy
server_pipeline.sh benchmark
server_pipeline.sh check
server_pipeline.sh export
server_pipeline.sh verify-export
server_pipeline.sh all
```

`preflight` runs `docker compose config`, digest pull/build verification, a `docker run --gpus all` smoke, mount/permission checks, source/license-lock validation, backup-credential checks, and secret scanning before downloads. `all` calls the same stage implementations in DAG order. Resume occurs only when the complete stage fingerprint matches; a changed repository commit, lock, image, config, revision, split, request, or decoding hash creates a new lineage rather than silently reusing stale results.

This design and the new server pipeline govern the VeloxMesh/vLLM v0.24 experiment. Top-level README/start links must point here. Existing `WORKFLOW.md`, `hy_*`, old Llama/LMSYS setup, and CUDA 12.1 vLLM-LTR fork instructions are explicitly `LEGACY FCFS/LTR REPRODUCTION` and cannot be invoked by the new default path.

## 14. Ready-to-Rent and Completion Gates

Readiness has three separate attestations and none implies another:

1. `REPOSITORY_READY`: source commit passes local/static/fixture checks; no GPU execution claim.
2. `SERVER_PREFLIGHT_PASSED`: rented host, persistent mount, containers, GPU visibility, locks, secrets, and backup target pass preflight.
3. `GPU_EXECUTION_COMPLETE`: full labels, training, deployment, benchmarks, export, and external checksum verification pass.

`READY_TO_RENT.md` may say `REPOSITORY_READY` only for an exact Git commit after an automated checker verifies executable pipeline stages, passing unit/contract tests, `docker compose config`, digest-resolvable images, fixture generation of every expected artifact directory, backup-checker dry run, and secret scan. It must also state `GPU_EXECUTION_NOT_RUN` and `SERVER_PREFLIGHT_PENDING`.

Repository readiness requires verified, not merely present, components:

- a reviewed top-level project `LICENSE` or explicit all-rights-reserved notice and copyright-ownership statement, kept separate from third-party notices;
- pinned dataset/model/backend registry;
- normalizers and schema validation;
- split and contamination checker;
- target-label generator and failure manifest checker;
- four-variant, three-seed training config;
- three seed-42 diagnostic configs plus required manifests/reports;
- evaluator and calibration builder;
- optimizer service and contract tests;
- VeloxMesh adapter patch and tests;
- modern vLLM request adapter/custom scheduler and checks;
- Phase-1 prefix-cache capability/telemetry checks;
- Compose/bootstrap/pipeline scripts;
- result and backup checkers; and
- commands that produce all expected artifact directories.

Server execution is complete only after:

- all requested records are labelled or typed-failed within the declared overall/per-source thresholds;
- 12 selected predictor checkpoints exist with manifests;
- the three seed-42 diagnostic runs have manifests and reports;
- train/selection/calibration/test/external-test component disjointness passes;
- predictor and calibration reports pass schema checks;
- VeloxMesh preserves a complete two-turn tool-call loop through the optimizer to a real tool-capable vLLM backend;
- FCFS, pure prediction, safe fallback, gated, and prefix-cache ablations meet the matched-manifest, repetition, completion, and error-rate gates;
- Phase-2 KV-policy ablations run only if the capability overlay and GPU safety tests pass; otherwise they remain explicitly `UNSUPPORTED_NOT_RUN`;
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
