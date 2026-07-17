# [SUPERSEDED 2026-07-17 → see 2026-07-17-final-training-plan.md] Overnight Training + Gateway Integration Plan (2026-07-16)

Owner: Alex (Dazhi Yang). Executors: Codex (engineering), Alex (owner decisions, rental, sign-offs), Claude (review, report drafting).

Deadlines this plan serves: **Final Report 2026-07-22**, Presentation 2026-07-29.

## Goal

Two components, both deliverables — neither is cut:

- **A. Tool-call length predictor**: BERT-base trained on toolace+lmcache Qwen labels,
  4-variant ablation (prompt / +schema / +history / +workflow), evaluated on
  bfcl / inferact / toolathlon (unseen agentic distributions). Produces the tau table.
- **B. Gated serving path (the novelty)**: optimizer `/v1/decision` service returning
  `PredictionBundle` with `prediction_reliable` gating; VeloxMesh transports it;
  fallback when unreliable. Per spec
  `docs/superpowers/specs/2026-07-15-server-training-data-storage-veloxmesh-design.md` §12.

A feeds B. The report claims: length-aware scheduling helps (midterm evidence),
chat predictors fail on tool-call traffic (midterm evidence), so we (1) train an
agentic predictor and study which context segments matter (A), and (2) gate the
scheduler on prediction confidence so it degrades safely (B).

## Definition of "results tomorrow" (honest)

Overnight realistic: snapshots frozen + pilot passed + full labeling done +
**first checkpoints and first tau numbers** (subset of the 15-run matrix).
Full 12+3 matrix + calibration + full evaluation: tomorrow evening, resumable
(ledger + per-run manifests). Do NOT promise the full matrix by morning.

## Decisions (owner) — status

| # | Decision | Status |
|---|---|---|
| D1 | Training gate: labeling/training needs only toolace+lmcache (the only label-eligible sources by design). The other 4 loaders are needed only at the `evaluate` stage. **Proposal: tonight land toolace+lmcache loaders → start labeling/training; the 4 external loaders land tomorrow daytime, before evaluate. All six still ship — this changes order, not scope.** | ⬜ Alex sign-off needed |
| D2 | Authority-hardening WIP (2 red tamper tests): pause on a side branch; `object.__setattr__` tampering declared out of threat model (single owner, single rented GPU). | ⬜ Alex sign-off needed |
| D3 | Qwen backend: local vLLM Qwen2.5-7B-Instruct (pinned rev) on the same rented 48GB GPU. | ⬜ Alex sign-off needed |
| D4 | Pilot gate thresholds: repo defaults (overall failure ≤1%, per-stratum ≤3%). | ⬜ Alex sign-off needed |
| D5 | Rows without authoritative parse_valid: excluded from classification metrics, kept for length/tau. | ⬜ Alex sign-off needed |
| D6 | Data policy: private exact snapshots; commit hashes/manifests only. | ⬜ Alex sign-off needed |
| D7 | Budgets: Alex fills `configs/bootstrap-budget.json` + `configs/rental-budget.json` with real provider rates (恒源云 4090 48G ~¥5/h) tonight. No invented values. | ⬜ Alex fills tonight |

## Phase 0 — Tonight, local (Codex; target ≤6h)

Order matters: artifact-producing work first.

1. Park authority WIP per D2 (side branch, 15 min). No new locks/gates/contracts
   anywhere in this plan. Existing fail-closed stops are answered, not extended.
2. `toolace` production loader + frozen-snapshot tests (descriptor-bound, pinned
   revision `6bda777c…`).
3. `lmcache` production loader + frozen-snapshot tests (pinned `9e1de874…`).
4. Compose pilot label runtime: wire `StagePlan` + live `LabelStageBudgetAuthorization`
   through `write_pilot_label_artifacts` / `write_label_artifacts` (the handoff's
   "Proposed next local implementation slice"), vLLM OpenAI-compatible loopback backend.
5. If D1 approved: scope the exact-six normalize gate so `normalize/split/pilot-label/
   label/train` run on the 2 label-eligible sources while `evaluate` still requires
   all six audited. Smallest honest change; keep fail-closed semantics.
6. Gate: `pytest` green + `ruff` + a dry-run of `download→normalize→split` on fixtures.

Exit criteria: one command sequence Alex can paste on the server.

## Phase 1 — Tonight, rented server (Alex rents; ~1h setup + overnight GPU)

Hardware: 恒源云 RTX 4090 48GB, CUDA 12.x image, ≥100G data disk
(WORKFLOW.md Phase 0 gates apply).

1. Clone repo @ tonight's commit; generate `requirements/data-predictor.lock` +
   `configs/environment.lock.json` on-server (Python 3.11 + CUDA per plan).
2. Download + freeze toolace/lmcache snapshots (pinned revisions); source-lock
   audit → Alex approves the audit report digest (phone/laptop, ~10 min).
3. `normalize → split` (leakage-safe, MinHash, seed 6806).
4. vLLM serve Qwen2.5-7B-Instruct (hermes parser, greedy, max_tokens 2048); smoke.
5. `pilot-label` (≤512 rows/stratum): record real throughput → real ETA for full run.
   Alex approves full-run projection (D4 thresholds).
6. `label` full run overnight (resumable Parquet ledger).
7. If labeling finishes early: start training queue — priority order
   `prompt_schema_history_workflow` seed 42 → `prompt` seed 42 → remaining runs.
   Checkpoints + manifests persist to OSS before any instance stop.

Overnight artifacts to persist (never lose to ephemeral disk): snapshots' hashes,
label ledger, all checkpoints, `usage_config`/manifests, logs.

## Phase 2 — Tomorrow daytime (parallel tracks)

- **GPU track**: finish the 12+3 matrix (resume queue) → `calibrate → evaluate`
  (needs the 4 external loaders by evaluate time) → tau table + CIs.
- **Codex track (CPU, parallel)**: `bfcl`, `inferact`, `toolathlon`, `semianalysis`
  loaders + audits (bfcl first — cleanest schema, needed for the headline
  "unseen agentic distribution" columns).
- **Claude track**: report skeleton from midterm evidence (see Phase 4).

## Phase 3 — Tomorrow evening: minimal gateway integration (B)

Scope = smallest honest demo per spec §12, NOT the full optimizer:

1. CPU predictor service loads one calibrated seed (Task 11 scaffold exists).
2. `/v1/decision` endpoint: echo `decision_id`, return `PredictionBundle` with
   `prediction_reliable` + reason precedence (`ood_rejected` → `low_reliability` →
   `missing_optional_features` → `prediction_reliable`).
3. VeloxMesh (pinned `fc20873`) transport smoke on LAN: two-turn tool-call E2E —
   assistant tool call → tool result (matching `tool_call_id`) → final response;
   reliable prediction transported via namespaced `vllm_xargs`; unreliable →
   `fallback_native`, field omitted.
4. Record one redacted decision log per request — this is report evidence
   ("the gate demonstrably works"), not a latency benchmark. Latency benchmarking
   of the gated path is post-report (IPDPS line), state so honestly.

Coordinate with Mingye: we need his gateway running + 30 min of joint smoke time.

## Phase 4 — Report (deadlines updated per professor's 7/15 post)

Deadlines: **GitHub repo URL due 7/22** (repo Public or add collaborator
`anithasaravanaedu-spec`; must click Submit). **Final Report due 7/27 8:59 AM**
(moved from 7/22). Source: `_inbox/FinalReport_Summer.docx`.

Hard format (binary rubric, 10 pts/section, no partial credit):
- IEEE 2-column LaTeX + BibTeX (IEEE style); title starts "Group XX: ...".
- Exact page counts, each section starts a new page: Abstract ¼pg (2 paragraphs,
  1-2 quantitative numbers, no citations) · Intro 1pg (contributions as bullets —
  the ONLY section allowed bullets) · Background 1.5pg ≥2 figs · Related Work 1pg
  (5-yr SOTA, subsections by approach, no figs) · Methodology 1pg ≥2 tables no figs
  (reproducibility: hw/OS/lang/libs, cite tools+benchmarks) · **Evaluation 3pg ≥6
  matplotlib figures** · Discussion 1pg (limitations + next-6-months) · Conclusion
  1 paragraph · Appendix A: screenshot of `latex_source/` dir on GitHub.
- Repo must contain `latex_source/` (full LaTeX) and `scripts/` (all plotting
  scripts, documented, reproduce every figure). No pseudocode, no flowcharts —
  architecture-style diagrams only. Figure fonts ≥10pt.
- **Generative-AI writing = 0 for the whole report.** Division of labor: AI may
  run experiments, generate data, write plotting scripts, organize evidence;
  ALL report prose is written by Alex.

Evidence base: midterm measured results (FCFS vs LTR 2.86×, 5-predictor tau,
generalization gap 0.559→0.315, honest PARS ablation) + whatever A/B produce by
7/25. Evaluation needs ≥6 figures: 4 midterm figures exist (`figures/`,
matplotlib, scripts present) + agentic tau/ablation figures from A + gated
decision-log figure from B.

Timeline: repo cleanup + collaborator access by 7/21, submit URL 7/22 ·
LaTeX skeleton + figure set 7/20 · Alex writes prose 7/22-25 · rubric
self-check against every binary item 7/26 · submit 7/27 before 8:59 AM.

## Risks / stops

- Pilot throughput unknown → full-label ETA unknown until Phase 1 step 5. If full
  labeling won't finish overnight, ledger resumes next night; training starts on
  whatever labeled shards are complete only if split integrity allows — otherwise wait.
- 恒源云 GitHub/HF access needs `/etc/network_turbo`; snapshots can stage via OSS.
- If D1 is rejected (strict all-six before any training), add the 4 external loaders
  to Phase 0 and expect training to slip one day. Alex's call.
- Any test failure on the critical path: fix forward, no new contract layers.
