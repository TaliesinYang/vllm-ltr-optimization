# Rental-day runbook v2 — 2026-07-26 (E5)

v2 incorporates the external Pro-tier design audit (grounded in this repo; archived at
`~/.claude/web-ask/archive/20260726-0230-rental-design-audit.md`). Purpose ruling: **(c)**
— core matrix feeds the Final Report (due 07-27 08:59); extensions feed the 07-29
presentation and beyond.

## 0. Hard rules (violating any one invalidates the day)

1. **`--reliability-threshold 0.5` exactly.** Never 0.5787 (S3's true value is 0.57867…;
   0.5787 marks all S3 unreliable). Preflight asserts BEHAVIOR, not HTTP 200:
   S2 → `prediction_reliable=false`, no `estimated_tokens`; S3 & S4 → reliable, with
   `estimated_tokens`.
2. Teardown between vLLM restarts: `pkill -9 -f "VLLM::EngineCore"` + nvidia-smi <5 GB.
3. `LTR_DECISION_TORCH_THREADS=2` on any CPU BERT arm.
4. **Warm-up must DRAIN**: warm-up requests → wait `waiting=0 ∧ running=0` → 10–20 s
   stabilization → fresh measurement arrival clock. Trimming first-N records is NOT enough.
5. Pre-registered criteria (§4) FROZEN. One primary comparison. No post-hoc thresholds.
6. Every result json → run tag → tar → backup BEFORE release.
7. Pin and record vLLM config: `max_num_seqs`, `max_num_batched_tokens`,
   `gpu_memory_utilization`, prefix-caching, chunked-prefill, preemption mode, swap,
   KV dtype, model/tokenizer hash, CUDA/driver/torch/vLLM versions. Same values every arm.
8. Failed requests are never silently dropped: a confirmatory cell with decision 429/503/
   timeout/fail-open events is invalidated (count and report them all).

## 1. Timeline (8h)

| Slot | Phase | Gate |
|---|---|---|
| 0:00–0:20 | Preflight: versions/hashes; threshold canaries (S2/S3/S4); PolicyFCFS registered; warm-up/drain works; zero fail-open on smoke | all pass or STOP |
| 0:20–0:35 | **Early live smoke**: 2–4 OpenCode agents through full stack — catches broken tool parser / threshold / VRAM fit BEFORE 4h of data. Pilot only, excluded from stats | no silent breakage |
| 0:35–4:30 | **Block 1 confirmatory**: frozen arrivals, randomized policy order, 5 repeats across ≥2 launches; FCFS sentinels at start/mid/end | kill conditions §5 |
| 4:30–6:30 | **Block 2 overhead**: 6 arms, ABBA order, stage-timing chain | — |
| 6:30–7:30 | **Block 3 live**: 4–8 agents closed-loop characterization + native trace + recording | time-boxed |
| 7:30–8:00 | Evidence freeze: checksums, manifests, tar, upload, verify readback, release | INTEGRITY_OK |
| ~4:30 | (parallel) Report-integration checkpoint: Block-1 jsons to Mac | — |

## 2. Block 1 — scheduling policies

**Confirmatory arms (5)** at ONE common offered RPS (calibrated once on PolicyFCFS full
path, ~90%; report realized utilization per arm, never recalibrate per policy):
`stock_fcfs` (deployed-system baseline) · **`policy_fcfs`** (same-AsyncScheduler-base
algorithmic baseline — removes the scheduler-base confound) · `prompt_sjf` · `pure_ltr` ·
**`gated_rule_c`** (slot-preserving: abstained keep FCFS slots, trusted sort among
trusted slots; #15).

**Diagnostic arms (no formal claims):** OracleLengthSJF (oracle-file predictor path; 2
repeats — headroom probe: if oracle ≤ SJF, learned repeats have no scientific headroom);
ZeroToolGate (skip BERT only for zero-tool requests — attribution control if natural
S1/S2 traffic is tiny); 70% + burst sentinels (1–2 runs).

**Estimand split:**
- **1A policy-only**: decision stub serves FROZEN precomputed per-request scores/strata
  (from committed per-row artifacts) to every arm — identical path, differences =
  queue ordering only.
- **1B deployed**: natural paths (SJF/FCFS bypass decision; PureLTR full BERT;
  GatedRuleC short-circuit; GPU micro-batch on). System-level differences.

**Workload**: T10 output (#14). Framing rule: "trace-calibrated multi-tenant **open-loop
replay of agent-derived request payloads**" — never "real agent sessions". The 75 rows =
"real input payloads replayed on Qwen3.5-9B", a marked anchor subset, NOT an independent
validation set. Pre-classify the frozen 75 rows and record exact stratum counts before
the rental (expected combined skip ≈41%, not 45%).

**Telemetry (musts)**: queue depth per tick; actual displacement count/magnitude/age;
full stage-timing chain (client release → dispatch → gateway → decision queue/infer →
vLLM queue → first token → last token); per-request stratum/confidence/abstained/BERT-
invoked/batch-size/raw-score/mapped-estimate/actual-tokens (→ online τ by stratum +
mapper tie/clipping rates); vLLM preemptions/KV-util/throughput; failure counts; output
equivalence (tokens, finish reason, tool calls); per-tenant + per-population latency.

## 3. Block 2 — overhead decomposition (6 arms)

| Arm | Path | Isolates |
|---|---|---|
| D0 | direct vLLM | absolute floor |
| G0 | gateway pass-through (no decision call) | proxy/serialization |
| G1 | gateway + decision STUB (no model) | RPC + validation |
| G2 | gateway + CPU BERT every eligible request | model inference (last rental's shape) |
| G3 | gateway + gate-first CPU BERT | short-circuit benefit |
| G4 | gateway + gate-first GPU micro-batch (fp16, batch 8, 3 ms) | GPU + batching benefit |

ABBA / randomized paired cycles with warm-up+drain per arm. **Primary endpoint =
all-request paired TTFT** + natural-output TTLT (report token-count deltas; optionally
regression-adjust). Matched-completions subset = **sensitivity only** (post-treatment
selection); report the match rate as a result. G4 must also report Qwen token-throughput
impact of co-located BERT (KV-capacity cost is not free).

## 4. Pre-registered statistics (FROZEN)

Paired design (common arrival seeds) ⇒ **paired effect ratios**, never marginal-CI
overlap. Hierarchical bootstrap: resample launches → sessions (paired across policies) →
keep whole sessions. Also show per-run effects (≥2 launches is thin — display, don't hide).

- **Primary superiority**: GatedRuleC vs PromptLengthSJF. Win = upper 95% CI of
  mean-TTLT ratio < 1.0 (practical win < 0.97). Report absolute ms too.
- **Primary safety (non-inferiority)**: GatedRuleC vs PolicyFCFS with margin δ=3%:
  safe ⟺ upper one-sided 95% CI of ratio < 1.03. "CI includes 0" is NOT safety.
  Evaluate separately on the ABSTAINED population (overall result can hide fallback harm).
- Secondary (no multiplicity claims): PureLTR vs SJF (mechanism), GatedRuleC vs
  ZeroToolGate (attribution), Oracle vs SJF (headroom), stock_fcfs vs policy_fcfs
  (implementation-path cost).
- p95/p99/max-wait reported as SAFETY DIAGNOSTICS, never superiority claims.
- Terminology: Rule C confidence = "stratum-level abstention/eligibility score", never
  "calibrated reliability probability".
- Outcome wordings (pre-drafted): win → "On trace-calibrated open-loop agent-derived
  traffic, slot-preserving gated ranking improves pooled mean TTLT over a free length
  heuristic by X% [paired CI]." tie/loss → "A free prompt-length heuristic captures most
  of the scheduling benefit on this workload; the Ranker's demonstrated value remains
  cold-start ranking quality and gate safety."

## 5. Kill conditions (stop/pivot immediately)

1. PolicyFCFS reproduces most of the prior ~15% gain → attribution bug; stop learned
   repeats, fix instrumentation. (This may explain last rental's uniform gains.)
2. Oracle ≤ SJF → no headroom; pivot budget to overhead + characterization.
3. Reorder-opportunity rate ≈ 0 (queue depth <2 mostly) → recalibrate contention first.
4. Material output mismatch across arms → fix determinism or switch to controlled-work
   comparison.
5. Any decision-path failures in a confirmatory cell → invalidate the cell.
6. G4 reduces Qwen throughput more than it saves decision time → record as negative
   deployability result.
7. Natural trace mix has ~no S1/S2 → run ZeroToolGate, narrow the Rule C claim.

## 6. Assets checklist (before paying)

- [ ] main green: #12 T8 ✅, #13 T9 ✅, #14 T10 ✅, #15 gated_rule_c + policy_fcfs (in flight)
- [ ] frozen-75 stratum classification recorded (pre-rental, 5-min script)
- [ ] decision stub mode for Block 1A (ConstantPredictor / frozen-score file — verify CLI)
- [ ] oracle-file predictor path smoke-tested locally
- [ ] checkpoints + tier2 sample/ledger + quantile inputs + probes/agent-traces local
- [ ] gateway binary builds; live-chain runbook fresh (07-26 PASS)
- [ ] ONNX rejected — no rental time on it
- [ ] launch flags: `--reliability-threshold 0.5 --device cuda --batch-max 8
      --batch-window-ms 3` (fp16 auto; expect p50 ~38 ms warm)
