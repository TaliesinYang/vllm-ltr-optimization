# Rental-day runbook — 2026-07-26 (E5: blocks 1/2/3)

Purpose ruling (user, 07-26): **(c)** — minimal core matrix first to feed the Final
Report (due 07-27 08:59), then extensions. Presentation 07-29 gets the live material.

## 0. Hard rules (violating any one invalidates the day)

1. **Decision service MUST launch with `--reliability-threshold 0.5`** (≤0.5787).
   Default 0.8 > Rule C max 0.6233 ⇒ every request unreliable ⇒ Ranker contributes
   nothing and the run silently measures noise. Pinned by test; check the launch line.
2. Teardown between vLLM restarts: `pkill -9 -f "VLLM::EngineCore"` + poll nvidia-smi
   <5 GB before relaunch (7/19 lesson; in run_matrix teardown + standalone runners).
3. `LTR_DECISION_TORCH_THREADS=2` for any CPU BERT arm (7/19 lesson).
4. Pre-registered criteria (§4) are FROZEN. No threshold moves after data.
5. Every result json lands under one run tag + tar + OSS/local backup before release.

## 1. Timeline (8h budget, order fixed)

| Slot | What | Output |
|---|---|---|
| T+0:00–0:45 | Restore env (runbook §2), preflight, saturation calibration | go/no-go |
| T+0:45–3:15 | **Block 1 core**: 4 policies × trace-calibrated workload × ≥5 repeats | report-feeding matrix |
| T+3:15–4:45 | **Block 2**: O1/O2/O3/O5 overhead ablation, matched-completions | the 975ms→53ms arc |
| T+4:45–5:30 | Report-integration checkpoint: pull jsons to Mac, quick pooled stats; **anything for the report leaves the box HERE** | numbers to prose |
| T+5:30–7:00 | **Block 3**: 4–8 parallel OpenCode live + native trace collection + demo recording | traces + footage |
| T+7:00–8:00 | Extensions if green (extra repeats, batch-size sweep), tar + upload + release | archive |

## 2. Launch sequence (adapted from scripts/server/README.md)

```
oss_login (or skip; local backups exist)
restore_from_oss OR rsync from Mac: repo@main (includes T8/T9/T10), checkpoints seed17,
  tier2 sample+ledger, probes/agent-traces-2026-07-26 (calibration source)
setup_env (torch cu; verify torch.cuda.is_available())
build_gateway (feat/ltr-decision-adapter, pin 888fba9+)
launch decision service:
  --predictor bert --checkpoint checkpoints_best_predictor \
  --quantile-manifest <built from labels-merged, 3 exclusions declared> \
  --reliability-threshold 0.5 --device cuda --batch-max 8 --batch-window-ms 3   # T9 flags
measure_decision_latency (conc 8, 200 samples) — expect p99 well under 100ms on GPU
launch gateway (LTR_DECISION_ENDPOINT, timeout per manifest)
build_server_workloads --trace-calibrated (T10) — manifest printed, verify params vs
  probes/agent-traces-2026-07-26/MANIFEST.md
calibrate_saturation (~90%)
run_matrix per block
```

## 3. Blocks

**Block 1** — policies: `stock_fcfs`, `prompt_sjf`, `pure_ltr`, `gated_rule_c`.
Workload: T10 output (synthetic multi-tenant, all four strata present in queue, 33%
zero-tool, per-client constant schemas; 75 real traces embedded + marked). ≥5 repeats.
Metrics per request: TTFT, TTLT, completion tokens, policy decision metadata
(stratum, abstained y/n).

**Block 2** — arms O1 direct / O2 gateway+sync-CPU-BERT / O3 +short-circuit /
O5 O3+GPU-micro-batch. Same workload subsample (≥150 rows), matched-completions
methodology for cross-arm claims (7/20 audit standard).

**Block 3** — live: Mac runs 4–8 `opencode run` instances in parallel through the
tunnel; tasks from run_trace_batch.sh list; capture via trace proxy; gateway logs
authoritative for latency (client side includes Mac↔box network). Record screen for
the 7/29 deck. **Characterization only — no hypothesis claims** (pre-registered).

## 4. Pre-registered criteria (FROZEN before data)

- Primary metric: **pooled mean TTLT** (p99 reported, never claimed — UniBoost closed
  the tail claim).
- C1 safety: GatedRuleC not worse than stock_fcfs (bootstrap CI includes 0 or better).
- C2 value: PureLTR / GatedRuleC vs PromptLengthSJF — win only if CI-separated.
- C3 overhead: O3+O5 decision-path p99 < 100 ms at conc 8 ⇒ "sync path revived";
  else report honestly that async remains the answer.
- Outcome wordings (both pre-drafted, SYNC-CHECKLIST #11 discipline):
  - C2 win: "On trace-calibrated heterogeneous agent traffic, schema-text ranking
    improves pooled mean TTLT over a free length heuristic by X% [CI]."
  - C2 tie/loss: "On this workload a free prompt-length heuristic captures most of the
    scheduling benefit; the Ranker's demonstrated value remains cold-start ranking
    quality and gate safety, not serving-latency gain."

## 5. Roles

- Machine box: benchmark execution (this runbook).
- Mac: OpenCode clients, trace proxy, report integration at T+4:45.
- Human: prose track continues in parallel; only decision points interrupt.

## 6. Assets checklist (verify BEFORE paying)

- [ ] main is green: T8 (#12 closed), T9 (#13), T10 (#14) merged, full suite passes
- [ ] checkpoints seed17 + tier2 sample/ledger + quantile inputs local
- [ ] probes/agent-traces-2026-07-26 present (T10 calibration source)
- [ ] gateway binary builds at pin; DEV_API_KEY flow tested (7/26 live chain PASS)
- [ ] demo kit reviewed (probes/full-stack-demo-2026-07-26/DEMO-KIT.md)
- [ ] web-ask design review verdicts folded in (docs/, pending tonight)
- [ ] ONNX: rejected (int8 parity FAIL, fp32 no speedup) — do NOT spend rental time on it
