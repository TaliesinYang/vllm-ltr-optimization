#!/usr/bin/env bash
# Three configurations the first Block-1 round did not exercise, run back to
# back so the machine is never idle.
#
#   B  the 15 ms decision contract the paper asserts in seven places and never
#      enforced. One policy, one launch. The expected outcome is that every
#      trusted request times out and the arm degrades to arrival order, which
#      is not a tautology: it measures whether fail-open is correct end to end,
#      and what fraction of traffic the gate can actually cover under the
#      documented budget.
#
#   A  vLLM prefix caching. The workload's tool schema is byte-identical every
#      turn, so a reusable prefix exists; round A measured 0.0% hit rate
#      because the cache was off. Same six arms, same workload, one variable
#      changed, so this is a controlled comparison against round A and not a
#      separate experiment.
#
#   C  the gateway's own scheduler path, which brings its scorer concurrency
#      cap and slow-threshold degradation with it. Executor concurrency is
#      raised to 64 deliberately: the default of 1 serialises the gateway and
#      would drain the vLLM queue that our policies reorder, producing a null
#      ordering result for a reason that has nothing to do with ordering.
#
# Every arm writes under its own RUN_TAG, so nothing overwrites round A.
set -euo pipefail

LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
REPO_ROOT="${REPO_ROOT:-$LTR_ROOT/repo}"
ARTIFACTS="${ARTIFACTS:-$LTR_ROOT/artifacts/current}"
WORKLOAD="${MIXED_WORKLOAD:-$ARTIFACTS/workload-block1.jsonl}"
CHECKPOINT="${CHECKPOINT:-$ARTIFACTS/checkpoints_best_predictor}"
ONLY="${ONLY:-B,A,C}"

log() { echo "$(date +%H:%M:%S) [sweep] $*"; }

matrix() {  # matrix <run-tag> <round-a-repeats>
  ROUND_A_REPEATS="$2" ROUND_B_REPEATS=1 RUN_TAG="$1" \
  MIXED_WORKLOAD="$WORKLOAD" CHECKPOINT="$CHECKPOINT" \
    bash "$REPO_ROOT/scripts/server/run_block1_matrix.sh"
  # Counters are collected per arm rather than at the end: a hole found after
  # the machine is returned cannot be filled, and a flag that silently did not
  # take looks exactly like a flag that did until someone reads the log.
  bash "$REPO_ROOT/scripts/server/collect_run_counters.sh" \
    "$LTR_ROOT/runs/$1" || log "WARNING: counters incomplete for $1"
}

if [[ ",$ONLY," == *",B,"* ]]; then
  log "B: 15 ms decision contract, GatedRuleC only"
  # The override is what makes this arm different; everything else is the
  # configuration round A used.
  LTR_DECISION_TIMEOUT_MS_OVERRIDE=15 \
  BLOCK1_ONLY_CLASS=scheduler_benchmark.vllm_scheduler.GatedRuleCScheduler \
  SKIP_ROUND_B=1 \
    matrix contract-15ms 1
  log "B done"
fi

if [[ ",$ONLY," == *",A,"* ]]; then
  log "A: vLLM prefix caching on, six arms"
  VLLM_PREFIX_CACHING=1 matrix prefix-cache 3
  log "A done"
fi

if [[ ",$ONLY," == *",C,"* ]]; then
  # Block 2 already exists for this and is better than a hand-rolled three-arm
  # version: six arms each adding exactly one component, so consecutive
  # differences are per-component costs, and an ABBA order so a monotone drift
  # cancels in each arm's two halves instead of being charged to whichever arm
  # ran late. D0 is the direct-to-vLLM baseline without which no claim about
  # the gateway repaying its hop can be made at all.
  #
  # The only change here is the arrival rate. A gateway buys governance and
  # pays a hop; below saturation there is nothing for admission control to
  # prevent, so overload is the one regime where the hop can pay for itself.
  OVERLOAD_RPS="${OVERLOAD_RPS:-6.4}"   # ~130% of the 4.9 rps saturation point
  log "C: Block-2 six-arm ABBA decomposition at ${OVERLOAD_RPS} rps (overload)"
  LTR_ROOT="$LTR_ROOT" REPO_ROOT="$REPO_ROOT" \
  MIXED_WORKLOAD="$WORKLOAD" CHECKPOINT="$CHECKPOINT" \
  CAPACITY_RPS_OVERRIDE="$OVERLOAD_RPS" \
  VLLM_PREFIX_CACHING=1 \
  SCHEDULER_ENABLED=true \
  SCHEDULER_TIMEOUT=15ms \
  SCHEDULER_SCORER_MAX_CONCURRENCY=4 \
  SCHEDULER_SCORER_SLOW_THRESHOLD=15ms \
  SCHEDULER_EXECUTOR_CONCURRENCY=64 \
  SCHEDULER_QUEUE_SOFT_LIMIT=64 \
  SCHEDULER_QUEUE_HARD_LIMIT=256 \
  SCHEDULER_QUEUE_BACKEND=memory \
  RUN_TAG=overload-block2 \
    bash "$REPO_ROOT/scripts/server/run_block2_overhead.sh"
  log "C done"
fi

log "sweep complete"
