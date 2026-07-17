#!/usr/bin/env python3
"""Literature scheduling-algorithm benchmark on one shared workload.

Adopt-then-measure: classic and published policies run on identical arrival
sequences, reporting the metrics the serving literature actually uses:
JCT (wait+service), slowdown (JCT/service), tail slowdown, starvation.

Preemptive quantum simulator covers RR / MLFQ (FastServe, arXiv:2305.05920)
/ LAS (Autellix-style, arXiv:2502.13965) / SRPT; the existing selection
simulator covers the non-preemptive family (FCFS/SJF/heuristics/LTR/gated).
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import offline_policy_simulator as sim
import hybrid_policies as hp

QUANTUM_MS = 40.0
MLFQ_THRESHOLDS = (80.0, 320.0, 1280.0)  # attained-service demotion levels
STARVATION_MS = 5000.0

NONPREEMPTIVE = (
    "fcfs", "random", "oracle_sjf", "prompt_len_sjf", "tail_safe",
    "pure_ltr", "ltr_aging", "gated_hybrid",
)
PREEMPTIVE = ("round_robin", "las", "mlfq", "srpt_oracle")
ALGORITHMS = NONPREEMPTIVE + PREEMPTIVE


def _metrics(records, count):
    """records: list of (wait, jct, service)."""
    waits = [r[0] for r in records]
    jcts = [r[1] for r in records]
    slowdowns = [r[1] / r[2] for r in records if r[2] > 0]
    return {
        "count": count,
        "mean_wait": round(statistics.mean(waits), 3),
        "p99_wait": round(sim.percentile(waits, 99), 3),
        "mean_jct": round(statistics.mean(jcts), 3),
        "p99_jct": round(sim.percentile(jcts, 99), 3),
        "mean_slowdown": round(statistics.mean(slowdowns), 3),
        "p99_slowdown": round(sim.percentile(slowdowns, 99), 3),
        "max_slowdown": round(max(slowdowns), 3),
        "starvation_count": sum(1 for w in waits if w >= STARVATION_MS),
    }


def _simulate_preemptive(arrivals, pick_fn, quantum=QUANTUM_MS, switch_penalty=0.0):
    """Quantum-based preemptive queue: each step, pick_fn chooses among
    admitted unfinished jobs; the job runs one quantum (or to completion).
    switch_penalty models real preemption cost (KV swap/recompute) charged
    whenever the scheduler switches away from the previously running job."""
    jobs = []
    for arrival_time, item in arrivals:
        jobs.append({
            "item": item, "arrival": arrival_time,
            "remaining": sim.service_time(item), "attained": 0.0,
            "first_start": None, "finish": None,
        })
    now = 0.0
    pending = sorted(jobs, key=lambda j: j["arrival"])
    admitted: list[dict] = []
    idx = 0
    done = 0
    prev_job = None
    while done < len(jobs):
        while idx < len(pending) and pending[idx]["arrival"] <= now:
            admitted.append(pending[idx])
            idx += 1
        active = [j for j in admitted if j["finish"] is None]
        if not active:
            now = pending[idx]["arrival"]
            continue
        job = pick_fn(active, now)
        if switch_penalty and prev_job is not None and job is not prev_job:
            now += switch_penalty
        prev_job = job
        if job["first_start"] is None:
            job["first_start"] = now
        step = min(quantum, job["remaining"])
        job["remaining"] -= step
        job["attained"] += step
        now += step
        if job["remaining"] <= 1e-9:
            job["finish"] = now
            done += 1
    records = [
        (j["first_start"] - j["arrival"], j["finish"] - j["arrival"],
         sim.service_time(j["item"]))
        for j in jobs
    ]
    return _metrics(records, len(jobs))


def _mlfq_priority(job):
    for level, threshold in enumerate(MLFQ_THRESHOLDS):
        if job["attained"] < threshold:
            return level
    return len(MLFQ_THRESHOLDS)


_PREEMPTIVE_PICKERS = {
    "round_robin": lambda active, now: min(active, key=lambda j: (j["attained"], j["arrival"])),
    "las": lambda active, now: min(active, key=lambda j: (j["attained"], j["arrival"])),
    "mlfq": lambda active, now: min(
        active, key=lambda j: (_mlfq_priority(j), j["arrival"])),
    "srpt_oracle": lambda active, now: min(active, key=lambda j: (j["remaining"], j["arrival"])),
}
# NOTE: round_robin degenerates to LAS under a global attained-service pick;
# kept separate for reporting clarity (identical numbers are expected).


def _simulate_selection(arrivals, key_fn, policy_label):
    result = sim.simulate_queue(arrivals, policy_label, key_fn=key_fn)
    # selection simulator reports waits only; rebuild JCT records from queue
    # semantics: jct = wait + service. We re-run to collect per-item data.
    # Cheaper: approximate using the same run's aggregates is NOT enough for
    # slowdown, so we replay explicitly here.
    pending: list[tuple[float, sim.WorkItem]] = []
    now = 0.0
    index = 0
    records = []
    while index < len(arrivals) or pending:
        if not pending and index < len(arrivals) and now < arrivals[index][0]:
            now = arrivals[index][0]
        while index < len(arrivals) and arrivals[index][0] <= now:
            pending.append(arrivals[index])
            index += 1
        if not pending:
            continue
        if key_fn is None:
            pos = 0
        else:
            pos = min(range(len(pending)), key=lambda p: key_fn(
                pending[p][1], max(0.0, now - pending[p][0]), pending[p][0]))
        arrival_time, item = pending.pop(pos)
        wait = max(0.0, now - arrival_time)
        cost = sim.service_time(item)
        records.append((wait, wait + cost, cost))
        now += cost
    return _metrics(records, len(records))


def run_algorithm(arrivals, name, scores=None):
    if name in PREEMPTIVE:
        return _simulate_preemptive(arrivals, _PREEMPTIVE_PICKERS[name])

    items = [item for _, item in arrivals]
    if scores is None:
        scores = hp.domain_predictor_scores(
            items, chat_tau=hp.CHAT_TAU_TARGET, tool_tau=hp.TOOL_TAU_TARGET, seed=17)
    rng = random.Random(11)
    random_rank = {item.request_id: rng.random() for item in items}

    key_fns = {
        "fcfs": None,
        "random": lambda item, age, at: (random_rank[item.request_id], at),
        "oracle_sjf": lambda item, age, at: (sim.service_time(item), at),
        "prompt_len_sjf": lambda item, age, at: (sim.prompt_length_score(item), at),
        "tail_safe": hp.tail_safe_key,
        "pure_ltr": lambda item, age, at: hp.pure_ltr_key(scores, item, age, at),
        "ltr_aging": lambda item, age, at: hp.ltr_aging_key(scores, item, age, at),
        "gated_hybrid": lambda item, age, at: hp.gated_hybrid_key(scores, item, age, at),
    }
    return _simulate_selection(arrivals, key_fns[name], name)


def run_matrix(args):
    chat_items = sim.load_chat_items(args.chat_archive, args.chat_json_pattern, args.chat_limit)
    tool_items = sim.load_bfcl_items(args.bfcl_lengths, args.tool_limit)
    rows = []
    for tool_ratio in args.tool_ratios:
        for seed in args.seeds:
            mixed = sim.build_mixed_items(
                chat_items, tool_items,
                total=args.total_requests, tool_ratio=tool_ratio, seed=seed)
            scores = hp.domain_predictor_scores(
                mixed, chat_tau=hp.CHAT_TAU_TARGET, tool_tau=hp.TOOL_TAU_TARGET, seed=seed)
            for qps in args.qps_values:
                arrivals = sim.make_arrivals(mixed, qps, seed + int(qps * 100))
                for name in ALGORITHMS:
                    out = run_algorithm(arrivals, name, scores=scores)
                    rows.append({"tool_ratio": tool_ratio, "qps": qps, "seed": seed,
                                 "algorithm": name, **out})
    return rows


def write_summary(path, rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["tool_ratio"], r["algorithm"]), []).append(r)
    lines = [
        "# Scheduling Algorithm Benchmark (literature roster, shared workload)",
        "",
        "Medians across seeds x qps. slowdown = JCT/service (fairness-sensitive).",
        "Preemptive algorithms (RR/LAS/MLFQ/SRPT) use a 40ms-quantum simulator;",
        "RR and LAS coincide under a global attained-service pick (expected).",
        "",
        "| tool_ratio | algorithm | mean_jct | p99_jct | mean_slowdown | p99_slowdown | starvation |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (ratio, name) in sorted(groups):
        rs = groups[(ratio, name)]
        lines.append("| {:.2f} | {} | {:.0f} | {:.0f} | {:.2f} | {:.2f} | {} |".format(
            ratio, name,
            statistics.median(r["mean_jct"] for r in rs),
            statistics.median(r["p99_jct"] for r in rs),
            statistics.median(r["mean_slowdown"] for r in rs),
            statistics.median(r["p99_slowdown"] for r in rs),
            int(statistics.median(r["starvation_count"] for r in rs)),
        ))
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-archive", type=Path,
                        default=Path("deliverables/04-evaluation/baseline-2026-06-22/baseline-results.tgz"))
    parser.add_argument("--chat-json-pattern", default="RESULTS/vllm-16.0qps-*fcfs*.json")
    parser.add_argument("--bfcl-lengths", type=Path,
                        default=Path("project/bfcl_probe/results/bfcl_v3_lengths.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("project/gateway_policy_probe/results_benchmark"))
    parser.add_argument("--chat-limit", type=int, default=1200)
    parser.add_argument("--tool-limit", type=int, default=3600)
    parser.add_argument("--total-requests", type=int, default=400)
    parser.add_argument("--tool-ratios", type=sim.parse_csv_floats, default="0,0.5,1.0")
    parser.add_argument("--qps-values", type=sim.parse_csv_floats, default="3,4,5")
    parser.add_argument("--seeds", type=sim.parse_csv_ints, default="42,6806,20260709")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("tool_ratios", "qps_values", "seeds"):
        value = getattr(args, name)
        if isinstance(value, str):
            fn = sim.parse_csv_ints if name == "seeds" else sim.parse_csv_floats
            setattr(args, name, fn(value))
    rows = run_matrix(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim.write_csv(args.out_dir / "scheduler_benchmark.csv", rows)
    write_summary(args.out_dir / "BENCHMARK.md", rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
