#!/usr/bin/env python3
"""Gated hybrid scheduling policies for mixed chat/tool-call traffic.

Models the REAL deployment situation: one chat-trained predictor whose
ranking quality is good on chat (tau ~ 0.6) and collapsed on tool-call
(tau ~ 0). Five policies from the 2026-07-13 deep-research recommendation:

  fcfs           arrival order (baseline)
  pure_ltr       trust the predictor everywhere (vLLM-LTR as-is)
  ltr_aging      predictor + aging protection (deadline-normalized decay)
  tail_safe      prediction-free, category-level tail-risk cost + aging
  gated_hybrid   traffic split: chat -> ltr_aging, tool -> tail_safe
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import offline_policy_simulator as sim
import tau_synth as ts

POLICIES = ("fcfs", "pure_ltr", "ltr_aging", "tail_safe", "gated_hybrid")
DEADLINE_MS = 2000.0

# Real measured taus: chat PARS 0.596; BFCL collapse ~0 (probe measured -0.015).
CHAT_TAU_TARGET = 0.596
TOOL_TAU_TARGET = 0.0


def _calibrate_strength(items, target_tau, seed, lo=0.0, hi=8.0, iters=18):
    """Bisection on corruption strength so achieved tau ~= target (vs service time)."""
    costs = [sim.service_time(i) for i in items]
    lengths = [i.output_tokens for i in items]

    def tau_at(strength):
        vals = []
        for rep in range(3):
            rng = random.Random(seed * 7919 + rep)
            scores = ts.corrupt_scores(lengths, "uniform", strength, rng)
            vals.append(ts.kendall_tau(scores, costs))
        return statistics.median(vals)

    if tau_at(lo) <= target_tau:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if tau_at(mid) > target_tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def domain_predictor_scores(items, *, chat_tau, tool_tau, seed):
    """Per-request predictor scores from ONE chat-trained predictor:
    good ranking on chat items, collapsed ranking on tool items."""
    scores: dict[str, float] = {}
    for kind, target in (("chat", chat_tau), ("tool", tool_tau)):
        subset = [i for i in items if i.kind == kind]
        if not subset:
            continue
        strength = _calibrate_strength(subset, target, seed)
        rng = random.Random(seed * 104_729 + hash(kind) % 1024)
        vals = ts.corrupt_scores([i.output_tokens for i in subset], "uniform", strength, rng)
        n = len(subset)
        for item, val in zip(subset, vals):
            # normalize to [0,1] rank-space so chat and tool scores are comparable
            scores[item.request_id] = val / max(1.0, float(n))
    return scores


def pure_ltr_key(scores, item, age, arrival_time):
    return (scores[item.request_id], arrival_time)


def ltr_aging_key(scores, item, age, arrival_time):
    return (scores[item.request_id] / (1.0 + age / DEADLINE_MS), arrival_time)


def tail_safe_key(item, age, arrival_time):
    """Prediction-free, DEPLOYABLE tail-safe cost: gateway-visible category
    features only (never true service time - that would be an oracle)."""
    expected = sim.category_cost_score(item)
    cost = expected * (1.0 + 0.35 * sim.tail_risk_multiplier(item))
    return (cost / (1.0 + age / DEADLINE_MS), arrival_time)


def gated_hybrid_key(scores, item, age, arrival_time, kind_override=None):
    kind = kind_override if kind_override is not None else item.kind
    if kind == "tool":
        return tail_safe_key(item, age, arrival_time)
    # tail-safe keys are cost-scaled (hundreds); scale chat rank scores into a
    # comparable magnitude so neither class starves the other structurally
    rank_cost = scores[item.request_id] * 400.0
    return (rank_cost / (1.0 + age / DEADLINE_MS), arrival_time)


def run_policy(arrivals, policy, scores, *, qps=0.0, seed=0, tool_ratio=0.0,
               flip_prob=0.0, flip_seed=0):
    if policy == "fcfs":
        return sim.simulate_queue(arrivals, "fcfs", qps=qps, seed=seed, tool_ratio=tool_ratio)
    flipped: dict[str, str] = {}
    if flip_prob > 0.0:
        rng = random.Random(flip_seed * 65_537 + 13)
        for _, item in arrivals:
            if rng.random() < flip_prob:
                flipped[item.request_id] = "chat" if item.kind == "tool" else "tool"
    key_fns = {
        "pure_ltr": lambda item, age, at: pure_ltr_key(scores, item, age, at),
        "ltr_aging": lambda item, age, at: ltr_aging_key(scores, item, age, at),
        "tail_safe": tail_safe_key,
        "gated_hybrid": lambda item, age, at: gated_hybrid_key(
            scores, item, age, at, kind_override=flipped.get(item.request_id)),
    }
    return sim.simulate_queue(
        arrivals, policy, qps=qps, seed=seed, tool_ratio=tool_ratio, key_fn=key_fns[policy])


def run_matrix(args):
    chat_items = sim.load_chat_items(args.chat_archive, args.chat_json_pattern, args.chat_limit)
    tool_items = sim.load_bfcl_items(args.bfcl_lengths, args.tool_limit)

    rows = []
    for tool_ratio in args.tool_ratios:
        for seed in args.seeds:
            mixed = sim.build_mixed_items(
                chat_items, tool_items,
                total=args.total_requests, tool_ratio=tool_ratio, seed=seed)
            scores = domain_predictor_scores(
                mixed, chat_tau=CHAT_TAU_TARGET, tool_tau=TOOL_TAU_TARGET, seed=seed)
            for qps in args.qps_values:
                arrivals = sim.make_arrivals(mixed, qps, seed + int(qps * 100))
                fcfs = run_policy(arrivals, "fcfs", scores, qps=qps, seed=seed, tool_ratio=tool_ratio)
                for policy in POLICIES:
                    res = run_policy(arrivals, policy, scores, qps=qps, seed=seed, tool_ratio=tool_ratio)
                    rows.append({
                        "tool_ratio": tool_ratio, "qps": qps, "seed": seed, "policy": policy,
                        "count": res.count,
                        "mean_wait": round(res.mean_wait, 3),
                        "p95_wait": round(res.p95_wait, 3),
                        "p99_wait": round(res.p99_wait, 3),
                        "max_wait": round(res.max_wait, 3),
                        "starvation_count": res.starvation_count,
                        "mean_speedup_vs_fcfs": round(fcfs.mean_wait / res.mean_wait, 3) if res.mean_wait else 0.0,
                        "p95_speedup_vs_fcfs": round(fcfs.p95_wait / res.p95_wait, 3) if res.p95_wait else 0.0,
                        "p99_ratio_vs_fcfs": round(res.p99_wait / fcfs.p99_wait, 3) if fcfs.p99_wait else 0.0,
                    })
    return rows


def write_summary(path, rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["tool_ratio"], r["policy"]), []).append(r)
    lines = [
        "# Gated Hybrid Policy Comparison",
        "",
        "One chat-trained predictor (chat tau ~0.596, tool tau ~0 = measured BFCL collapse)",
        "applied to mixed traffic. Medians across seeds and qps.",
        "",
        "| tool_ratio | policy | mean_speedup | p95_speedup | p99_ratio | starvation(med) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for (ratio, policy) in sorted(groups):
        rs = groups[(ratio, policy)]
        lines.append("| {:.2f} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            ratio, policy,
            statistics.median(r["mean_speedup_vs_fcfs"] for r in rs),
            statistics.median(r["p95_speedup_vs_fcfs"] for r in rs),
            statistics.median(r["p99_ratio_vs_fcfs"] for r in rs),
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
    parser.add_argument("--out-dir", type=Path, default=Path("project/gateway_policy_probe/results_hybrid"))
    parser.add_argument("--chat-limit", type=int, default=1200)
    parser.add_argument("--tool-limit", type=int, default=3600)
    parser.add_argument("--total-requests", type=int, default=500)
    parser.add_argument("--tool-ratios", type=sim.parse_csv_floats, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--qps-values", type=sim.parse_csv_floats, default="2,3,4,5")
    parser.add_argument("--seeds", type=sim.parse_csv_ints, default="42,6806,20260709")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("tool_ratios", "qps_values", "seeds"):
        value = getattr(args, name)
        if isinstance(value, str):
            parser_fn = sim.parse_csv_ints if name == "seeds" else sim.parse_csv_floats
            setattr(args, name, parser_fn(value))
    rows = run_matrix(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim.write_csv(args.out_dir / "hybrid_matrix.csv", rows)
    write_summary(args.out_dir / "HYBRID.md", rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
