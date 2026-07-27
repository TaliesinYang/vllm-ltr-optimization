"""Prefix caching on versus off: the controlled comparison against round A.

Round A ran all six arms with vLLM prefix caching disabled (verified 0.0% hit
rate); the prefix-cache run repeats them with it enabled and nothing else
changed. Same workload, same arrival offsets, same policies, so each arm pairs
with its round-A self at session level.

The comparison is only as good as its bridge. The two sessions are hours
apart on a rented GPU, and the measured session drift (1.15%) is the same
order as effects worth reporting. Stock and shim are the bridge arms: neither
consults the ranker, so prefix caching is the only thing that should move
them, and their on/off ratio bounds session drift plus the caching effect on
uncached-policy traffic. A policy arm's ratio is interpreted RELATIVE to the
bridge, and if the two bridge arms disagree with each other beyond their
intervals, the session moved and every cross-session delta here is suspect --
the script says so instead of printing numbers anyway.

Run: python3 fig_prefix_cache.py [on-tag] [off-tag]
"""

from __future__ import annotations

import sys

import numpy as np

from fig_block1 import (
    ARMS, BOOTSTRAP_DRAWS, RUNS, hierarchical_draws, interval, load_arm,
    pooled_mean, session_of_request,
)

BRIDGE = ("stock_fcfs", "StockFCFSShim")


def load_run(tag: str, sessions) -> dict:
    root = RUNS.parent / tag / "matrix"
    out = {}
    for stem in ARMS:
        directory = root / f"{stem}.runs"
        if not directory.exists():
            print(f"  {tag}/{stem}: absent, skipping")
            continue
        out[stem] = load_arm_from(directory, sessions)
    return out


def load_arm_from(directory, sessions):
    import csv
    launches = []
    for path in sorted(directory.glob("*.samples.csv")):
        by_session = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("error") or "").strip():
                    continue
                key = sessions.get(row["request_id"])
                if key is None:
                    continue
                by_session.setdefault(key, []).append(float(row["ttlt_ms"]))
        launches.append(by_session)
    return launches


def main() -> None:
    on_tag = sys.argv[1] if len(sys.argv) > 1 else "prefix-cache"
    off_tag = sys.argv[2] if len(sys.argv) > 2 else "block1-main"
    sessions = session_of_request()

    on = load_run(on_tag, sessions)
    off = {stem: load_arm(stem, sessions) for stem in ARMS
           if (RUNS / "matrix" / f"{stem}.runs").exists()}
    common = [s for s in ARMS if s in on and s in off]
    if not common:
        raise SystemExit("no arm present in both runs")

    shared = sorted(set.intersection(*(
        set(l) for run in (on, off) for stem in common for l in run[stem])))
    print(f"{len(common)} arms in both runs, {len(shared)} shared sessions")

    # One joint resampling across BOTH runs so every ratio is paired.
    joint = {f"on::{s}": on[s] for s in common} | {f"off::{s}": off[s] for s in common}
    counts = {len(v) for v in joint.values()}
    if len(counts) != 1:
        # Launch counts may legitimately differ (3 vs 3 expected); truncate to min.
        m = min(counts)
        joint = {k: v[:m] for k, v in joint.items()}
        print(f"  note: launch counts differ; truncated all arms to {m}")
    draws = hierarchical_draws(joint, shared, seed=20260728)

    print(f"\n{'arm':22} {'off ms':>9} {'on ms':>9} {'on/off':>8}  95% CI")
    bridge_ratios = {}
    for stem in common:
        p_on = pooled_mean(joint[f"on::{stem}"], shared)
        p_off = pooled_mean(joint[f"off::{stem}"], shared)
        ratios = draws[f"on::{stem}"] / draws[f"off::{stem}"]
        low, high = interval(ratios)
        tagm = "  <- bridge" if stem in BRIDGE else ""
        print(f"{ARMS[stem][0]:22} {p_off:9.1f} {p_on:9.1f} {p_on/p_off:8.4f}"
              f"  [{low:.4f}, {high:.4f}]{tagm}")
        if stem in BRIDGE:
            bridge_ratios[stem] = (p_on / p_off, low, high)

    if len(bridge_ratios) == 2:
        (r1, l1, h1), (r2, l2, h2) = bridge_ratios.values()
        agree = (l1 <= r2 <= h1) or (l2 <= r1 <= h2)
        print(f"\nbridge agreement: {'YES' if agree else 'NO'}"
              f"  ({r1:.4f} vs {r2:.4f})")
        if not agree:
            print("BRIDGE FAILED: the two ranker-free arms moved differently "
                  "between sessions. Session drift is not uniform and every "
                  "cross-session delta above is suspect. Do not quote them.")
            raise SystemExit(2)
        print("Interpretation: the bridge ratio is drift + caching on "
              "policy-free traffic; a policy arm matters only where its ratio "
              "separates from the bridge's.")


if __name__ == "__main__":
    main()
