#!/usr/bin/env python3
"""Phase 2 analysis: does Frozen Thin's token cut become real prefill/KV savings?

Cold start (cache OFF, or cache ON but first-ever request for a session) is where
the offline probe said Thin should pay: 22.3% fewer prompt tokens on the vanilla
trace. This checks whether the saving survives contact with the scheduler.
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path

POLICIES = ["Original", "Frozen Thin"]

# Thresholds from the scoped Go/No-Go.
GO_PREFILL_DROP = 0.20      # cold-start prefill time or TTFT
GO_TOKEN_DROP = 0.30        # new prefill tokens or KV footprint


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fnum(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def clean(xs):
    return [x for x in xs if x is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    if not rows:
        raise SystemExit("no rows")

    errs = [r for r in rows if r.get("error")]
    ooms = [r for r in rows if r.get("oom") == "1"]
    print(f"total requests: {len(rows)}   errors: {len(errs)}   OOM: {len(ooms)}")
    if errs:
        print(f"  first error: {errs[0]['error'][:160]}")
    runs = sorted({r["run_id"] for r in rows})
    print(f"server restarts per arm: {runs}\n")

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["cache_mode"], r["policy"])].append(r)

    hdr = (f"{'cache':<6}{'policy':<14}{'n':>4}{'tools':>7}{'prompt_tok':>12}"
           f"{'new_prefill':>13}{'wall_ms':>10}{'kv_usage':>11}")
    print(hdr)
    print("-" * len(hdr))
    S = {}
    for cache in ("on", "off"):
        for pol in POLICIES:
            rs = [r for r in by.get((cache, pol), []) if not r.get("error")]
            if not rs:
                continue
            S[(cache, pol)] = {
                "n": len(rs),
                "tools": med(clean([fnum(r, "n_tools_sent") for r in rs])),
                "prompt": med(clean([fnum(r, "prompt_tokens") for r in rs])),
                "new": med(clean([fnum(r, "new_prefill_tokens") for r in rs])),
                "wall": med(clean([fnum(r, "wall_ms") for r in rs])),
                "kv": med(clean([fnum(r, "g_kv_cache_usage_perc") for r in rs])),
            }
            s = S[(cache, pol)]
            print(f"{cache:<6}{pol:<14}{s['n']:>4}{s['tools']:>7.0f}"
                  f"{s['prompt']:>12.0f}{s['new']:>13.0f}{s['wall']:>10.1f}"
                  f"{s['kv']:>11.5f}")

    print("\n--- cold start (cache OFF): Original vs Frozen Thin ---")
    evaluate(S, "off")
    print("\n--- cache ON (session heads still cold on first contact) ---")
    evaluate(S, "on")


def evaluate(S, cache):
    o, t = S.get((cache, "Original")), S.get((cache, "Frozen Thin"))
    if not (o and t):
        print("  missing an arm")
        return
    def drop(a, b):
        return (a - b) / a if a else float("nan")
    d_prompt = drop(o["prompt"], t["prompt"])
    d_new = drop(o["new"], t["new"])
    d_wall = drop(o["wall"], t["wall"])
    d_kv = drop(o["kv"], t["kv"]) if o["kv"] else float("nan")

    print(f"  prompt tokens   {o['prompt']:>8.0f} -> {t['prompt']:>8.0f}   {d_prompt:+7.1%}")
    print(f"  new prefill tok {o['new']:>8.0f} -> {t['new']:>8.0f}   {d_new:+7.1%}")
    print(f"  wall time (ms)  {o['wall']:>8.1f} -> {t['wall']:>8.1f}   {d_wall:+7.1%}")
    print(f"  kv usage        {o['kv']:>8.5f} -> {t['kv']:>8.5f}   {d_kv:+7.1%}")

    hits = []
    if d_wall >= GO_PREFILL_DROP:
        hits.append(f"wall/TTFT drop {d_wall:.1%} >= {GO_PREFILL_DROP:.0%}")
    if d_new >= GO_TOKEN_DROP:
        hits.append(f"new prefill tokens drop {d_new:.1%} >= {GO_TOKEN_DROP:.0%}")
    if d_kv >= GO_TOKEN_DROP:
        hits.append(f"KV footprint drop {d_kv:.1%} >= {GO_TOKEN_DROP:.0%}")
    print(f"  Go criteria met: {hits if hits else 'none'}")

    # The honest caveat: a token cut that does not beat its own proportion is
    # just doing less work, not doing work better.
    if d_prompt > 0:
        ratio = d_wall / d_prompt
        print(f"  wall-time drop as a share of the token drop: {ratio:.2f}x "
              f"({'super-proportional' if ratio > 1.1 else 'proportional or less'})")


if __name__ == "__main__":
    main()
