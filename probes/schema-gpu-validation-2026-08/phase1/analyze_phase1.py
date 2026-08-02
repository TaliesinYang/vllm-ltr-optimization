#!/usr/bin/env python3
"""Phase 1 analysis: does real vLLM reproduce the offline block simulation?

The decisive check is the Shuffled Full positive control. If reordering the tool
schema does NOT degrade real cache hits, the measurement chain is broken and no
conclusion about the direction can be drawn from any of these numbers.
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path

POLICIES = ["Original", "Stable Full", "Shuffled Full", "Frozen Thin"]

# What the offline probe predicted for the vanilla 10-tool trace, block size 16.
OFFLINE_REUSE = {"Original": 0.9894, "Stable Full": 0.9895,
                 "Shuffled Full": 0.7388, "Frozen Thin": 0.9868}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open()))
    if not rows:
        raise SystemExit("no rows")

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["cache_mode"], r["policy"])].append(r)

    print(f"total measurements: {len(rows)}")
    runs = sorted({r["run_id"] for r in rows})
    print(f"server restarts per arm: {runs}\n")

    hdr = (f"{'cache':<6}{'policy':<15}{'n':>4}{'prompt_tok':>12}{'cached':>10}"
           f"{'new_prefill':>13}{'cached_frac':>13}{'wall_ms':>10}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for cache in ("on", "off"):
        for pol in POLICIES:
            rs = by.get((cache, pol), [])
            if not rs:
                continue
            prompt = [fnum(r, "prompt_tokens") for r in rs]
            cached = [fnum(r, "cached_tokens") for r in rs]
            new = [fnum(r, "uncached_prompt_tokens") for r in rs]
            frac = [fnum(r, "cached_frac") for r in rs]
            wall = [fnum(r, "wall_ms") for r in rs]
            clean = lambda xs: [x for x in xs if x is not None]
            summary[(cache, pol)] = {
                "n": len(rs), "prompt": med(clean(prompt)),
                "cached": med(clean(cached)), "new": med(clean(new)),
                "frac": med(clean(frac)), "wall": med(clean(wall)),
            }
            s = summary[(cache, pol)]
            print(f"{cache:<6}{pol:<15}{s['n']:>4}{s['prompt']:>12.0f}"
                  f"{s['cached']:>10.0f}{s['new']:>13.0f}{s['frac']:>13.4f}"
                  f"{s['wall']:>10.1f}")

    # ---- positive control
    print("\n--- positive control: does Shuffled Full break the cache? ---")
    on_orig = summary.get(("on", "Original"))
    on_shuf = summary.get(("on", "Shuffled Full"))
    if on_orig and on_shuf:
        d_frac = (on_orig["frac"] - on_shuf["frac"]) * 100
        d_new = (on_shuf["new"] - on_orig["new"]) / max(on_orig["new"], 1)
        print(f"  cached_frac  Original {on_orig['frac']:.4f} -> Shuffled "
              f"{on_shuf['frac']:.4f}  ({-d_frac:+.1f} pp)")
        print(f"  new prefill  Original {on_orig['new']:.0f} -> Shuffled "
              f"{on_shuf['new']:.0f} tokens  ({d_new:+.1%})")
        verdict = ("PASS - the chain measures caching"
                   if d_frac > 5 else
                   "FAIL - shuffling changed nothing; do not trust any "
                   "conclusion from this run")
        print(f"  verdict: {verdict}")
    else:
        print("  cannot evaluate: missing an arm")

    # ---- stable vs original
    print("\n--- does stabilising the schema buy anything? ---")
    on_stab = summary.get(("on", "Stable Full"))
    if on_orig and on_stab:
        print(f"  cached_frac  {on_orig['frac']:.4f} -> {on_stab['frac']:.4f}  "
              f"({(on_stab['frac'] - on_orig['frac']) * 100:+.2f} pp)")
        print(f"  new prefill  {on_orig['new']:.0f} -> {on_stab['new']:.0f} tokens")

    # ---- offline vs measured
    print("\n--- offline block simulation vs measured (cache on) ---")
    print(f"  {'policy':<15}{'offline':>10}{'measured':>10}{'delta':>10}")
    for pol in POLICIES:
        s = summary.get(("on", pol))
        if s and pol in OFFLINE_REUSE:
            o = OFFLINE_REUSE[pol]
            print(f"  {pol:<15}{o:>10.4f}{s['frac']:>10.4f}{s['frac'] - o:>+10.4f}")

    # ---- cache off sanity
    print("\n--- cache OFF: policy differences should collapse ---")
    offs = [(p, summary[("off", p)]) for p in POLICIES if ("off", p) in summary]
    for p, s in offs:
        print(f"  {p:<15} cached_frac={s['frac']:.4f}  new_prefill={s['new']:.0f}")
    if len(offs) >= 2:
        spread = max(s["frac"] for _, s in offs) - min(s["frac"] for _, s in offs)
        print(f"  spread across policies: {spread:.4f} "
              f"({'collapsed as expected' if spread < 0.05 else 'NOT collapsed - investigate'})")


if __name__ == "__main__":
    main()
