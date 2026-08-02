#!/usr/bin/env python3
"""Layout analysis: as-is vs hoisted schema, cross-session.

Reports pooled medians and the per-head paired comparison. The offline audit
predicted the effect flips sign with whether sessions share a toolset, so the
split is reported rather than only the pooled number.
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fnum(r, k):
    v = r.get(k, "")
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
    rows = [r for r in csv.DictReader(args.csv.open()) if not r["error"]]
    if not rows:
        raise SystemExit("no usable rows")

    runs = sorted({r["run_id"] for r in rows})
    print(f"measurements: {len(rows)}   server restarts: {runs}\n")

    by = collections.defaultdict(list)
    for r in rows:
        by[r["layout"]].append(r)

    print(f"{'layout':<10}{'n':>4}{'prompt_tok':>12}{'cached':>10}"
          f"{'new_prefill':>13}{'cached_frac':>13}{'wall_ms':>10}")
    print("-" * 72)
    S = {}
    for lay in ("as-is", "hoisted"):
        rs = by.get(lay, [])
        if not rs:
            continue
        S[lay] = {
            "n": len(rs),
            "prompt": med([fnum(r, "prompt_tokens") for r in rs]),
            "cached": med([fnum(r, "cached_tokens") for r in rs]),
            "new": med([fnum(r, "new_prefill_tokens") for r in rs]),
            "frac": med([fnum(r, "cached_frac") for r in rs]),
            "wall": med([fnum(r, "wall_ms") for r in rs]),
        }
        s = S[lay]
        print(f"{lay:<10}{s['n']:>4}{s['prompt']:>12.0f}{s['cached']:>10.0f}"
              f"{s['new']:>13.0f}{s['frac']:>13.4f}{s['wall']:>10.1f}")

    if len(S) < 2:
        print("\nonly one layout present")
        return

    a, h = S["as-is"], S["hoisted"]
    print(f"\npooled: cached_frac {a['frac']:.4f} -> {h['frac']:.4f} "
          f"({(h['frac'] - a['frac']) * 100:+.1f} pp)")
    print(f"        new prefill {a['new']:.0f} -> {h['new']:.0f} tokens "
          f"({(a['new'] - h['new']) / max(a['new'], 1):+.1%} saved)")
    print(f"        wall {a['wall']:.1f} -> {h['wall']:.1f} ms "
          f"({(a['wall'] - h['wall']) / max(a['wall'], 1):+.1%})")

    # paired by (run, head): the same request under both layouts
    keyed = collections.defaultdict(dict)
    for r in rows:
        keyed[(r["run_id"], r["head_index"])][r["layout"]] = r
    pairs = [v for v in keyed.values() if len(v) == 2]
    better = sum(1 for v in pairs
                 if fnum(v["hoisted"], "cached_frac") > fnum(v["as-is"], "cached_frac"))
    worse = sum(1 for v in pairs
                if fnum(v["hoisted"], "cached_frac") < fnum(v["as-is"], "cached_frac"))
    print(f"\npaired heads: {len(pairs)}   hoisted better {better}   "
          f"worse {worse}   tied {len(pairs) - better - worse}")

    # head 0 of each run is cold by construction; exclude it to see steady state
    warm = [v for v in pairs if v["as-is"]["head_index"] != "0"]
    if warm:
        wa = med([fnum(v["as-is"], "cached_frac") for v in warm])
        wh = med([fnum(v["hoisted"], "cached_frac") for v in warm])
        print(f"excluding each run's first (necessarily cold) head, n={len(warm)}: "
              f"{wa:.4f} -> {wh:.4f} ({(wh - wa) * 100:+.1f} pp)")

    # split by tool count: heads sharing the 10-tool set vs the others
    print("\nby tool-set size (the offline audit said the sign flips with sharing):")
    groups = collections.defaultdict(list)
    for v in pairs:
        groups[v["as-is"]["n_tools"]].append(v)
    for ntools, vs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ga = med([fnum(v["as-is"], "cached_frac") for v in vs])
        gh = med([fnum(v["hoisted"], "cached_frac") for v in vs])
        print(f"  {ntools:>3} tools  n={len(vs):>3}  {ga:.4f} -> {gh:.4f} "
              f"({(gh - ga) * 100:+.1f} pp)")


if __name__ == "__main__":
    main()
