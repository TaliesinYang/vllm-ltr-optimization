#!/usr/bin/env python3
"""Go/No-Go evaluation + the two paired figures for the kill test."""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ["Original", "Stable Full", "Shuffled Full", "Frozen Thin"]
NUM = {"total_tokens", "schema_tokens", "byte_lcp", "token_lcp", "reusable_blocks",
       "reusable_block_tokens", "reuse_ratio", "earliest_mutation_token",
       "earliest_mutation_block", "invalidated_suffix_tokens", "turn_index",
       "n_tools_original", "n_tools_universe", "n_tools_thin"}


def load(path: Path):
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            for k in list(r):
                if k in NUM:
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def median(xs):
    xs = sorted(xs)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args()
    rows = load(args.results / "pair_metrics.csv")

    # index pairs so the same request pair can be compared across policies
    by_pair = collections.defaultdict(dict)
    for r in rows:
        by_pair[(r["session_id"], r["prev_request_id"], r["request_id"])][r["policy"]] = r
    pairs = list(by_pair.values())
    pairs = [p for p in pairs if all(pol in p for pol in POLICIES)]

    med = {pol: {m: median([p[pol][m] for p in pairs])
                 for m in ("reuse_ratio", "invalidated_suffix_tokens", "token_lcp",
                           "total_tokens", "schema_tokens", "reusable_blocks")}
           for pol in POLICIES}

    print(f"n_pairs = {len(pairs)}\n")
    for pol in POLICIES:
        m = med[pol]
        print(f"{pol:>14}  reuse={m['reuse_ratio']:.4f}  inval={m['invalidated_suffix_tokens']:.0f}"
              f"  lcp={m['token_lcp']:.0f}  total={m['total_tokens']:.0f}"
              f"  schema_tok={m['schema_tokens']:.0f}  blocks={m['reusable_blocks']:.0f}")

    seg = collections.Counter(p["Original"]["first_changed_segment"] for p in pairs)
    seg_shuf = collections.Counter(p["Shuffled Full"]["first_changed_segment"] for p in pairs)
    print(f"\nfirst_changed_segment (Original): {dict(seg)}")
    print(f"first_changed_segment (Shuffled): {dict(seg_shuf)}")

    d1 = (med["Stable Full"]["reuse_ratio"] - med["Original"]["reuse_ratio"]) * 100
    inv_s, inv_sh = (med["Stable Full"]["invalidated_suffix_tokens"],
                     med["Shuffled Full"]["invalidated_suffix_tokens"])
    d2 = (inv_sh - inv_s) / inv_sh * 100 if inv_sh else 0.0
    better = sum(1 for p in pairs
                 if p["Stable Full"]["reusable_blocks"] > p["Original"]["reusable_blocks"])
    mutation_earlier = sum(1 for p in pairs
                           if p["Shuffled Full"]["earliest_mutation_token"]
                           < p["Original"]["earliest_mutation_token"])
    pre_tools = sum(1 for p in pairs
                    if p["Original"]["first_changed_segment"] == "system")

    print("\n--- Go / No-Go criteria ---")
    print(f"1. Stable-vs-Original median reuse gain : {d1:+.2f} pp   (Go needs >= +20)")
    print(f"2. Stable-vs-Shuffled invalidated cut   : {d2:.1f} %     (Go needs >= 30)")
    print(f"3. Shuffle moves earliest mutation up   : {mutation_earlier}/{len(pairs)} pairs")
    print(f"4. Pairs where Stable > Original blocks : {better}/{len(pairs)} "
          f"({better / len(pairs) * 100:.0f} %, Go needs >= 70)")
    print(f"5. Change originates before tools       : {pre_tools}/{len(pairs)} pairs "
          f"in 'system'; {seg.get('tools', 0)} in 'tools'; {seg.get('messages', 0)} in 'messages'")

    figdir = args.results
    for metric, ylabel, fname, logy in [
        ("reuse_ratio", "Reusable full-KV-block ratio", "fig1_reuse_ratio.png", False),
        ("invalidated_suffix_tokens", "Invalidated suffix (tokens)",
         "fig2_invalidated_suffix.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        xs = range(len(POLICIES))
        for p in pairs:
            ax.plot(xs, [p[pol][metric] for pol in POLICIES],
                    color="0.6", lw=0.7, alpha=0.6, marker="o", ms=3, zorder=1)
        ax.plot(xs, [med[pol][metric] for pol in POLICIES],
                color="crimson", lw=2.0, marker="D", ms=6, zorder=3, label="median")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(POLICIES, rotation=12)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        ax.set_title(f"{len(pairs)} adjacent request pairs (paired)", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=200)
        plt.close(fig)
    print(f"\nfigures -> {figdir}/fig1_reuse_ratio.png, {figdir}/fig2_invalidated_suffix.png")


if __name__ == "__main__":
    main()
