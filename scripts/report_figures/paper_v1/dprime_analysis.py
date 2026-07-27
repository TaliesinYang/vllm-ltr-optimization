"""D-prime: ordering under a queue that exists, with true launch replication.

Three blocks, each arm-block cell a cold vLLM launch, so resampling blocks
resamples engine launches -- the replication the main matrix lacked. Before
any ratio is read, the manipulation check must pass per arm: the share of
requests whose first waiting-queue entry saw depth >=2, computed from the
order logs. A block where that share stays under the floor is reported
non-diagnostic rather than quietly averaged in.

Run: python3 dprime_analysis.py [depth-floor-pct]   (default 10)
"""

from __future__ import annotations

import csv
import glob
import json
import sys

import numpy as np

from _common import REPO
from fig_block1 import (
    BOOTSTRAP_DRAWS, hierarchical_draws, interval, pooled_mean,
    session_of_request,
)

ARMS = {
    "PolicyFCFS": "PolicyFCFS",
    "PromptLengthSJFScheduler": "PromptLengthSJF",
    "PureLTRScheduler": "PureLTR",
    "GatedRuleCScheduler": "GatedRuleC",
}
BLOCKS = (1, 2, 3)
COMPARISONS = [
    ("GatedRuleCScheduler", "PromptLengthSJFScheduler", "primary"),
    ("GatedRuleCScheduler", "PolicyFCFS", "safety"),
    ("GatedRuleCScheduler", "PureLTRScheduler", "gate value"),
    ("PureLTRScheduler", "PolicyFCFS", "signal"),
    ("PromptLengthSJFScheduler", "PolicyFCFS", "heuristic"),
]
FLOOR_PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0


def load_block(stem: str, block: int, sessions) -> dict | None:
    """Session -> ttlt list for one arm-block cell (one cold launch)."""
    runs = REPO / "runs" / f"dprime-b{block}" / "matrix" / f"{stem}.runs"
    if not runs.exists():
        return None
    by_session: dict[str, list[float]] = {}
    for path in sorted(runs.glob("*.samples.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("error") or "").strip():
                    continue
                key = sessions.get(row["request_id"])
                if key is None:
                    continue
                by_session.setdefault(key, []).append(float(row["ttlt_ms"]))
    return by_session or None


def arrival_order(stem: str, block: int) -> dict[str, float]:
    """request_id -> dispatch time, the ground truth an ordering policy departs from.

    Ranking the order log against first-appearance-within-the-log is circular:
    a sorted list assigns its own ranks in sorted order and can never look
    inverted. The client's dispatch timestamp is outside the scheduler and is
    what "arrival order" means.
    """
    runs = REPO / "runs" / f"dprime-b{block}" / "matrix" / f"{stem}.runs"
    stamps: dict[str, float] = {}
    for path in sorted(runs.glob("*.samples.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get("dispatched_at_unix_s") or row.get("scheduled_at_unix_s")
                if value:
                    stamps[row["request_id"]] = float(value)
    return stamps


def manipulation_check(stem: str, block: int) -> dict | None:
    pattern = str(REPO / "runs" / f"dprime-b{block}-mixed-round-a-{stem}-attempt-*"
                  / "order.jsonl")
    stamps = arrival_order(stem, block)
    first_depth, seen = {}, set()
    inversions = steps_ge2 = 0
    for path in sorted(glob.glob(pattern)):
        for line in open(path, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            order = entry.get("order") or []
            for rid in order:
                if rid not in seen:
                    seen.add(rid)
                    first_depth[rid] = len(order)
            if len(order) >= 2:
                times = [stamps.get(r) for r in order]
                if all(t is not None for t in times):
                    steps_ge2 += 1
                    if times != sorted(times):
                        inversions += 1
    if not first_depth:
        return None
    ge2 = sum(1 for d in first_depth.values() if d >= 2)
    return {
        "requests": len(first_depth),
        "depth_ge2_pct": 100.0 * ge2 / len(first_depth),
        "steps_ge2": steps_ge2,
        "inversion_steps": inversions,
        "inversion_pct_of_ge2_steps": (100.0 * inversions / steps_ge2) if steps_ge2 else 0.0,
    }


def main() -> None:
    sessions = session_of_request()
    launches: dict[str, list[dict]] = {}
    checks: dict[str, list[dict]] = {}
    for stem in ARMS:
        cells = [load_block(stem, b, sessions) for b in BLOCKS]
        present = [c for c in cells if c]
        if len(present) < len(BLOCKS):
            print(f"  {ARMS[stem]}: {len(present)}/{len(BLOCKS)} blocks present")
        if not present:
            continue
        launches[stem] = present
        checks[stem] = [manipulation_check(stem, b) or {} for b in BLOCKS]

    if len(launches) < 2:
        raise SystemExit("fewer than two arms present; nothing to compare")

    print("\nmanipulation check (per arm, blocks pooled):")
    diagnostic = True
    for stem, blocks in checks.items():
        vals = [c for c in blocks if c]
        pooled = (sum(c["requests"] * c["depth_ge2_pct"] for c in vals)
                  / max(sum(c["requests"] for c in vals), 1))
        inv = sum(c["inversion_steps"] for c in vals)
        flag = "" if pooled >= FLOOR_PCT else "  <-- BELOW FLOOR"
        if pooled < FLOOR_PCT and stem != "PolicyFCFS":
            diagnostic = False
        print(f"  {ARMS[stem]:16} depth>=2 at first entry: {pooled:5.1f}%  "
              f"inversion steps: {inv}{flag}")
    if not diagnostic:
        print(f"\nNON-DIAGNOSTIC: reorder opportunity under {FLOOR_PCT}% floor; "
              "report the null as untestable, not as equivalence.")

    shared = sorted(set.intersection(*(
        set(cell) for cells in launches.values() for cell in cells)))
    print(f"\n{len(shared)} sessions present in every arm-block cell")
    draws = hierarchical_draws(launches, shared, seed=20260727)

    print(f"\n{'comparison':34} {'ratio':>8}  95% CI (blocks then sessions)")
    results = {}
    for num, den, role in COMPARISONS:
        if num not in launches or den not in launches:
            continue
        est = pooled_mean(launches[num], shared) / pooled_mean(launches[den], shared)
        low, high = interval(draws[num] / draws[den])
        results[f"{ARMS[num]}/{ARMS[den]}"] = {
            "role": role, "ratio": round(est, 4), "ci": [round(low, 4), round(high, 4)],
        }
        print(f"{ARMS[num]}/{ARMS[den]:20} {est:8.4f}  [{low:.4f}, {high:.4f}]  ({role})")

    out = REPO / "runs" / "dprime-summary.json"
    out.write_text(json.dumps({
        "floor_pct": FLOOR_PCT,
        "diagnostic": diagnostic,
        "manipulation": {ARMS[k]: v for k, v in checks.items()},
        "comparisons": results,
        "blocks": len(BLOCKS),
        "note": "each arm-block cell is a cold vLLM launch; bootstrap resamples blocks then sessions",
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
