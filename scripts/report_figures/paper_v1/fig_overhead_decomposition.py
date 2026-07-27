"""fig:decomposition -- what each layer of the synchronous path costs.

The earlier overhead figure compared two arms, direct and full gateway, and
could therefore only report one lumped number that a review correctly refused
to let us attribute to the Ranker. Six arms each add exactly one component:

    D0  direct to vLLM, no gateway          transport floor
    G0  gateway, no decision endpoint       + gateway proxying
    G1  gateway + stub decision service     + decision round-trip
    G2  gateway + CPU BERT, gate disabled   + model cost on every request
    G3  gateway + CPU BERT, gate-first      + gating (this is what saves)
    G4  gateway + GPU batched, gate-first   + GPU batching

so consecutive differences are per-component costs and the Ranker's own share
is finally separable from the hop that carries it.

Two properties of the run matter here and both are handled rather than
assumed. Arms run ABBA (D0..G4 then G4..D0), so a monotone drift across the
session cancels in the mean of each arm's two halves instead of being charged
to whichever arm ran late -- this script averages the halves and reports their
spread, because a large spread means the drift was not monotone and the
cancellation did not work. And under overload the guard sheds requests, so
mean latency is the wrong summary: a rejected request contributes no latency
and shedding would look like an improvement. Goodput is computed instead,
against the slowdown thresholds frozen in runs/slo-preregistration.json before
any of these arms ran.

Run: python3 fig_overhead_decomposition.py [run-tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from _common import COLOR, IEEE_DOUBLE_WIDTH, REPO, record_provenance, save

ARMS = [
    ("D0", "direct to vLLM"),
    ("G0", "+ gateway"),
    ("G1", "+ decision hop"),
    ("G2", "+ ranker, no gate"),
    ("G3", "+ gate-first"),
    ("G4", "+ GPU batching"),
]
PREREG = REPO / "runs" / "slo-preregistration.json"


def load_arm(run_dir: Path, arm: str) -> dict:
    """Both ABBA halves of one arm, or None if the run did not reach it."""
    halves = {}
    for half in ("forward", "reverse"):
        path = run_dir / "arms" / f"{arm}-{half}.json"
        if path.exists():
            halves[half] = json.loads(path.read_text())
    return halves


def samples_of(blob: dict) -> list[dict]:
    for key in ("gateway", "direct"):
        section = blob.get(key) or {}
        if section.get("samples"):
            return section["samples"]
    return []


def summarize(halves: dict, thresholds: dict) -> dict | None:
    """Mean TTLT and goodput per half, then the ABBA average and its spread."""
    per_half = {}
    for half, blob in halves.items():
        rows = samples_of(blob)
        served, shed = [], 0
        for row in rows:
            status = row.get("http_status")
            if status in (429, 503):
                shed += 1
                continue
            ttlt = row.get("ttlt_ms")
            base = row.get("baseline_service_ms") or 0
            if ttlt is None or base <= 0:
                continue
            served.append((float(ttlt), float(ttlt) / float(base)))
        if not served:
            continue
        offered = len(served) + shed
        wall = (blob.get("gateway") or blob.get("direct") or {}).get("wall_time_s") or 1.0
        per_half[half] = {
            "mean_ttlt_ms": float(np.mean([t for t, _ in served])),
            "shed": shed,
            "offered": offered,
            # Goodput counts a shed request as offered-and-failed: it must not
            # improve the metric by removing its own latency from the mean.
            "goodput_tight": sum(1 for _, s in served if s <= thresholds["tight"]) / wall,
            "goodput_lax": sum(1 for _, s in served if s <= thresholds["lax"]) / wall,
        }
    if not per_half:
        return None
    keys = ("mean_ttlt_ms", "goodput_tight", "goodput_lax")
    out = {k: float(np.mean([h[k] for h in per_half.values()])) for k in keys}
    out["halves"] = len(per_half)
    out["drift_spread_pct"] = (
        100 * abs(per_half["forward"]["mean_ttlt_ms"] - per_half["reverse"]["mean_ttlt_ms"])
        / out["mean_ttlt_ms"] if len(per_half) == 2 else float("nan"))
    out["shed"] = sum(h["shed"] for h in per_half.values())
    out["offered"] = sum(h["offered"] for h in per_half.values())
    return out


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "overload-block2"
    run_dir = REPO / "runs" / tag
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    thresholds = json.loads(PREREG.read_text())["slo_slowdown"]
    print(f"pre-registered slowdown thresholds: {thresholds}")

    rows = []
    for arm, label in ARMS:
        summary = summarize(load_arm(run_dir, arm), thresholds)
        if summary is None:
            print(f"  {arm}: absent, skipping")
            continue
        rows.append((arm, label, summary))
    if len(rows) < 2:
        raise SystemExit("fewer than two arms present; nothing to decompose")

    print(f"\n{'arm':4} {'label':20} {'mean TTLT':>10} {'goodput/s':>10} "
          f"{'shed':>6} {'drift%':>7}")
    for arm, label, s in rows:
        print(f"{arm:4} {label:20} {s['mean_ttlt_ms']:9.0f}ms {s['goodput_tight']:10.2f} "
              f"{s['shed']:6d} {s['drift_spread_pct']:6.1f}%")

    print("\nper-component cost (consecutive difference in mean TTLT):")
    for (a, la, sa), (b, lb, sb) in zip(rows, rows[1:]):
        delta = sb["mean_ttlt_ms"] - sa["mean_ttlt_ms"]
        print(f"  {a} -> {b:3} {lb:22} {delta:+9.0f} ms")

    fig, (ax_cost, ax_good) = plt.subplots(
        1, 2, figsize=(IEEE_DOUBLE_WIDTH, 2.8),
        gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)

    y = np.arange(len(rows))
    labels = [f"{a}  {la}" for a, la, _ in rows]
    means = [s["mean_ttlt_ms"] / 1000 for _, _, s in rows]
    ax_cost.barh(y, means, color=COLOR["neutral"], height=.6)
    ax_cost.set_yticks(y, labels)
    ax_cost.invert_yaxis()
    ax_cost.set_xlabel("Mean TTLT (s), ABBA-averaged")
    ax_cost.xaxis.grid(True, zorder=0)
    ax_cost.text(-0.42, 1.04, "(a)", transform=ax_cost.transAxes,
                 fontweight="bold", fontsize=10)

    width = .38
    ax_good.barh(y - width / 2, [s["goodput_tight"] for _, _, s in rows],
                 height=width, color=COLOR["prompt_schema"], label="tight SLO")
    ax_good.barh(y + width / 2, [s["goodput_lax"] for _, _, s in rows],
                 height=width, color=COLOR["neutral"], label="lax SLO")
    ax_good.set_yticks(y, ["" for _ in rows])
    ax_good.invert_yaxis()
    ax_good.set_xlabel("SLO-qualified goodput (req/s)")
    ax_good.xaxis.grid(True, zorder=0)
    ax_good.legend(loc="lower right")
    ax_good.text(-0.10, 1.04, "(b)", transform=ax_good.transAxes,
                 fontweight="bold", fontsize=10)

    save(fig, "decomposition.pdf")
    plt.close(fig)
    record_provenance("decomposition.pdf",
                      sorted((run_dir / "arms").glob("*.json")) + [PREREG])


if __name__ == "__main__":
    main()
