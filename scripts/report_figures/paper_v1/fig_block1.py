"""fig:block1 -- serving-level scheduling results.

The statistics here are the ones the paper pre-registered before the data
existed, and they are not the ones the earlier rental figure used:

  * the primary endpoint is pooled MEAN time-to-last-token, not a percentile;
  * every arm replays the same sessions with the same arrival offsets, so the
    comparison is PAIRED at session level -- the numerator and denominator of
    each ratio are computed over the same resampled sessions, never over two
    independently drawn samples;
  * resampling is HIERARCHICAL, launches first and then sessions within the
    resampled launches, because requests inside one session are not
    independent draws and a request-level bootstrap would understate the
    interval;
  * safety against PolicyFCFS is judged against a pre-declared 3%
    non-inferiority margin. An interval containing 1.0 is uncertainty, not a
    demonstration of safety, and the figure draws the margin so a reader can
    see which of the two they are looking at.

The script refuses to draw a partial matrix. A missing arm is a missing
result, and a figure that quietly averages whatever happened to finish is
worse than no figure.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from _common import (
    COLOR, IEEE_DOUBLE_WIDTH, REPO, load_json, load_jsonl, record_provenance, save,
)

RUNS = REPO / "runs" / "block1-main"
WORKLOAD = REPO / "runs" / "block1-2026-07-26" / "workload-block1.jsonl"
QUEUE_DEPTH = REPO / "runs" / "queue-depth.json"
BOOTSTRAP_DRAWS = 5000
SAFETY_MARGIN = 1.03  # pre-declared non-inferiority margin on mean TTLT

# Directory stem -> (display label, is this ours). Order is the reading order
# of panel (b): baselines first, then the arms under test.
ARMS: dict[str, tuple[str, bool]] = {
    "stock_fcfs": ("Stock FCFS", False),
    "StockFCFSShim": ("Stock FCFS (shim)", False),
    "PolicyFCFS": ("PolicyFCFS", False),
    "PromptLengthSJFScheduler": ("PromptLengthSJF", False),
    "PureLTRScheduler": ("PureLTR", True),
    "GatedRuleCScheduler": ("GatedRuleC", True),
}

# Each comparison states what it is for, because a forest plot of unlabelled
# ratios invites the reader to treat an incidental contrast as a claim.
COMPARISONS: list[tuple[str, str, str, str]] = [
    ("GatedRuleCScheduler", "PromptLengthSJFScheduler", "primary", "superiority"),
    ("GatedRuleCScheduler", "PolicyFCFS", "safety", "non-inferiority"),
    # The gate's own hypothesis: selective trust beats blind trust. This is
    # the only comparison that isolates the gate, since the two arms share
    # the Ranker and differ solely in whether its score is acted on.
    ("GatedRuleCScheduler", "PureLTRScheduler", "gate value", ""),
    ("PureLTRScheduler", "PolicyFCFS", "secondary", ""),
    ("PromptLengthSJFScheduler", "PolicyFCFS", "secondary", ""),
    ("PolicyFCFS", "stock_fcfs", "attribution", ""),
]


# Directory stem -> arm key in queue-depth.json. Only the four policy arms log
# scheduling-step queue depth; the stock arms have no policy hook to instrument.
QUEUE_ARMS: dict[str, str] = {
    "PolicyFCFS": "PolicyFCFS",
    "PromptLengthSJFScheduler": "PromptLengthSJF",
    "PureLTRScheduler": "PureLTR",
    "GatedRuleCScheduler": "GatedRuleC",
}


def load_queue_depth() -> dict[str, dict]:
    """Per-arm waiting-queue depth stats, verified against the panel's claim.

    The panel prints a categorical sentence ("empty at p90; >=2 waiting in
    <1% of steps"), so the script refuses to draw it if the committed data
    ever stops supporting it.
    """
    stats = load_json(QUEUE_DEPTH)["arms"]
    missing = [key for key in QUEUE_ARMS.values() if key not in stats]
    if missing:
        raise SystemExit(f"queue-depth.json lacks arms: {', '.join(missing)}")
    for key in QUEUE_ARMS.values():
        if stats[key]["p90"] != 0 or stats[key]["ge2_pct"] >= 1.0:
            raise SystemExit(
                f"{key}: p90={stats[key]['p90']}, ge2={stats[key]['ge2_pct']}% "
                "contradicts the panel's printed claim; update the label."
            )
    return stats


def session_of_request() -> dict[str, str]:
    """request_id -> session_id, read from the workload the arms replayed."""
    return {row["request_id"]: row["session_id"] for row in load_jsonl(WORKLOAD)}


def load_arm(stem: str, sessions: dict[str, str]) -> list[dict[str, list[float]]]:
    """One entry per launch; each maps session_id -> that session's TTLTs.

    Failed requests are kept out of the latency vector but counted, because
    dropping them silently would let an arm look fast by failing.
    """
    directory = RUNS / "matrix" / f"{stem}.runs"
    files = sorted(directory.glob("*.samples.csv"))
    if not files:
        raise FileNotFoundError(f"no completed launches for {stem} under {directory}")
    launches = []
    for path in files:
        by_session: dict[str, list[float]] = {}
        errors = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("error") or "").strip():
                    errors += 1
                    continue
                session = sessions.get(row["request_id"])
                if session is None:
                    raise KeyError(f"{row['request_id']} is absent from the workload")
                by_session.setdefault(session, []).append(float(row["ttlt_ms"]))
        if errors:
            print(f"  note: {stem} {path.name} had {errors} failed requests", file=sys.stderr)
        launches.append(by_session)
    return launches


def pooled_mean(launches: list[dict[str, list[float]]], keys: list[str]) -> float:
    values = [v for launch in launches for key in keys for v in launch.get(key, ())]
    return float(np.mean(values)) if values else float("nan")


def hierarchical_draws(
    arms: dict[str, list[dict[str, list[float]]]],
    shared_sessions: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap every arm together, on one shared resampling of the design.

    Launch indices and session ids are drawn ONCE per replicate and applied to
    all arms, which is what makes the ratios paired: arm A and arm B are always
    scored on the same sessions in the same replicate.
    """
    rng = np.random.default_rng(seed)
    counts = {len(launches) for launches in arms.values()}
    if len(counts) != 1:
        raise ValueError(f"arms disagree on launch count: {counts}")
    n_launches = counts.pop()
    n_sessions = len(shared_sessions)
    sessions = np.asarray(shared_sessions, dtype=object)

    out = {stem: np.empty(BOOTSTRAP_DRAWS) for stem in arms}
    for draw in range(BOOTSTRAP_DRAWS):
        launch_idx = rng.integers(0, n_launches, size=n_launches)
        session_idx = rng.integers(0, n_sessions, size=n_sessions)
        keys = list(sessions[session_idx])
        for stem, launches in arms.items():
            resampled = [launches[i] for i in launch_idx]
            out[stem][draw] = pooled_mean(resampled, keys)
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def build_figure(arms, draws, shared_sessions, queue):
    point = {stem: pooled_mean(launches, shared_sessions) for stem, launches in arms.items()}
    fig, (ax_forest, ax_level, ax_queue) = plt.subplots(
        1, 3, figsize=(IEEE_DOUBLE_WIDTH, 3.05),
        gridspec_kw={"width_ratios": [1.6, 1.35, 1.5]}, constrained_layout=True,
    )

    # (a) paired ratios. A ratio below one means the numerator arm finished
    # requests faster on the same sessions.
    rows = list(reversed(COMPARISONS))
    y = np.arange(len(rows))
    ax_forest.axvspan(1.0, SAFETY_MARGIN, color=COLOR["neutral"], alpha=0.13, lw=0, zorder=0)
    ax_forest.axvline(1.0, color="#333333", lw=0.9, zorder=1)
    labels = []
    for i, (num, den, role, _) in enumerate(rows):
        ratios = draws[num] / draws[den]
        est = point[num] / point[den]
        low, high = interval(ratios)
        colour = COLOR["prompt_schema"] if role in ("primary", "safety") else COLOR["neutral"]
        ax_forest.plot([low, high], [i, i], color=colour, lw=1.5, solid_capstyle="butt", zorder=3)
        ax_forest.plot([est], [i], "o", color=colour, ms=4.6, zorder=4)
        labels.append(f"{ARMS[num][0]} /\n{ARMS[den][0]}")
        if role in ("primary", "safety"):
            # Above the point, not beside the whisker: text past the axes edge
            # makes constrained_layout reserve that width as decoration.
            ax_forest.annotate(role, (est, i), xytext=(0, 4), textcoords="offset points",
                               ha="center", va="bottom", fontsize=10, color=colour)
    ax_forest.set_yticks(y, labels, linespacing=0.9)
    ax_forest.set_ylim(-0.6, len(rows) + 0.25)
    ax_forest.set_xlabel("Paired ratio of mean TTLT (95% hierarchical CI)")
    ax_forest.xaxis.grid(True, zorder=0)
    ax_forest.text(SAFETY_MARGIN + 0.008, -0.5, f"{SAFETY_MARGIN:g} margin",
                   rotation=90, ha="left", va="bottom",
                   fontsize=10, color="#555555")
    ax_forest.text(-0.34, 1.04, "(a)", transform=ax_forest.transAxes,
                   fontweight="bold", fontsize=10)

    # (b) the levels the ratios are formed from, so a reader can see whether a
    # 2% difference sits on 3 s or on 30 s.
    stems = list(ARMS)
    yl = np.arange(len(stems))
    for i, stem in enumerate(stems):
        low, high = interval(draws[stem])
        ours = ARMS[stem][1]
        colour = COLOR["prompt_schema"] if ours else COLOR["neutral"]
        ax_level.plot([low / 1000, high / 1000], [i, i], color=colour, lw=1.5,
                      solid_capstyle="butt", zorder=3)
        ax_level.plot([point[stem] / 1000], [i], "o", color=colour, ms=4.6, zorder=4)
    ax_level.set_yticks(yl, [ARMS[stem][0] for stem in stems])
    ax_level.set_ylim(-0.6, len(stems) - 0.4)
    ax_level.set_xlabel("Pooled mean TTLT (s)")
    ax_level.xaxis.grid(True, zorder=0)
    # Colour carries arm ownership in all three panels; say so once, in the
    # panel whose upper-right corner is empty (the short policy-arm rows).
    from matplotlib.lines import Line2D

    ax_level.legend(handles=[
        Line2D([], [], color=COLOR["prompt_schema"], marker="o", ms=4.6,
               lw=1.5, label="ranker arms (ours)"),
        Line2D([], [], color=COLOR["neutral"], marker="o", ms=4.6,
               lw=1.5, label="baseline arms"),
    ], loc="upper right", fontsize=8, frameon=False, handlelength=1.3,
        borderaxespad=0.2, labelspacing=0.4)
    ax_level.text(-0.42, 1.04, "(b)", transform=ax_level.transAxes,
                  fontweight="bold", fontsize=10)

    # (c) whether the experiment could have detected an ordering effect at all:
    # the waiting-queue depth each policy saw at its scheduling steps. Rows sit
    # on panel (b)'s y grid, so each strip reads against the same arm label.
    for i, stem in enumerate(stems):
        key = QUEUE_ARMS.get(stem)
        if key is None:
            continue  # stock arms: no policy hook, hence no order log
        d = queue[key]
        colour = COLOR["prompt_schema"] if ARMS[stem][1] else COLOR["neutral"]
        ax_queue.plot([d["p50"]], [i], "o", color=colour, ms=4.6, zorder=4)
        ax_queue.plot([d["p90"]], [i], "o", mfc="none", mec=colour, ms=9.5,
                      mew=1.0, zorder=3)
        ax_queue.plot([d["p99"]], [i], "o", mfc="white", mec=colour, ms=4.6,
                      mew=1.2, zorder=4)
        ax_queue.text(3.3, i, f"{d['ge2_pct']:.1f}%", ha="right", va="center",
                      fontsize=10, color=colour)
    # Column footers sit in the band between the policy rows and the stock
    # rows, labelling the marks directly instead of through a legend.
    ax_queue.text(0.1, 1.35, "p50·p90", ha="center", va="center",
                  fontsize=10, color="#555555")
    ax_queue.text(1.0, 0.82, "p99", ha="center", va="center",
                  fontsize=10, color="#555555")
    ax_queue.text(3.3, 1.35, "steps ≥2", ha="right", va="center",
                  fontsize=10, color="#555555")
    ax_queue.text(3.3, 0.3, "stock arms:\nno reorder hook", ha="right",
                  va="center", fontsize=10, color="#999999")
    ax_queue.set_xlim(-0.5, 3.45)
    ax_queue.set_xticks([0, 1, 2])
    ax_queue.set_ylim(-0.6, len(stems) - 0.4)
    ax_queue.set_yticks([])
    ax_queue.set_xlabel("Waiting-queue depth")
    ax_queue.xaxis.grid(True, zorder=0)
    # The sentence panel (c) exists to license; guarded by load_queue_depth().
    ax_queue.set_title("queue the policies could\n"
                       "reorder: empty at p90;\n"
                       "≥2 waiting in <1% of steps",
                       fontsize=10, color="#333333", pad=13)
    ax_queue.text(-0.06, 1.04, "(c)", transform=ax_queue.transAxes,
                  fontweight="bold", fontsize=10)
    return fig, point


def main() -> None:
    sessions = session_of_request()
    missing = [stem for stem in ARMS if not (RUNS / "matrix" / f"{stem}.runs").exists()]
    if missing:
        raise SystemExit(
            "Block-1 matrix is incomplete; refusing to draw a partial result.\n"
            f"  missing arms: {', '.join(missing)}"
        )
    arms = {stem: load_arm(stem, sessions) for stem in ARMS}

    shared = set.intersection(*(set(launch) for launches in arms.values() for launch in launches))
    shared_sessions = sorted(shared)
    print(f"{len(shared_sessions)} sessions replayed by every arm")

    queue = load_queue_depth()
    draws = hierarchical_draws(arms, shared_sessions, seed=20260727)
    fig, point = build_figure(arms, draws, shared_sessions, queue)
    save(fig, "block1.pdf")
    plt.close(fig)

    record_provenance("block1.pdf", sorted(
        p for stem in ARMS for p in (RUNS / "matrix" / f"{stem}.runs").glob("*.samples.csv")
    ) + [WORKLOAD, QUEUE_DEPTH])

    # Printed so the prose in 06.evaluation.tex quotes the figure's own numbers.
    for stem in ARMS:
        low, high = interval(draws[stem])
        print(f"{ARMS[stem][0]:<20} mean_ms={point[stem]:8.1f}  CI=[{low:.1f}, {high:.1f}]")
    for num, den, role, test in COMPARISONS:
        ratios = draws[num] / draws[den]
        low, high = interval(ratios)
        est = point[num] / point[den]
        verdict = ""
        if test == "superiority":
            verdict = "PASS" if high < 1.0 else "FAIL"
        elif test == "non-inferiority":
            verdict = "PASS" if high < SAFETY_MARGIN else "FAIL"
        print(f"[{role:<11}] {ARMS[num][0]} / {ARMS[den][0]}: "
              f"{est:.4f} CI=[{low:.4f}, {high:.4f}] {verdict}")
    for stem, key in QUEUE_ARMS.items():
        d = queue[key]
        print(f"[queue      ] {ARMS[stem][0]}: p50={d['p50']} p90={d['p90']} "
              f"p99={d['p99']} ge2={d['ge2_pct']}% over {d['events']} steps")


if __name__ == "__main__":
    main()
