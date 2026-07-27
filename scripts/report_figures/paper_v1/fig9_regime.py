"""Figure 9 -- the regime flip: one engine knob decides the paper's answer.

THE CLAIM THIS FIGURE MAKES
    The same four scheduling policies, the same workload and the same arrival
    offsets produce OPPOSITE conclusions depending on a single engine setting,
    the running-batch cap. Left at vLLM's default the engine held 11-19
    requests in flight, no waiting queue ever formed, and every ordering
    comparison sat on 1.00 with an interval that crossed it. Capping the
    running batch at 16 puts 57-64% of requests behind another request, and
    the same four comparisons separate: none of their intervals cross 1.00.

WHY IT IS BUILT THE WAY IT IS
  * ONE x SCALE, TWO PANELS. Both panels are drawn on identical limits and
    identical ticks, so the reader's eye travels left-to-right on the same
    ruler twice and the spread is a change in POSITION, never a change in
    zoom. Nothing else in the figure differs between the panels: same four
    rows, same order, same colours, same band, same rule.
  * THE ROWS ARE COMPARISONS, and each is named by BOTH its arms, stacked as
    a fraction. Colour is assigned by the NUMERATOR arm, on the one EXION
    family ramp ordered by how much of the ordering decision that arm's
    ranker owns, so the colour of a row means the same thing in both panels.
  * THE CALLOUT CARRIES THE REGIME, not the header strip. Each panel states
    the cap, the occupancy and the queue that occupancy produced -- every one
    of them read from a committed artifact at build time. The header strip
    carries only a label, which is what a strip is for. It does NOT state how
    many intervals cross 1.00: the reader sees that against the rule, so the
    line was spent on plate height instead and the fact became a build
    assertion (assert_regime_flip), which is stronger than printing it.
  * WHAT IS NOT DRAWN. No absolute latencies: this figure is about whether a
    contrast exists, and putting seconds next to ratios would invite the
    reader to compare two regimes that ran at different offered rates. The
    rates differ by construction (the capped round had to be driven harder
    to fill 16 slots), so any absolute comparison across panels would be a
    confound the figure cannot resolve. The ratios are within-regime and
    paired, and those are the only quantities the two panels share.

HONESTY NOTES THAT SURVIVE INTO THE CAPTION
  * Panel (a) resamples three REPLAYS inside one engine launch (the round-A
    matrix holds three samples.csv per arm from a single vLLM attempt, and
    matrix-round-b is not read); panel (b) has ONE
    launch per arm, so its intervals resample sessions within that launch and
    carry no machine-level replication. The two panels' intervals are
    therefore not equally strong. main() prints that sentence as the caption
    qualifier; it is not set inside the artwork, where a full sentence in a
    band costs plate height and reads as filler.
  * Panel (a) is drawn through fig_block1's own loader, bootstrap and seed, so
    its four rows are bit-identical to the ratios that module computes. This
    figure is now the only one that DRAWS the default-cap ordering contrasts --
    block1.pdf kept the attribution control and the order logs and dropped its
    ratio panel -- so these rows must not be able to disagree with the numbers
    the prose quotes from that module's stdout.
  * The 256-slot figure for panel (a) is the engine default: the round-A
    launch never set max_num_seqs, which the build asserts by reading the
    engine-arg dump out of both regimes' vLLM logs.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _common import IEEE_DOUBLE_WIDTH, REPO, load_json, record_provenance, save
from style import EXION  # noqa: E402  (_common puts report_figures on sys.path)

import fig_block1 as block1  # the round-A loader; never re-derived here

RUNS = REPO / "runs"
UNCONTENDED = RUNS / "block1-main"
CONTENDED = RUNS / "capped-batch"
QUEUE_DEPTH = RUNS / "queue-depth.json"
REORDER = RUNS / "reorder-opportunity.json"
CAPPED_SUMMARY = RUNS / "capped-batch-summary.json"
UNCONTENDED_LOG = next(
    (UNCONTENDED / "vllm-evidence").glob("*mixed-round-a-PolicyFCFS-attempt-*/vllm.log")
)
CONTENDED_LOG = next(
    (CONTENDED / "vllm-evidence").glob("*mixed-round-a-PolicyFCFS-attempt-*/vllm.log")
)

# Short arm name (how the capped summary keys its arms) -> matrix directory
# stem (how the round-A matrix keys its arms). One map, so a row is named once.
STEM = {
    "PolicyFCFS": "PolicyFCFS",
    "PromptLengthSJF": "PromptLengthSJFScheduler",
    "PureLTR": "PureLTRScheduler",
    "GatedRuleC": "GatedRuleCScheduler",
}

# Reading order, top to bottom, identical in both panels.
ROWS: list[tuple[str, str]] = [
    ("GatedRuleC", "PromptLengthSJF"),
    ("GatedRuleC", "PolicyFCFS"),
    ("PureLTR", "PolicyFCFS"),
    ("PromptLengthSJF", "PolicyFCFS"),
]

# One EXION family ramp, light -> dark by how much of the ordering decision
# that arm's ranker owns: arrival order -> a scalar heuristic -> the learned
# score -> the learned score behind a gate. A row takes the colour of its
# NUMERATOR, and the mapping is shared by both panels.
ARM_COLOR = {
    "PolicyFCFS": EXION["family"][0],
    "PromptLengthSJF": EXION["family"][1],
    "PureLTR": EXION["family"][2],
    "GatedRuleC": EXION["family"][3],
}
NUMERATORS = ["PromptLengthSJF", "PureLTR", "GatedRuleC"]

TEXT = "#333333"
FRAME = EXION["structure"][3]
STRIP_FILL = EXION["structure"][0]
CALLOUT_FILL = EXION["structure"][2]
BAND_FILL = EXION["structure"][0]
LEADER = EXION["structure"][1]
MUTED = EXION["structure"][3]

BOOTSTRAP_SEED = 20260727

# ---------------------------------------------------------------------------
# Geometry, in inches. Spelled out rather than left to a layout engine: the
# two panels have to sit on ONE x scale with pixel-identical axis rectangles,
# which no automatic layout guarantees.
FIG_W = IEEE_DOUBLE_WIDTH
FIG_H = 2.73
MARGIN = 0.10
CONTENT_L = MARGIN
CONTENT_R = FIG_W - MARGIN

TEXT_INSET = 0.06
LABEL_GUTTER = 0.07
STRIP_H = 0.19
STRIP_GAP = 0.06
CALLOUT_PAD_X = 0.07
CALLOUT_PAD_Y = 0.055
CALLOUT_LINE = 0.115
# Two lines, and the box is sized for exactly two: the regime (cap plus
# occupancy) and the queue that regime produces. A third line would have to
# earn a taller plate, so callout() refuses one rather than overflowing.
CALLOUT_LINES = 2

MARKER_PT = 3.6
CAP_HALF_IN = 0.048
LEADER_CLEAR_IN = 0.055
LEADER_MIN_IN = 0.18
VALUE_GAP_IN = 0.055   # gap between the upper interval cap and its printed value

# Three tiers, no others.
SIZE_AXIS = 10
SIZE_ANNOT = 9
SIZE_DENSE = 8
BOLD_FAMILY = ["DejaVu Sans"]

BLOCK_GAP = 0.12
BLOCK_W = (CONTENT_R - CONTENT_L - BLOCK_GAP) / 2
BLOCK_A = (CONTENT_L, CONTENT_L + BLOCK_W)
BLOCK_B = (BLOCK_A[1] + BLOCK_GAP, CONTENT_R)

LABEL_W = 1.34            # the fraction column: "vs PromptLengthSJF" is widest
AX_W = BLOCK_W - LABEL_W - LABEL_GUTTER - 0.02

STRIP_TOP = FIG_H - MARGIN
CALLOUT_TOP = STRIP_TOP - STRIP_H - STRIP_GAP
CALLOUT_H = CALLOUT_LINES * CALLOUT_LINE + 2 * CALLOUT_PAD_Y
AX_T = CALLOUT_TOP - CALLOUT_H - 0.08
AX_H = 1.16
AX_B = AX_T - AX_H
XLABEL_TOP = AX_B - 0.21
LEGEND_BOX = (CONTENT_L, MARGIN, CONTENT_R - CONTENT_L, 0.24)

# The double-column height cap (FIGURE-SPEC.md §1), enforced where the layout
# is decided rather than discovered in the PDF. The saved page is the canvas
# plus one pad on each side, because savefig runs with a tight bbox and this
# figure paints a full-canvas spacer.
PLATE_H = FIG_H + 2 * plt.rcParams["savefig.pad_inches"]
if PLATE_H > 2.80:
    raise SystemExit(f"regime plate is {PLATE_H:.2f}in tall; the cap is 2.80in")

AX_A = (BLOCK_A[0] + LABEL_W + LABEL_GUTTER, AX_B, AX_W, AX_H)
AX_BB = (BLOCK_B[0] + LABEL_W + LABEL_GUTTER, AX_B, AX_W, AX_H)

XLIM = (0.925, 1.175)
XTICKS = [0.95, 1.00, 1.05, 1.10, 1.15]
YLIM = (-0.65, 3.65)
LABEL_LINE = 0.105        # baseline pitch of the two-line fraction label


# ---------------------------------------------------------------------------
# Reading the artifacts. Nothing below returns a literal that a rebuilt run
# could contradict.
def matrix_arm_means(matrix: Path) -> dict[str, float]:
    """stem -> that arm's mean TTLT in seconds, averaged over its launches."""
    out: dict[str, float] = {}
    for path in sorted(matrix.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        values = [
            run["metrics"]["mean_ttlt_ms"] / 1000.0
            for scenario in record["scenarios"]
            for run in scenario["runs"]
            if run["status"] == "complete"
        ]
        if not values:
            raise SystemExit(f"{path} holds no completed launch")
        out[path.stem] = float(np.mean(values))
    return out


def launches_per_arm(matrix: Path) -> int:
    """Cold launches behind every arm of a regime; disagreement is a defect."""
    counts = {len(list((matrix / f"{stem}.runs").glob("*.samples.csv")))
              for stem in STEM.values()}
    if len(counts) != 1 or counts == {0}:
        raise SystemExit(f"{matrix} arms disagree on launch count: {counts}")
    return counts.pop()


def offered_rps(matrix: Path) -> float:
    """The one offered arrival rate every arm in a regime was driven at."""
    rates = set()
    for path in sorted(matrix.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for scenario in record["scenarios"]:
            rates.add(round(record["capacity_rps"] * scenario["scenario"]["saturation"], 6))
    if len(rates) != 1:
        raise SystemExit(f"{matrix} mixes offered rates {sorted(rates)}")
    return rates.pop()


def in_flight_range(matrix: Path) -> tuple[int, int]:
    """Little's law occupancy per arm: offered rate x mean time in system.

    This is the quantity the capped round was designed around, so it is
    derived the same way here rather than read off a prose sentence.
    """
    rate = offered_rps(matrix)
    counts = [rate * seconds for seconds in matrix_arm_means(matrix).values()]
    return round(min(counts)), round(max(counts))


def engine_cap(log: Path) -> int | None:
    """max_num_seqs as the engine actually recorded it, or None if unset."""
    text = log.read_text(encoding="utf-8", errors="ignore")
    found = {int(value) for value in re.findall(r"'max_num_seqs':\s*(\d+)", text)}
    if len(found) > 1:
        raise SystemExit(f"{log} records more than one running-batch cap: {found}")
    return found.pop() if found else None


def declared_caps() -> tuple[int, int]:
    """(default cap, capped cap), read from the round's own summary.

    The capped round set the cap explicitly and the engine logged it, so that
    value is cross-checked against the log. The default round set nothing, so
    its cap is the engine default, which only the summary records -- and the
    build asserts that the round-A log really did leave it unset before it is
    allowed to print the default.
    """
    summary = load_json(CAPPED_SUMMARY)
    default = int(re.search(r"(\d+)-slot running batch", summary["why"]).group(1))
    capped = int(re.search(r"VLLM_MAX_NUM_SEQS=(\d+)", summary["source"]).group(1))
    if engine_cap(CONTENDED_LOG) != capped:
        raise SystemExit(
            f"capped round declares max_num_seqs={capped} but its engine logged "
            f"{engine_cap(CONTENDED_LOG)}"
        )
    if engine_cap(UNCONTENDED_LOG) is not None:
        raise SystemExit(
            "the default round DID set max_num_seqs; it cannot be described as "
            "running at the engine default"
        )
    if default <= capped:
        raise SystemExit(f"default cap {default} is not above the capped cap {capped}")
    return default, capped


def span_pct(values: list[float]) -> tuple[int, int]:
    """Widest whole-percent interval that contains every arm's percentage."""
    return math.floor(min(values)), math.ceil(max(values))


def uncontended_queue() -> dict[str, object]:
    """Queue evidence for the default-cap regime, cross-checked across two files.

    queue-depth.json and reorder-opportunity.json were written by separate
    passes over the same order logs and both carry the per-request exposure,
    so disagreement between them is a real defect rather than a rounding
    nuisance; the build refuses to draw if they disagree.
    """
    depth = load_json(QUEUE_DEPTH)
    reorder = load_json(REORDER)["rounds"]["round_a"]
    request_level = depth["request_level"]["arms"]
    behind = []
    for short, stem in STEM.items():
        key = stem if stem in request_level else short
        mine = request_level[key]["first_entry_depth_ge2_pct"]
        theirs = reorder[stem]["depth_ge2_pct"]
        if abs(mine - theirs) > 0.05:
            raise SystemExit(
                f"{stem}: queue-depth says {mine}% of requests enter behind "
                f"another, reorder-opportunity says {theirs}%"
            )
        behind.append(mine)
    p90 = [depth["arms"][short]["p90"] for short in STEM]
    return {
        "p90_max": max(p90),
        "behind": span_pct(behind),
        "carryover": (
            min(reorder[stem]["steps_with_two_waiting_carryover"] for stem in STEM.values()),
            max(reorder[stem]["steps_with_two_waiting_carryover"] for stem in STEM.values()),
        ),
    }


def contended_queue(summary: dict) -> dict[str, object]:
    check = summary["manipulation_check"]
    return {
        "p90": (
            min(check[stem]["queue_p90"] for stem in STEM.values()),
            max(check[stem]["queue_p90"] for stem in STEM.values()),
        ),
        "behind": span_pct([check[stem]["depth_ge2_at_entry_pct"] for stem in STEM.values()]),
        "carryover": (
            min(check[stem]["carryover_steps"] for stem in STEM.values()),
            max(check[stem]["carryover_steps"] for stem in STEM.values()),
        ),
    }


def pooled_ttlt_from_samples(matrix: Path, stem: str) -> float:
    """Mean TTLT in seconds over every completed request of an arm."""
    values: list[float] = []
    for path in sorted((matrix / f"{stem}.runs").glob("*.samples.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("error") or "").strip():
                    continue
                values.append(float(row["ttlt_ms"]))
    if not values:
        raise SystemExit(f"no completed samples for {stem} under {matrix}")
    return float(np.mean(values)) / 1000.0


def contended_ratios(summary: dict) -> dict[tuple[str, str], tuple[float, float, float]]:
    """The capped round's published ratios, verified against its raw samples.

    The summary is the committed result, so it is what the figure draws; the
    raw per-request rows are re-read only to prove the summary still describes
    them. A summary that has drifted from its own samples stops the build.
    """
    means = summary["pooled_mean_ttlt_s"]
    for short, stem in STEM.items():
        raw = pooled_ttlt_from_samples(CONTENDED / "matrix", stem)
        if abs(raw - means[short]) / means[short] > 0.005:
            raise SystemExit(
                f"{short}: summary says mean TTLT {means[short]:.4f}s, its own "
                f"samples say {raw:.4f}s"
            )
    out = {}
    for num, den in ROWS:
        entry = summary["comparisons"][f"{num}/{den}"]
        implied = means[num] / means[den]
        if abs(implied - entry["ratio"]) / entry["ratio"] > 0.002:
            raise SystemExit(
                f"{num}/{den}: published ratio {entry['ratio']} disagrees with "
                f"the pooled means it should come from ({implied:.4f})"
            )
        low, high = entry["ci"]
        out[(num, den)] = (entry["ratio"], float(low), float(high))
    return out


def uncontended_ratios() -> tuple[dict[tuple[str, str], tuple[float, float, float]], int]:
    """Round-A ratios, through fig_block1's loader and its paired bootstrap."""
    sessions = block1.session_of_request()
    arms = {stem: block1.load_arm(stem, sessions) for stem in STEM.values()}
    shared = sorted(set.intersection(
        *(set(launch) for launches in arms.values() for launch in launches)
    ))
    draws = block1.hierarchical_draws(arms, shared, seed=BOOTSTRAP_SEED)
    point = {stem: block1.pooled_mean(launches, shared) for stem, launches in arms.items()}
    out = {}
    for num, den in ROWS:
        low, high = block1.interval(draws[STEM[num]] / draws[STEM[den]])
        out[(num, den)] = (point[STEM[num]] / point[STEM[den]], low, high)
    return out, len(shared)


def span(low, high) -> str:
    """A range, written as one number when both ends agree ('0', not '0-0')."""
    return f"{low}" if low == high else f"{low}-{high}"


def crossing(ratios: dict[tuple[str, str], tuple[float, float, float]]) -> int:
    """How many of a panel's intervals contain 1.00."""
    return sum(1 for _, low, high in ratios.values() if low <= 1.0 <= high)


def assert_regime_flip(uncontended, contended) -> None:
    """The figure's whole claim, checked rather than captioned.

    The panels used to print their own verdict ("All 4 intervals cross 1.00"),
    which restated what a reader sees in the artwork and cost a callout line.
    Dropping the line drops a self-check with it, so the check moves here: if a
    rebuilt run ever stopped flipping, this figure would still LOOK like the
    flip and no line inside it would contradict that. It stops the build now.
    """
    if crossing(uncontended) != len(uncontended):
        raise SystemExit(
            f"default cap: {crossing(uncontended)} of {len(uncontended)} "
            "intervals cross 1.00; the panel no longer shows a null result"
        )
    if crossing(contended) != 0:
        raise SystemExit(
            f"reduced cap: {crossing(contended)} of {len(contended)} intervals "
            "cross 1.00; the panel no longer shows four separated comparisons"
        )


# ---------------------------------------------------------------------------
# Drawing primitives. Sized in inches so the two panels are identical by
# construction rather than by two numbers that happen to agree.
def rect(spec: tuple[float, float, float, float]) -> list[float]:
    left, bottom, width, height = spec
    return [left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H]


def row_y(spec, row: float) -> float:
    _left, bottom, _width, height = spec
    frac = (row - YLIM[0]) / (YLIM[1] - YLIM[0])
    return (bottom + height * frac) / FIG_H


def x_per_inch(spec) -> float:
    return spec[2] / (XLIM[1] - XLIM[0])


def cap_units(spec) -> float:
    return CAP_HALF_IN / (spec[3] / (YLIM[1] - YLIM[0]))


def text_width(fig, string: str, fontsize: float, weight: str = "normal") -> float:
    family = BOLD_FAMILY if weight == "bold" else None
    probe = fig.text(0, 0, string, fontsize=fontsize, fontweight=weight,
                     fontfamily=family)
    width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    probe.remove()
    return width


def check_fits(fig, string: str, fontsize: float, budget: float, where: str,
               weight: str = "normal") -> None:
    """Shorten the wording, never shrink the glyphs -- so overflow is a build
    error rather than a silent drop below the smallest permitted size."""
    width = text_width(fig, string, fontsize, weight)
    if width > budget:
        raise SystemExit(
            f"{where}: {width:.2f}in of text in a {budget:.2f}in slot; shorten "
            f"the wording rather than shrinking it -- {string!r}"
        )


def full_canvas_spacer(fig) -> None:
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                             facecolor="none", edgecolor="none", linewidth=0,
                             zorder=-10))


def header_strip(fig, block: tuple[float, float], text: str) -> None:
    """A strip carries a LABEL: at most four words, no numbers, no verdict."""
    words = [word for word in text.split() if not word.startswith("(")]
    if len(words) > 4 or any(character.isdigit() for character in text[3:]):
        raise SystemExit(f"header strip must be a short label -- {text!r}")
    x0, x1 = block
    y0 = (STRIP_TOP - STRIP_H) / FIG_H
    height = STRIP_H / FIG_H
    fig.add_artist(Rectangle((x0 / FIG_W, y0), (x1 - x0) / FIG_W, height,
                             transform=fig.transFigure, facecolor=STRIP_FILL,
                             edgecolor=FRAME, linewidth=0.6, zorder=5))
    check_fits(fig, text, SIZE_ANNOT, (x1 - x0) - 2 * TEXT_INSET, "header strip",
               weight="bold")
    fig.text((x0 + TEXT_INSET) / FIG_W, y0 + height / 2, text, ha="left",
             va="center", fontsize=SIZE_ANNOT, fontweight="bold",
             fontfamily=BOLD_FAMILY, color="#222222", zorder=6)


def callout(fig, block: tuple[float, float], lines: list[str], where: str) -> None:
    """Framed result box under the strip: the regime's evidence, stated once."""
    x0, x1 = block
    if len(lines) != CALLOUT_LINES:
        raise SystemExit(
            f"{where}: {len(lines)} lines in a box sized for {CALLOUT_LINES}"
        )
    budget = (x1 - x0) - 2 * CALLOUT_PAD_X
    for line in lines:
        check_fits(fig, line, SIZE_DENSE, budget, where)
    y0 = CALLOUT_TOP - CALLOUT_H
    fig.add_artist(Rectangle((x0 / FIG_W, y0 / FIG_H), (x1 - x0) / FIG_W,
                             CALLOUT_H / FIG_H, transform=fig.transFigure,
                             facecolor=CALLOUT_FILL, edgecolor=FRAME,
                             linewidth=0.6, zorder=7))
    for index, line in enumerate(lines):
        y = CALLOUT_TOP - CALLOUT_PAD_Y - (index + 0.5) * CALLOUT_LINE
        fig.text((x0 + CALLOUT_PAD_X) / FIG_W, y / FIG_H, line, ha="left",
                 va="center", fontsize=SIZE_DENSE, color="#222222", zorder=8)


def bare_axis(ax) -> None:
    left_pad = XTICKS[0] - XLIM[0]
    right_pad = XLIM[1] - XTICKS[-1]
    if abs(left_pad - right_pad) > 1e-9:
        raise SystemExit("pad both ends of the shared axis alike")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS, [f"{value:.2f}" for value in XTICKS])
    ax.tick_params(axis="x", labelsize=SIZE_ANNOT, direction="out", length=2.5, pad=2)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)


def draw_panel(fig, spec, ratios, margin: float) -> None:
    """Four comparison rows on the shared ratio scale."""
    ax = fig.add_axes(rect(spec))
    bare_axis(ax)
    scale = x_per_inch(spec)
    cap = cap_units(spec)

    ax.axvspan(1.0 - margin, 1.0 + margin, facecolor=BAND_FILL, edgecolor="none",
               zorder=0)
    ax.axvline(1.0, color=MUTED, linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=1)

    for index, (num, den) in enumerate(ROWS):
        row = len(ROWS) - 1 - index
        est, low, high = ratios[(num, den)]
        colour = ARM_COLOR[num]
        stop = low - LEADER_CLEAR_IN / scale
        if (stop - XLIM[0]) * scale >= LEADER_MIN_IN:
            ax.plot([XLIM[0], stop], [row, row], color=LEADER, lw=0.5,
                    linestyle=(0, (1, 2)), zorder=0)
        ax.plot([est], [row], "o", color=colour, ms=MARKER_PT, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot([low, high], [row, row], color=colour, lw=1.3,
                solid_capstyle="butt", zorder=5)
        for end in (low, high):
            ax.plot([end, end], [row - cap, row + cap], color=colour, lw=1.0,
                    solid_capstyle="butt", zorder=6)
        value = f"{est:.3f}"
        check_fits(fig, value, SIZE_DENSE,
                   (XLIM[1] - high) * scale - VALUE_GAP_IN, "row value")
        ax.text(high + VALUE_GAP_IN / scale, row, value, ha="left", va="center",
                fontsize=SIZE_DENSE, color=TEXT, zorder=6)


def row_labels(fig, spec, block: tuple[float, float]) -> None:
    """The comparison names, stacked as a fraction, right-aligned on the axis.

    The numerator takes the row's colour and the denominator a structure grey,
    which is what makes 'colour means numerator' legible without a key that
    has to be read first.
    """
    x_right = (spec[0] - LABEL_GUTTER)
    budget = x_right - block[0]
    for index, (num, den) in enumerate(ROWS):
        row = len(ROWS) - 1 - index
        y = row_y(spec, row) * FIG_H
        for offset, (text, colour, weight) in enumerate((
            (num, ARM_COLOR[num], "bold"),
            (f"vs {den}", MUTED, "normal"),
        )):
            check_fits(fig, text, SIZE_DENSE, budget, "row label", weight=weight)
            fig.text(x_right / FIG_W,
                     (y + LABEL_LINE * (0.5 - offset)) / FIG_H,
                     text, ha="right", va="center", fontsize=SIZE_DENSE,
                     color=colour, fontweight=weight,
                     fontfamily=BOLD_FAMILY if weight == "bold" else None,
                     zorder=6)


def axis_label(fig, text: str) -> None:
    """One label for one shared scale, centred on the figure, not on a panel."""
    width = text_width(fig, text, SIZE_AXIS)
    if width > CONTENT_R - CONTENT_L:
        raise SystemExit(f"shared axis label does not fit: {text!r}")
    fig.text(0.5, XLABEL_TOP / FIG_H, text, ha="center", va="top",
             fontsize=SIZE_AXIS, color=TEXT, zorder=6)


def legend_band(fig, margin: float) -> None:
    """One boxed key, one row, outside every axes, at the foot of the plate.

    The band used to carry a second row saying that the two panels' intervals
    are not equally strong. That sentence still has to be read before the
    figure is believed, so it did not disappear: main() prints it as the
    caption's required qualifier and the LaTeX caption carries it, which is
    where a sentence belongs.
    """
    left, bottom, width, height = LEGEND_BOX
    fig.add_artist(Rectangle((left / FIG_W, bottom / FIG_H), width / FIG_W,
                             height / FIG_H, transform=fig.transFigure,
                             facecolor="white", edgecolor=FRAME, linewidth=0.6,
                             zorder=5))
    chip, chip_gap, pad = 0.085, 0.018, 0.06
    mark_w = 3 * chip + 2 * chip_gap
    entries = [
        ("ci", "Ratio, 95% interval"),
        ("rule", "Equal at 1.00"),
        ("band", f"±{margin * 100:.0f}% equivalence band"),
        ("ramp", "Colour: numerator arm"),
    ]
    columns = len(entries)
    col_w = (width - 2 * TEXT_INSET) / columns
    y = bottom + height * 0.5
    for column, (kind, label) in enumerate(entries):
        x = left + TEXT_INSET + column * col_w
        check_fits(fig, label, SIZE_DENSE, col_w - mark_w - pad, f"legend {column + 1}")
        if kind == "ci":
            fig.add_artist(Line2D([x / FIG_W, (x + mark_w) / FIG_W],
                                  [y / FIG_H, y / FIG_H], transform=fig.transFigure,
                                  color=TEXT, linewidth=1.3, marker="|",
                                  markersize=5.0, zorder=6))
            fig.add_artist(Line2D([(x + mark_w / 2) / FIG_W], [y / FIG_H],
                                  transform=fig.transFigure, color=TEXT, marker="o",
                                  markersize=MARKER_PT, linestyle="none",
                                  markeredgecolor="white", markeredgewidth=0.8,
                                  zorder=7))
        elif kind == "rule":
            fig.add_artist(Line2D([x / FIG_W, (x + mark_w) / FIG_W],
                                  [y / FIG_H, y / FIG_H], transform=fig.transFigure,
                                  color=MUTED, linewidth=0.9,
                                  linestyle=(0, (4, 2.5)), zorder=6))
        elif kind == "band":
            fig.add_artist(Rectangle((x / FIG_W, (y - 0.055) / FIG_H),
                                     mark_w / FIG_W, 0.11 / FIG_H,
                                     transform=fig.transFigure, facecolor=BAND_FILL,
                                     edgecolor=FRAME, linewidth=0.5, zorder=6))
        else:
            for index, arm in enumerate(NUMERATORS):
                fig.add_artist(Rectangle(
                    ((x + index * (chip + chip_gap)) / FIG_W, (y - chip / 2) / FIG_H),
                    chip / FIG_W, chip / FIG_H, transform=fig.transFigure,
                    facecolor=ARM_COLOR[arm], edgecolor="none", zorder=6))
        fig.text((x + mark_w + pad) / FIG_W, y / FIG_H, label, ha="left",
                 va="center", fontsize=SIZE_DENSE, color=TEXT, zorder=6)


# ---------------------------------------------------------------------------
def build_figure(uncontended, contended, facts):
    margin = block1.SAFETY_MARGIN - 1.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    full_canvas_spacer(fig)

    header_strip(fig, BLOCK_A, "(a) Default batch cap")
    header_strip(fig, BLOCK_B, "(b) Reduced batch cap")
    callout(fig, BLOCK_A, facts["a"], "panel (a) callout")
    callout(fig, BLOCK_B, facts["b"], "panel (b) callout")

    for spec, block, ratios in ((AX_A, BLOCK_A, uncontended), (AX_BB, BLOCK_B, contended)):
        draw_panel(fig, spec, ratios, margin)
        row_labels(fig, spec, block)

    axis_label(fig, "Paired ratio of mean TTLT (<1: numerator faster)")
    legend_band(fig, margin)
    return fig


def main() -> None:
    summary = load_json(CAPPED_SUMMARY)
    default_cap, capped_cap = declared_caps()

    uncontended, sessions = uncontended_ratios()
    contended = contended_ratios(summary)

    a_flight = in_flight_range(UNCONTENDED / "matrix")
    b_flight = in_flight_range(CONTENDED / "matrix")
    a_queue = uncontended_queue()
    b_queue = contended_queue(summary)

    if a_queue["p90_max"] != 0:
        raise SystemExit(
            "the default-cap round no longer has an empty queue at p90; the "
            "panel (a) callout would be false"
        )

    assert_regime_flip(uncontended, contended)

    # Two lines per panel: the regime the engine was in, and the queue that
    # regime produced. Whether the intervals cross 1.00 is left to the artwork,
    # which shows it against the rule; assert_regime_flip guards it instead.
    facts = {
        "a": [
            f"Batch cap {default_cap}, {span(*a_flight)} in flight",
            f"Waiting queue empty at p90; {span(*a_queue['behind'])}% queue on entry",
        ],
        "b": [
            f"Batch cap {capped_cap}, {span(*b_flight)} in flight",
            f"Waiting queue {span(*b_queue['p90'])} deep at p90; "
            f"{span(*b_queue['behind'])}% queue on entry",
        ],
    }

    if summary["sessions"] != sessions:
        raise SystemExit(
            f"the two regimes replayed different session sets ({sessions} vs "
            f"{summary['sessions']}); the panels are not the same workload"
        )
    launches_a = launches_per_arm(UNCONTENDED / "matrix")
    launches_b = launches_per_arm(CONTENDED / "matrix")
    # Both regimes resample WITHIN a single engine launch: round A's three
    # files are repeats inside one vLLM attempt, not three attempts. Saying
    # otherwise would claim a machine-level replication neither panel has.
    caveat = (
        f"Same {sessions} sessions in both panels; each arm resamples within "
        f"one engine launch ({launches_a} replays in (a), {launches_b} in (b)), "
        f"so neither panel's intervals carry machine-level replication."
    )

    fig = build_figure(uncontended, contended, facts)
    save(fig, "regime.pdf")
    plt.close(fig)

    record_provenance("regime.pdf", sorted(
        list((UNCONTENDED / "matrix").glob("*/*.samples.csv"))
        + list((CONTENDED / "matrix").glob("*/*.samples.csv"))
        + [block1.WORKLOAD, QUEUE_DEPTH, REORDER, CAPPED_SUMMARY,
           UNCONTENDED_LOG, CONTENDED_LOG]
    ))

    # Printed so the prose quotes the figure's own numbers rather than its own.
    # The first line is the qualifier the caption must carry: it was moved out
    # of the legend band to keep the plate inside the height cap, and it is
    # still derived here so the caption cannot drift from the runs.
    print(f"caption qualifier: {caveat}")
    print(f"intervals crossing 1.00: default {crossing(uncontended)} of "
          f"{len(uncontended)}, capped {crossing(contended)} of {len(contended)}")
    print(f"offered rate: default cap {offered_rps(UNCONTENDED / 'matrix'):.2f} rps, "
          f"capped {offered_rps(CONTENDED / 'matrix'):.2f} rps")
    print(f"in flight: {a_flight[0]}-{a_flight[1]} of {default_cap} slots vs "
          f"{b_flight[0]}-{b_flight[1]} of {capped_cap} slots")
    print(f"steps able to express an order: {a_queue['carryover'][0]}-"
          f"{a_queue['carryover'][1]} vs {b_queue['carryover'][0]}-"
          f"{b_queue['carryover'][1]}")
    for panel, ratios in (("default", uncontended), ("capped", contended)):
        for num, den in ROWS:
            est, low, high = ratios[(num, den)]
            verdict = "crosses 1.00" if low <= 1.0 <= high else "separated"
            inside = "inside" if 1 - (block1.SAFETY_MARGIN - 1) <= est <= block1.SAFETY_MARGIN else "outside"
            print(f"[{panel:<7}] {num}/{den}: {est:.4f} CI=[{low:.4f}, {high:.4f}] "
                  f"{verdict}, point {inside} the equivalence band")


if __name__ == "__main__":
    main()
