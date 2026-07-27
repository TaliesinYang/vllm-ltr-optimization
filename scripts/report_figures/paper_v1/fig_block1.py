"""fig:block1 -- serving-level scheduling results.

The statistics here are the ones the paper pre-registered before the data
existed:

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
    non-inferiority margin, drawn here two-sided (0.97-1.03) because it is
    presented as an equivalence region and four of the six ratios sit below
    1.0. A one-sided band would shade only the half that flatters us.

Rendering rules, which are load-bearing for what the reader is allowed to
conclude:

  * ONE AXIS PER PANEL. Panel (a) carries the five ordering contrasts on a
    single ratio axis. The implementation contrast (PolicyFCFS vs stock) is a
    different order of magnitude, so it is not squeezed onto that axis nor
    given a second competing tick strip underneath it; it is reported with its
    interval in panel (b)'s header, over the two absolute rows it compares.
  * NO BAR-IFIED ESTIMATES. Every estimate in all three panels is a dot with
    its interval. Nothing is a bar grown from zero with a CI on the tip.
  * AXES ARE FITTED TO THEIR DATA, padded symmetrically about their first and
    last tick. A dot-and-interval panel has no zero baseline to honour, so an
    axis that starts far below every interval only buys dead white.
  * EVERY INFERENTIAL MARK IS VALUE-LABELLED, in one right-aligned column per
    panel, in the same "point [lo, hi]" form in all three panels. A row that
    has no measurement says so in that same column instead of carrying a
    floating note through the panel's data channel.
  * THE MARGIN IS ADJUDICATED BY A NUMBER THAT CANNOT SIT ON A KNIFE EDGE.
    Panel (a) reports the WIDEST excursion of any interval and compares that
    with the margin, rather than counting how many intervals cross a line they
    can be four ten-thousandths away from. A per-row verdict at that
    resolution would print a word the rendering cannot support.
  * ORANGE MEANS DEGRADED OR OVERSTATED and is used exactly once: the double
    dagger on panel (c), whose binomial interval assumes independence between
    scheduling steps that a session does not provide.
  * PANEL (a) ROWS ARE COMPARISONS, so they carry no arm colour. A ratio has
    two arms and one mark; colouring it by its numerator would make the swatch
    key describe something the panel does not encode. The blue/grey ramps key
    panels (b) and (c), whose rows are arms.
  * ONE LABELLING SYSTEM: every row in every panel is named in a left column,
    right-aligned to its own axis with the same gutter, and every left-most
    text element in the figure shares one margin with the header strips.
  * LEADERS ARE MEASURED IN INCHES, not in axis fractions, so the clearance
    between a leader's last dot and an interval cap is the same visible gap in
    every panel, and a leader is drawn only where the mark is far enough from
    its name to need one.
  * NO VERTICAL GRIDLINES. With a value column in every panel the grid buys
    nothing, and it is what puts a rule underneath an interval cap.

The script refuses to draw a partial matrix. A missing arm is a missing
result, and a figure that quietly averages whatever happened to finish is
worse than no figure.
"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _common import (
    COLOR, IEEE_DOUBLE_WIDTH, REPO, load_json, load_jsonl, record_provenance, save,
)
from style import EXION  # noqa: E402  (_common puts report_figures on sys.path)

RUNS = REPO / "runs" / "block1-main"
WORKLOAD = REPO / "runs" / "block1-2026-07-26" / "workload-block1.jsonl"
REORDER = REPO / "runs" / "reorder-opportunity.json"
BOOTSTRAP_DRAWS = 5000
SAFETY_MARGIN = 1.03  # pre-declared non-inferiority margin on mean TTLT
Z95 = 1.959963984540054

# Directory stem -> (display label, is this ours).
ARMS: dict[str, tuple[str, bool]] = {
    "stock_fcfs": ("Stock FCFS", False),
    "StockFCFSShim": ("Stock FCFS (shim)", False),
    "PolicyFCFS": ("PolicyFCFS", False),
    "PromptLengthSJFScheduler": ("PromptLengthSJF", False),
    "PureLTRScheduler": ("PureLTR", True),
    "GatedRuleCScheduler": ("GatedRuleC", True),
}

# Panel (b) reading order, top to bottom. The shim arm is deliberately not a
# row and is deliberately not a clause on any axis label either: it lands
# within 0.01 s of native stock, which is a statement about two numbers being
# equal, and an axis label is not the place to put a result. It is loaded (so
# the completeness guard still sees it) and printed to stdout for the prose.
LEVEL_STEMS = [
    "stock_fcfs",
    "PolicyFCFS",
    "PromptLengthSJFScheduler",
    "PureLTRScheduler",
    "GatedRuleCScheduler",
]
SHIM_STEM = "StockFCFSShim"

# The five ordering contrasts panel (a) draws. Each is named by its two arms,
# which is the whole of what the row asserts; the earlier one-word "role"
# column was a private taxonomy the figure never defined, so it is gone. The
# pre-registered test is kept because stdout reports its verdict for the prose.
COMPARISONS: list[tuple[str, str, str, str]] = [
    ("GatedRuleCScheduler", "PromptLengthSJFScheduler", "primary", "superiority"),
    ("GatedRuleCScheduler", "PolicyFCFS", "safety", "non-inferiority"),
    # The gate's own hypothesis: selective trust beats blind trust. This is
    # the only comparison that isolates the gate, since the two arms share
    # the Ranker and differ solely in whether its score is acted on.
    ("GatedRuleCScheduler", "PureLTRScheduler", "gate value", ""),
    ("PureLTRScheduler", "PolicyFCFS", "secondary", ""),
    ("PromptLengthSJFScheduler", "PolicyFCFS", "secondary", ""),
]

# The one contrast whose two arms both serve FCFS order, so nothing it shows
# can be credited to ranking. It is not an ordering row: it is the headline of
# panel (b), stated there with its interval, directly above the two absolute
# rows it divides.
ATTRIBUTION = ("PolicyFCFS", "stock_fcfs")

# One EXION family ramp for the four arms that carry a policy hook, light ->
# dark by how much of the ordering decision that arm's ranker owns: PolicyFCFS
# (none, arrival order) -> PromptLengthSJF (a scalar heuristic) -> PureLTR (the
# learned score, unconditionally) -> GatedRuleC (the learned score plus the
# gate). Stock vLLM is the engine we did not build, so it takes the EXION
# baseline mint, and its shim takes a structure grey.
ARM_COLOR: dict[str, str] = {
    "stock_fcfs": EXION["baseline"],
    "StockFCFSShim": EXION["structure"][3],
    "PolicyFCFS": EXION["family"][0],
    "PromptLengthSJFScheduler": EXION["family"][1],
    "PureLTRScheduler": EXION["family"][2],
    "GatedRuleCScheduler": EXION["family"][3],
}
RANKER_RAMP = ["PromptLengthSJFScheduler", "PureLTRScheduler", "GatedRuleCScheduler"]
FCFS_RAMP = ["stock_fcfs", "PolicyFCFS"]

TEXT = "#333333"            # ink: every label, value and panel (a) mark
MUTED = EXION["structure"][3]
BAND = EXION["structure"][3]
WARN = COLOR["overstate"]   # reserved: degraded or overstated quantities only
LEADER = EXION["structure"][1]
FRAME = EXION["structure"][3]
STRIP_FILL = EXION["structure"][0]
CALLOUT_FILL = EXION["structure"][2]

# ---------------------------------------------------------------------------
# Geometry, in inches. Written out rather than left to a layout engine because
# the header strips, the row-label column and the value columns all have to sit
# on shared margins, which needs positions that do not move.
FIG_W = IEEE_DOUBLE_WIDTH
FIG_H = 4.95
MARGIN = 0.10
CONTENT_L = MARGIN
CONTENT_R = FIG_W - MARGIN

TEXT_INSET = 0.06     # every left-most text element sits this far into its block
LABEL_GUTTER = 0.07   # every row label ends this far left of its own axis
VALUE_GUTTER = 0.10   # every value column starts this far right of its own axis
STRIP_H = 0.19
STRIP_GAP = 0.07
CALLOUT_PAD_X = 0.07   # framed-result box: horizontal padding, in inches
CALLOUT_PAD_Y = 0.055
CALLOUT_LINE = 0.115

# Interval glyph, in inches, so the same shape is drawn on every axis whatever
# each axis's units are. The caps are taller than the marker on purpose: the
# zero-count control in panel (c) has an interval narrower than any legible
# dot, and caps that protrude above and below the dot are what keep that row
# readable as an interval rather than as a bare point.
MARKER_PT = 3.6
CAP_HALF_IN = 0.050
LEADER_CLEAR_IN = 0.055   # gap between a leader's last dot and the interval cap
LEADER_MIN_IN = 0.18      # shorter than this and a leader reads as dirt

# --- panel (a) -------------------------------------------------------------
ORDER_XLIM = (0.955, 1.045)
ORDER_TICKS = [0.96, 0.98, 1.00, 1.02, 1.04]
YLIM_ORDER = (-0.6, 6.0)   # headroom above row 4 for the band tag and callout
ANNOT_ROW = 5.1            # the band tag and the framed result share one line

STRIP_TOP_A = FIG_H - MARGIN
# The name column carries the longest string in the figure -- a contrast is
# named by BOTH its arms -- so it is sized from that string's rendered width in
# the set's one family, not from a number that happened to fit a narrower one.
A_LABEL_R = 1.92                    # right edge of the comparison-name column
A_AX_L = A_LABEL_R + LABEL_GUTTER
A_AX_R = 5.74
A_AX_W = A_AX_R - A_AX_L
A_VALUE_R = CONTENT_R - 0.04   # every value column sits 0.04in inside its block
A_AX_H = 1.55
A_AX_T = STRIP_TOP_A - STRIP_H - STRIP_GAP
A_AX_B = A_AX_T - A_AX_H
A_XLABEL_Y = A_AX_B - 0.20

AX_ORDER = (A_AX_L, A_AX_B, A_AX_W, A_AX_H)

# --- panels (b) and (c) ----------------------------------------------------
STRIP_TOP_BC = 2.37
BC_AX_T = STRIP_TOP_BC - STRIP_H - STRIP_GAP
BC_AX_H = 1.06
BC_AX_B = BC_AX_T - BC_AX_H
BC_XLABEL_Y = BC_AX_B - 0.20

BLOCK_A = (CONTENT_L, CONTENT_R)
BLOCK_B = (CONTENT_L, 4.06)
BLOCK_C = (4.30, CONTENT_R)

# (c) plots a SUBSET of (b)'s arms, so the two panels share one row grid and
# one name column: same rows, same pitch, named once. The arm (b) shows but (c)
# cannot instrument is then an empty row whose value column says why, rather
# than a note floating across the panel.
B_AX_L, B_AX_W = 1.22, 1.70
C_AX_L, C_AX_W = 4.40, 1.55
B_VALUE_R = BLOCK_B[1] - 0.04
C_VALUE_R = CONTENT_R - 0.04
AX_LEVEL = (B_AX_L, BC_AX_B, B_AX_W, BC_AX_H)
AX_QUEUE = (C_AX_L, BC_AX_B, C_AX_W, BC_AX_H)
YLIM_BC = (-0.6, 4.6)
QUEUE_XLIM = (-2.0, 102.0)
QUEUE_TICKS = [0, 50, 100]

LEGEND_BOX = (CONTENT_L, MARGIN, CONTENT_R - CONTENT_L, 0.40)

# The set is one family. DejaVu Sans is the only one requested anywhere in this
# figure because it is the only one that ships regular, bold, oblique AND math
# faces here; the previous Arial-for-bold fallback was what put a second family
# on the plate. Named explicitly rather than inherited so a bold request can
# never silently resolve elsewhere.
BOLD_FAMILY = ["DejaVu Sans"]

# Three tiers, no others: 10pt is unused in this figure (it has no axis title
# beyond the axis labels), 9pt carries annotations, tick labels and header
# strips, 8pt carries the dense in-panel columns -- row names, value columns,
# callout and legend text.
SIZE_ANNOT = 9
SIZE_DENSE = 8
AXIS_LABEL_LINE = 0.135   # baseline pitch of a wrapped axis label, in inches


def rect(spec: tuple[float, float, float, float]) -> list[float]:
    left, bottom, width, height = spec
    return [left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H]


def row_y(spec: tuple[float, float, float, float],
          ylim: tuple[float, float], row: float) -> float:
    """Figure-fraction y of a data row inside an axes rectangle."""
    _left, bottom, _width, height = spec
    frac = (row - ylim[0]) / (ylim[1] - ylim[0])
    return (bottom + height * frac) / FIG_H


def x_per_inch(spec: tuple[float, float, float, float],
               xlim: tuple[float, float]) -> float:
    """Inches of paper per one unit of an axes' x data."""
    return spec[2] / (xlim[1] - xlim[0])


def cap_units(spec: tuple[float, float, float, float],
              ylim: tuple[float, float]) -> float:
    """CAP_HALF_IN expressed in that axes' y units."""
    return CAP_HALF_IN / (spec[3] / (ylim[1] - ylim[0]))


def text_width(fig, string: str, fontsize: float, weight: str = "normal") -> float:
    """Rendered width of `string` in inches, so nothing is sized by guesswork."""
    family = BOLD_FAMILY if weight == "bold" else None
    probe = fig.text(0, 0, string, fontsize=fontsize, fontweight=weight,
                     fontfamily=family)
    width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    probe.remove()
    return width


def check_fits(fig, string: str, fontsize: float, budget: float,
               where: str, weight: str = "normal") -> None:
    """House rule: shorten the wording, never shrink the glyphs. So an
    overflow is a build-time error rather than something to fix with a
    smaller font."""
    width = text_width(fig, string, fontsize, weight)
    if width > budget:
        raise SystemExit(
            f"{where}: {width:.2f}in of text in a {budget:.2f}in slot; "
            f"shorten the wording rather than shrinking it -- {string!r}"
        )


def header_strip(fig, block: tuple[float, float], top: float, text: str) -> None:
    """Light-grey bold strip spanning exactly its panel's block.

    A strip carries a LABEL, not a finding: at most four words, no numbers, no
    verdicts. A finding belongs to a mark, a value column, or the framed
    callout that sits inside the panel it is about.
    """
    words = [w for w in text.split() if not w.startswith("(")]
    if len(words) > 4 or any(character.isdigit() for character in text[3:]):
        raise SystemExit(
            f"header strip must be a label of at most four words with no "
            f"numbers -- {text!r}"
        )
    x0, x1 = block
    y0 = (top - STRIP_H) / FIG_H
    height = STRIP_H / FIG_H
    fig.add_artist(Rectangle((x0 / FIG_W, y0), (x1 - x0) / FIG_W, height,
                             transform=fig.transFigure, facecolor=STRIP_FILL,
                             edgecolor=FRAME, linewidth=0.6, zorder=5))
    usable = (x1 - x0) - 2 * TEXT_INSET
    check_fits(fig, text, 9, usable, "header strip", weight="bold")
    fig.text((x0 + TEXT_INSET) / FIG_W, y0 + height / 2, text, ha="left",
             va="center", fontsize=9, fontweight="bold", fontfamily=BOLD_FAMILY,
             color="#222222", zorder=6)


def callout(fig, x_right: float, y_centre: float, lines: list[str],
            budget: float, where: str) -> None:
    """Framed result box, right-aligned inside its panel.

    This is where a finding goes once it is off the header strip: still inside
    the panel that measures it, still next to the marks it summarises, but
    framed so it reads as a stated result rather than as a title.
    """
    width = max(text_width(fig, line, 8) for line in lines)
    box_w = width + 2 * CALLOUT_PAD_X
    if box_w > budget:
        raise SystemExit(
            f"{where}: callout needs {box_w:.2f}in in a {budget:.2f}in slot; "
            f"shorten the wording rather than shrinking it -- {lines!r}"
        )
    box_h = len(lines) * CALLOUT_LINE + 2 * CALLOUT_PAD_Y
    x0 = x_right - box_w
    y0 = y_centre - box_h / 2
    fig.add_artist(Rectangle((x0 / FIG_W, y0 / FIG_H), box_w / FIG_W,
                             box_h / FIG_H, transform=fig.transFigure,
                             facecolor=CALLOUT_FILL, edgecolor=FRAME,
                             linewidth=0.6, zorder=7))
    for index, line in enumerate(lines):
        y = y0 + box_h - CALLOUT_PAD_Y - (index + 0.5) * CALLOUT_LINE
        fig.text((x0 + CALLOUT_PAD_X) / FIG_W, y / FIG_H, line, ha="left",
                 va="center", fontsize=8, color="#222222", zorder=8)


def inches_y(spec: tuple[float, float, float, float],
             ylim: tuple[float, float], row: float) -> float:
    """Inches from the figure's bottom edge of a data row on `spec`."""
    return row_y(spec, ylim, row) * FIG_H


def full_canvas_spacer(fig) -> None:
    """Reserve the whole canvas so the tight bounding box keeps the margins."""
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                             facecolor="none", edgecolor="none", linewidth=0,
                             zorder=-10))


def bare_axis(ax, xlim, ylim, ticks, labels=None) -> None:
    """One bottom rule, ticks outward, no grid, no y spine.

    The grid is omitted on purpose: every panel prints its values in a column,
    so a rule behind the marks adds no readable quantity and is what ends up
    drawn underneath an interval cap.
    """
    left_pad = ticks[0] - xlim[0]
    right_pad = xlim[1] - ticks[-1]
    if abs(left_pad - right_pad) > 1e-9 * max(1.0, abs(ticks[-1])):
        raise SystemExit(
            f"axis padded {left_pad:.4g} left of its first tick but "
            f"{right_pad:.4g} right of its last; pad both ends alike."
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(ticks, labels if labels is not None else None)
    ax.tick_params(axis="x", labelsize=SIZE_ANNOT, direction="out", length=2.5,
                   pad=2)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)


# The four arms that carry a policy hook and therefore log an emitted order.
POLICY_STEMS = [
    "PolicyFCFS",
    "PromptLengthSJFScheduler",
    "PureLTRScheduler",
    "GatedRuleCScheduler",
]
FCFS_CONTROL = "PolicyFCFS"


def load_reorder() -> dict[str, tuple[int, int]]:
    """stem -> (steps that reordered, steps that COULD have reordered).

    Panel (c) asks the one question that decides whether a null ordering
    result is a result at all: when the scheduler was actually in a position
    to change the order, did it? The denominator is therefore not "all logged
    steps" -- a step made only of first-time arrivals has no prior order to
    change, and dividing by those would report a policy's opportunity rather
    than its behaviour. PolicyFCFS is the control inside the panel: it emits
    arrival order by construction, so anything other than zero there would
    mean the counter, not the scheduler, is what is being measured.
    """
    payload = load_json(REORDER)
    round_a = payload["rounds"]["round_a"]
    out: dict[str, tuple[int, int]] = {}
    for stem in POLICY_STEMS:
        if stem not in round_a:
            raise SystemExit(f"reorder-opportunity.json lacks {stem}")
        eligible = int(round_a[stem]["steps_with_two_waiting_carryover"])
        if eligible <= 0:
            raise SystemExit(
                f"{stem} had {eligible} steps able to express an order; panel "
                "(c) would divide by zero opportunity."
            )
        out[stem] = (int(round_a[stem]["reorder_events"]), eligible)
    if out[FCFS_CONTROL][0] != 0:
        raise SystemExit(
            f"{FCFS_CONTROL} logged {out[FCFS_CONTROL][0]} reorderings; it "
            "serves arrival order by construction, so the counter is suspect "
            "and panel (c) must not be drawn."
        )
    if any(out[stem][0] == 0 for stem in POLICY_STEMS if stem != FCFS_CONTROL):
        raise SystemExit(
            "a ranked arm reordered nothing; panel (c)'s claim that ordering "
            "was applied at all no longer holds."
        )
    return out


def wilson_pct(count: int, n: int) -> tuple[float, float, float]:
    """Point estimate and 95% Wilson interval, as percentages.

    Panels (a) and (b) resample sessions inside launches; panel (c) counts
    scheduling steps, which the run logs individually and which no session
    bootstrap covers. The interval is therefore binomial -- and binomial
    intervals assume independent trials, which consecutive scheduling steps
    inside one session are not. The panel carries the reserved warning mark
    for exactly that reason: the width drawn is a floor.
    """
    p = count / n
    denominator = 1.0 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denominator
    return p * 100.0, (centre - half) * 100.0, (centre + half) * 100.0


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


def draw_estimate(ax, x: float, low: float, high: float, row: float,
                  colour: str, cap: float) -> None:
    """Point estimate with an interval that stays readable as an interval.

    The marker is laid down FIRST with a white rim and the interval's caps go
    over it, so on a row whose interval is narrower than any legible dot --
    panel (c)'s zero-count control -- the caps still protrude above and below
    the marker instead of vanishing underneath it.
    """
    ax.plot([x], [row], "o", color=colour, ms=MARKER_PT, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot([low, high], [row, row], color=colour, lw=1.3,
            solid_capstyle="butt", zorder=5)
    for end in (low, high):
        ax.plot([end, end], [row - cap, row + cap], color=colour, lw=1.0,
                solid_capstyle="butt", zorder=6)


def leader(ax, xlim: tuple[float, float], stop: float, row: float,
           scale: float) -> None:
    """Hairline leader from the axis origin to the interval.

    The clearance to the cap and the minimum useful length are both in inches,
    so the visible gap is identical on every axis regardless of that axis's
    units, and a leader appears only where the mark is genuinely far from the
    name it belongs to.
    """
    end = stop - LEADER_CLEAR_IN / scale
    if (end - xlim[0]) * scale >= LEADER_MIN_IN:
        ax.plot([xlim[0], end], [row, row], color=LEADER, lw=0.5,
                linestyle=(0, (1, 2)), zorder=0)


def value_column(fig, x_right: float, y: float, text: str,
                 colour: str = TEXT) -> None:
    """One right-aligned numeric column per panel, at the panel's right edge."""
    fig.text(x_right / FIG_W, y, text, ha="right", va="center", fontsize=8,
             color=colour, zorder=6)


def row_label(fig, x_right: float, y: float, text: str) -> None:
    """Row name, right-aligned to LABEL_GUTTER left of its own axis."""
    fig.text(x_right / FIG_W, y, text, ha="right", va="center", fontsize=8,
             color=TEXT, zorder=6)


def legend_band(fig) -> None:
    """The boxed key, at the foot of the figure rather than between the panels.

    Two rows on ONE two-column grid, and one mark slot of one width, so every
    key's swatch shares a left edge with the key above or below it and every
    key's text does too. One glyph carries one meaning: there is no key here
    that has to be read with a scoping parenthesis to know which of two things
    it is talking about.
    """
    left, bottom, width, height = LEGEND_BOX
    fig.add_artist(Rectangle((left / FIG_W, bottom / FIG_H), width / FIG_W,
                             height / FIG_H, transform=fig.transFigure,
                             facecolor="white", edgecolor=FRAME,
                             linewidth=0.6, zorder=5))
    inset = TEXT_INSET   # same left margin as the header strips
    chip, gap, pad = 0.085, 0.018, 0.06
    mark_w = 3 * chip + 2 * gap      # widest mark; every mark is drawn in this slot
    columns = 2
    col_w = (width - 2 * inset) / columns
    rows = [
        # The ramps are scoped to (b) and (c) on purpose: panel (a)'s rows are
        # comparisons, so no single arm colour can describe one of them.
        (bottom + height * 0.70, [
            ("ramp", RANKER_RAMP, "Ranker-ordered arms in (b), (c)"),
            ("ramp", FCFS_RAMP, "FCFS-ordered arms in (b), (c)"),
        ]),
        (bottom + height * 0.30, [
            ("ci", None, "95% CI: paired session bootstrap"),
            ("flag", None, "‡ (c) is a binomial CI: steps are not independent"),
        ]),
    ]

    for y, entries in rows:
        for column, (kind, ramp, label) in enumerate(entries):
            x = left + inset + column * col_w
            budget = col_w - mark_w - pad - inset
            check_fits(fig, label, 8, budget, f"legend column {column + 1}")
            if kind == "ramp":
                for index, stem in enumerate(ramp):
                    fig.add_artist(Rectangle(
                        ((x + index * (chip + gap)) / FIG_W, (y - chip / 2) / FIG_H),
                        chip / FIG_W, chip / FIG_H, transform=fig.transFigure,
                        facecolor=ARM_COLOR[stem], edgecolor="none", zorder=6))
            elif kind == "ci":
                fig.add_artist(Line2D(
                    [x / FIG_W, (x + 0.16) / FIG_W], [y / FIG_H, y / FIG_H],
                    transform=fig.transFigure, color=TEXT, linewidth=1.3,
                    marker="|", markersize=5.0, zorder=6))
                fig.add_artist(Line2D(
                    [(x + 0.08) / FIG_W], [y / FIG_H], transform=fig.transFigure,
                    color=TEXT, marker="o", markersize=MARKER_PT, linestyle="none",
                    markeredgecolor="white", markeredgewidth=0.8, zorder=7))
            else:
                fig.text(x / FIG_W, y / FIG_H, "‡", ha="left", va="center",
                         fontsize=10, color=WARN, fontweight="bold",
                         fontfamily=BOLD_FAMILY, zorder=6)
            fig.text((x + mark_w + pad) / FIG_W, y / FIG_H, label, ha="left",
                     va="center", fontsize=8, color=TEXT, zorder=6)


def draw_panel_a(fig, ax, ratio):
    """Five ordering contrasts, one ratio axis, one shaded equivalence band.

    The band is named where it is drawn, so nothing about the single most
    load-bearing shape in the panel depends on a swatch in the legend that
    prints as near-white on white.
    """
    band = (2.0 - SAFETY_MARGIN, SAFETY_MARGIN)
    ax.axvspan(*band, color=BAND, alpha=0.15, lw=0, zorder=0)
    ax.axvline(1.0, color=FRAME, lw=0.9, zorder=1)

    scale = x_per_inch(AX_ORDER, ORDER_XLIM)
    cap = cap_units(AX_ORDER, YLIM_ORDER)
    label_budget = A_LABEL_R - (CONTENT_L + TEXT_INSET)
    value_budget = A_VALUE_R - (A_AX_R + VALUE_GUTTER)

    ordering = []
    for index, (num, den, _role, _test) in enumerate(COMPARISONS):
        est, low, high = ratio[(num, den)]
        row = len(COMPARISONS) - 1 - index
        y_fig = row_y(AX_ORDER, YLIM_ORDER, row)
        if not ORDER_XLIM[0] <= low <= high <= ORDER_XLIM[1]:
            raise SystemExit(
                f"{ARMS[num][0]} / {ARMS[den][0]}: [{low:.4f}, {high:.4f}] falls "
                f"outside {ORDER_XLIM}; widen panel (a) rather than clip."
            )
        leader(ax, ORDER_XLIM, low, row, scale)
        draw_estimate(ax, est, low, high, row, TEXT, cap)

        pair = f"{ARMS[num][0]} / {ARMS[den][0]}"
        check_fits(fig, pair, 8, label_budget, "panel (a) row label")
        row_label(fig, A_LABEL_R, y_fig, pair)
        value = f"{est:.3f} [{low:.3f}, {high:.3f}]"
        check_fits(fig, value, 8, value_budget, "panel (a) value column")
        value_column(fig, A_VALUE_R, y_fig, value)
        ordering.append((est, low, high))

    bare_axis(ax, ORDER_XLIM, YLIM_ORDER, ORDER_TICKS,
              [f"{t:.2f}" for t in ORDER_TICKS])
    # The band carries its own name, inside itself, clear of the 1.00 rule and
    # clear of every data row.
    # Named inside the half of the band left of the 1.00 rule -- the only part
    # of it no interval and no callout occupies -- so the wording is budgeted
    # against 0.03 of ratio, not against the whole band. "band" is dropped from
    # the wording because the shading already is one.
    tag = f"±{(SAFETY_MARGIN - 1) * 100:g}% equivalence"
    tag_units = text_width(fig, tag, SIZE_DENSE) / scale
    centre = (band[0] + 1.0) / 2
    if centre - tag_units / 2 < band[0] or centre + tag_units / 2 > 1.0:
        raise SystemExit(f"panel (a) band label does not fit inside the band: {tag!r}")
    ax.text(centre, ANNOT_ROW, tag, ha="center", va="center",
            fontsize=8, color=MUTED, zorder=3)
    return ordering


def bc_row(stem: str) -> int:
    """Row index shared by panels (b) and (c) for an arm."""
    return len(LEVEL_STEMS) - 1 - LEVEL_STEMS.index(stem)


def level_axis(bounds: list[tuple[float, float]]) -> tuple[tuple[float, float], list[float]]:
    """Ticks and symmetric limits for panel (b), fitted to what it plots.

    Nothing in a dot-and-interval panel is grown from zero, so the axis owes
    zero nothing: it is fitted to the intervals it carries, then padded by the
    SAME amount outside its first and its last tick.
    """
    low = min(lo for lo, _hi in bounds)
    high = max(hi for _lo, hi in bounds)
    for step in (0.25, 0.5, 1.0, 2.0, 5.0):
        first = math.ceil(low / step - 1e-9) * step
        last = math.floor(high / step + 1e-9) * step
        count = int(round((last - first) / step)) + 1
        if 4 <= count <= 7:
            ticks = [round(first + i * step, 6) for i in range(count)]
            pad = max(first - low, high - last) + 0.02 * (high - low)
            return (round(first - pad, 6), round(last + pad, 6)), ticks
    raise SystemExit(f"no readable tick step for panel (b) over [{low}, {high}]")


def draw_panel_b(fig, ax, point, draws):
    """Absolute pooled mean TTLT, as dots with intervals -- not bars.

    This panel owns the name column that (c) also reads from, so its labels are
    drawn here for both.
    """
    bounds = []
    for stem in LEVEL_STEMS:
        low, high = interval(draws[stem])
        bounds.append((low / 1000, high / 1000))
    xlim, ticks = level_axis(bounds)
    scale = x_per_inch(AX_LEVEL, xlim)
    cap = cap_units(AX_LEVEL, YLIM_BC)
    label_r = B_AX_L - LABEL_GUTTER
    budget = label_r - (BLOCK_B[0] + TEXT_INSET)
    value_budget = B_VALUE_R - (B_AX_L + B_AX_W + VALUE_GUTTER)

    for stem, (low, high) in zip(LEVEL_STEMS, bounds):
        row = bc_row(stem)
        est = point[stem] / 1000
        y_fig = row_y(AX_LEVEL, YLIM_BC, row)
        leader(ax, xlim, low, row, scale)
        draw_estimate(ax, est, low, high, row, ARM_COLOR[stem], cap)
        check_fits(fig, ARMS[stem][0], 8, budget, "row name column")
        row_label(fig, label_r, y_fig, ARMS[stem][0])
        value = f"{est:.2f} [{low:.2f}, {high:.2f}]"
        check_fits(fig, value, 8, value_budget, "panel (b) value column")
        value_column(fig, B_VALUE_R, y_fig, value)
    bare_axis(ax, xlim, YLIM_BC, ticks, [f"{t:.1f}" for t in ticks])
    return xlim


def draw_panel_c(fig, ax, reorder):
    """Of the steps that could express an order, the share that did.

    PolicyFCFS is the control inside the panel, so (c) has an FCFS baseline to
    be read against without the stock arm, which carries no policy hook and
    therefore logs no emitted order. That arm's row keeps its place in the
    shared grid and says what is missing in the same value column every other
    row uses, so nothing is written across the panel's data channel.
    """
    scale = x_per_inch(AX_QUEUE, QUEUE_XLIM)
    cap = cap_units(AX_QUEUE, YLIM_BC)
    value_budget = C_VALUE_R - (C_AX_L + C_AX_W + VALUE_GUTTER)
    for stem in POLICY_STEMS:
        row = bc_row(stem)
        count, total = reorder[stem]
        est, low, high = wilson_pct(count, total)
        leader(ax, QUEUE_XLIM, low, row, scale)
        draw_estimate(ax, est, low, high, row, ARM_COLOR[stem], cap)
        value = f"{est:.1f} [{low:.1f}, {high:.1f}]"
        check_fits(fig, value, 8, value_budget, "panel (c) value column")
        value_column(fig, C_VALUE_R, row_y(AX_QUEUE, YLIM_BC, row), value)
    for stem in LEVEL_STEMS:
        if stem in reorder:
            continue
        missing = "not order-logged"
        check_fits(fig, missing, 8, value_budget, "panel (c) missing-value column")
        value_column(fig, C_VALUE_R, row_y(AX_QUEUE, YLIM_BC, bc_row(stem)),
                     missing, colour=MUTED)
    bare_axis(ax, QUEUE_XLIM, YLIM_BC, QUEUE_TICKS, [str(t) for t in QUEUE_TICKS])


def axis_label(fig, centre: float, y: float, block: tuple[float, float],
               lines: list[str], flag: str = "") -> None:
    """One axis label, centred on its axis as a whole, wrapping if it must.

    Measured at the size it is DRAWN at. It used to be measured at 8.5pt and
    drawn at 9, so the guard was reading a label ~6% narrower than the one on
    the page -- the check passed on text that did not fit.

    A label may be given as several lines. They share one left edge, which is
    what keeps a wrapped label reading as one block rather than as two stray
    notes, and the whole block is centred on the axis. Panel (c) needs this:
    its label carries the eligible-step range, which is a number the paper
    re-derives and therefore cannot be dropped, and no single line holding it
    fits between (b)'s value column and the page edge.

    When a label carries the reserved warning mark, the mark rides on the last
    line and is part of what gets centred, so the composite sits on the axis's
    centre rather than to the right of it.
    """
    widths = [text_width(fig, line, SIZE_ANNOT) for line in lines]
    flag_w = (0.05 + text_width(fig, flag, SIZE_ANNOT)) if flag else 0.0
    widths[-1] += flag_w
    span = max(widths)
    left = centre - span / 2
    if left < block[0] + TEXT_INSET or left + span > block[1] - TEXT_INSET:
        raise SystemExit(
            f"axis label {lines!r} spans {left:.2f}-{left + span:.2f}in, "
            f"outside its {block} block; shorten the wording or wrap it."
        )
    for index, line in enumerate(lines):
        fig.text(left / FIG_W, (y - index * AXIS_LABEL_LINE) / FIG_H, line,
                 ha="left", va="top", fontsize=SIZE_ANNOT, color=TEXT, zorder=6)
    if flag:
        last_y = y - (len(lines) - 1) * AXIS_LABEL_LINE
        fig.text((left + widths[-1] - flag_w + 0.05) / FIG_W, last_y / FIG_H,
                 flag, ha="left", va="top", fontsize=SIZE_ANNOT, color=WARN,
                 fontweight="bold", fontfamily=BOLD_FAMILY, zorder=6)


def build_figure(arms, draws, shared_sessions, reorder):
    point = {stem: pooled_mean(launches, shared_sessions)
             for stem, launches in arms.items()}
    ratio: dict[tuple[str, str], tuple[float, float, float]] = {}
    for num, den in [(a, b) for a, b, _r, _t in COMPARISONS] + [ATTRIBUTION]:
        low, high = interval(draws[num] / draws[den])
        ratio[(num, den)] = (point[num] / point[den], low, high)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    full_canvas_spacer(fig)
    ax_order = fig.add_axes(rect(AX_ORDER))
    ax_level = fig.add_axes(rect(AX_LEVEL))
    ax_queue = fig.add_axes(rect(AX_QUEUE))

    ordering = draw_panel_a(fig, ax_order, ratio)
    xlim_level = draw_panel_b(fig, ax_level, point, draws)
    draw_panel_c(fig, ax_queue, reorder)

    eligible = sorted(total for _count, total in reorder.values())
    axis_label(fig, A_AX_L + A_AX_W / 2, A_XLABEL_Y, BLOCK_A,
               ["Paired ratio of mean TTLT (<1: numerator faster)"])
    axis_label(fig, B_AX_L + B_AX_W / 2, BC_XLABEL_Y, BLOCK_B,
               ["Pooled mean TTLT (s)"])
    axis_label(fig, C_AX_L + C_AX_W / 2, BC_XLABEL_Y, BLOCK_C,
               ["Reordered (%),", f"{eligible[0]}–{eligible[-1]} eligible steps"],
               flag="‡")
    legend_band(fig)

    # ---- header strips are labels; the findings sit in framed callouts -----
    header_strip(fig, BLOCK_A, STRIP_TOP_A, "(a) Paired ordering ratios")
    header_strip(fig, BLOCK_B, STRIP_TOP_BC, "(b) Absolute TTLT")
    header_strip(fig, BLOCK_C, STRIP_TOP_BC, "(c) Ordering applied")

    # Panel (a)'s finding, on the row the band tag shares, right of the 1.00
    # rule where no interval reaches. Every number is recomputed from the
    # values the panel drew.
    margin_pct = (SAFETY_MARGIN - 1) * 100
    widest = max(max(abs(low - 1.0), abs(high - 1.0))
                 for _est, low, high in ordering) * 100
    n_one = sum(1 for _est, low, high in ordering if low <= 1.0 <= high)
    lead = (f"All {len(ordering)} CIs contain 1.00" if n_one == len(ordering)
            else f"{n_one} of {len(ordering)} CIs contain 1.00")
    relation = ">" if widest > margin_pct else "<"
    order_scale = x_per_inch(AX_ORDER, ORDER_XLIM)
    rule_x = A_AX_L + (1.0 - ORDER_XLIM[0]) * order_scale
    callout(fig, A_AX_R, inches_y(AX_ORDER, YLIM_ORDER, ANNOT_ROW),
            [lead, f"widest ±{widest:.1f}% {relation} ±{margin_pct:g}% margin"],
            A_AX_R - rule_x, "panel (a) callout")

    # Panel (b)'s finding: the one contrast whose two arms both serve arrival
    # order, so it is the shim-plus-hook cost, not a ranking result. It sits
    # over the two absolute rows it divides.
    attr_est, attr_low, attr_high = ratio[ATTRIBUTION]
    level_scale = B_AX_W / (xlim_level[1] - xlim_level[0])
    clear_from = max(interval(draws[stem])[1] / 1000
                     for stem in LEVEL_STEMS if stem != "stock_fcfs") + 0.08
    left_in = B_AX_L + (clear_from - xlim_level[0]) * level_scale
    callout(fig, B_AX_L + B_AX_W, inches_y(AX_LEVEL, YLIM_BC, 1.5),
            ["Stock / PolicyFCFS",
             f"{1 / attr_est:.2f}× [{1 / attr_high:.2f}, {1 / attr_low:.2f}]"],
            B_AX_L + B_AX_W - left_in, "panel (b) callout")
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

    reorder = load_reorder()
    draws = hierarchical_draws(arms, shared_sessions, seed=20260727)
    fig, point = build_figure(arms, draws, shared_sessions, reorder)
    save(fig, "block1.pdf")
    plt.close(fig)

    record_provenance("block1.pdf", sorted(
        p for stem in ARMS for p in (RUNS / "matrix" / f"{stem}.runs").glob("*.samples.csv")
    ) + [WORKLOAD, REORDER])

    # Printed so the prose in 06.evaluation.tex quotes the figure's own numbers.
    for stem in ARMS:
        low, high = interval(draws[stem])
        print(f"{ARMS[stem][0]:<20} mean_ms={point[stem]:8.1f}  CI=[{low:.1f}, {high:.1f}]")
    shim_delta = (point[SHIM_STEM] - point["stock_fcfs"]) / 1000
    print(f"[shim cost  ] {ARMS[SHIM_STEM][0]} - Stock FCFS = {shim_delta:+.3f} s "
          "(not drawn; a two-number equality, reported in prose)")
    for num, den, role, test in COMPARISONS + [ATTRIBUTION + ("attribution", "")]:
        ratios = draws[num] / draws[den]
        low, high = interval(ratios)
        est = point[num] / point[den]
        verdict = ""
        if test == "superiority":
            verdict = "PASS" if high < 1.0 else "FAIL"
        elif test == "non-inferiority":
            verdict = "PASS" if high < SAFETY_MARGIN else "FAIL"
        print(f"[{role:<11}] {ARMS[num][0]} / {ARMS[den][0]}: "
              f"{est:.4f} CI=[{low:.4f}, {high:.4f}] {verdict}".rstrip())
    for stem in POLICY_STEMS:
        count, total = reorder[stem]
        est, low, high = wilson_pct(count, total)
        print(f"[reorder    ] {ARMS[stem][0]}: {count}/{total} eligible steps "
              f"= {est:.2f}% CI=[{low:.2f}, {high:.2f}]")


if __name__ == "__main__":
    main()
