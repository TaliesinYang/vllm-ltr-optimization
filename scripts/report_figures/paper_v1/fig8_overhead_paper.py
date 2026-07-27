"""fig:overhead -- publishes the gateway overhead figure into the paper.

Provenance: the measurement behind this figure comes from the 2026-07-19
rental, NOT from the trace-calibrated Block-1 session, and that distinction has
to be recorded somewhere a reader can check rather than left implicit in a
manually copied file.

E3's text is scoped to match: it claims a cost of placing the Ranker on the
synchronous path, measured against a direct-to-engine baseline, and says
nothing about the Block-1 workload.

The data contract (which requests pair, which pairs survive the output-length
match) is imported from scripts/report_figures/fig8_overhead.py and is never
re-derived here; only the drawing is paper-specific.

Layout: each metric gets a pair of stacked axes that share one left and one
right edge -- the arms above, the paired difference below. The estimand the
figure exists to claim is the paired mean difference, so that is the estimate
that carries the only inferential mark on the page: a capped 95% bootstrap
interval with its numbers printed beside it. The per-arm sampling error is not
drawn at all, because an unpaired per-arm interval licences no difference claim
and drawing it as the most prominent glyph would invite exactly that reading.
Individual pairs are likewise not drawn as connecting lines: 150 opaque
hairlines saturate into a flat slab that out-inks the density bodies and breaks
the gridlines it crosses, and the pairing they were there to show is what the
difference axis already reports.

Both arms panels are drawn on the same log scale. A duration axis extended to
its physical zero is honest but not comparable: it pins the cheap arm into the
floor of its own panel and leaves the panel's upper half empty, and it makes
the visual size of the TTFT effect a function of a different scale treatment
than the TTLT effect beside it. On one shared treatment the two multiplicative
effects can be read against each other, and neither panel has a dead quadrant.

Each difference axis is drawn broken: a narrow stub carries the null (0 ms, no
change) at the block's left edge, then a break, then the bootstrap
distribution. The reader therefore sees the effect separated from the null
rather than having to trust a printed interval on an axis that never shows
zero, and the interval keeps the resolution it needs.

Hierarchy: the two numbers the figure exists to deliver are the only framed,
enlarged elements on the page. The exclusion caveat is set as one line of
ordinary footnote text in the house's caveat colour, so the caveat no longer
out-ranks the finding.

Every number drawn or printed is computed from the loaded pairs. Nothing
distributional is written here as a literal. The p5-p95 body/axis trim is
stated on the canvas; the full observed range of every arm is printed to stdout
for the LaTeX caption rather than spent as canvas prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.mlab import GaussianKDE
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

from _common import IEEE_DOUBLE_WIDTH, OKABE_ITO, REPO, record_provenance, save

sys.path.insert(0, str(REPO / "scripts" / "report_figures"))

from fig8_overhead import DATA_PATH, load_overhead_pairs  # noqa: E402
from style import EXION, bootstrap_ci, set_log_axis_plain  # noqa: E402

# The default 0.025 in crop leaves the left y-label effectively flush with the
# column trim, while the inner y-label sits far off the neighbouring spine. A
# wider uniform pad gives the outer label the same order of clearance as the
# inner one, so no element is trimmed tighter than the rest of the page.
plt.rcParams["savefig.pad_inches"] = 0.09


# --- one frame language for every boxed element on the page -----------------
# Every non-data mark on the page is drawn from EXION["structure"]: spines,
# frames, header strips, gridlines and footnote text share one grey ladder, so
# nothing structural competes with a data body for the reader's attention.
STRUCTURE = EXION["structure"]
FRAME_EDGE = STRUCTURE[3]
FRAME_LW = 0.6
SPINE_COLOR = STRUCTURE[3]
GRID_COLOR = STRUCTURE[2]
STRIP_FACE = STRUCTURE[0]
# One inset for every framed element. The legend's own borderpad is 0.6 em at
# 8 pt, so the drawn frames use the same 4.8 pt and no box on the page is
# padded tighter than its neighbours.
FRAME_PAD_PT = 4.8

# Direct is the engine we did not build, so it takes the EXION baseline mint
# with a structure-grey outline. Gateway is ours and takes the EXION method
# family: the light ramp tone as the body, a dark ramp tone as its outline.
ARM_HUE = {"direct": EXION["baseline"], "gateway": EXION["family"][0]}
ARM_EDGE = {"direct": STRUCTURE[3], "gateway": EXION["family"][2]}
ARM_LABEL = {"direct": "Direct", "gateway": "Gateway"}

# The paired-difference mark is the darkest ink on the page -- the deepest tone
# of the same family the Gateway arm is drawn from, since the estimate it
# carries is a statement about that arm. Its bootstrap body is the lightest
# object anywhere in the figure, so the structure greys alone separate a
# contextual body from a primary data body. What that body *is* is stated in
# its own legend key rather than left to the axis caption.
MARK_COLOR = EXION["family"][3]
DELTA_BODY_FACE = STRUCTURE[0]
DELTA_BODY_EDGE = STRUCTURE[3]
CI_LW = 1.6
CAP_HALF = 0.32  # in delta-axes y units
DELTA_BODY_HALF = 0.50
DELTA_YLIM = (-0.85, 3.20)
CAVEAT_COLOR = OKABE_ITO["vermillion"]

# Two estimators in two quantity spaces get two glyphs: a circle for a per-arm
# mean on a duration axis, a diamond for the paired mean difference on a
# difference axis. Nothing on the page carries one shape for both.
MARKER_PT = 5.0
DELTA_MARKER = "D"
DELTA_MARKER_PT = 5.2
VIOLIN_WIDTH = 0.72
MEAN_DODGE = 0.44  # x units; clears the widest half-violin (0.36) by 0.08
ARM_XLIM = (-0.52, 1.52)
MEDIAN_LW = 1.0
NOTE_PT = 8.0
LABEL_PT = 9.0
RESULT_PT = 12.0  # the headline difference: the largest type in the figure

# Violin support. A kernel evaluated over the full min..max support collapses
# to a sub-point sliver wherever the density is a few percent of its mode; on
# a heavy right tail that sliver reads as a stray rule, not as a density. The
# body is drawn over the central 90% of each arm and the y-axis is set to the
# same range, so one trim rule governs both panels instead of two.
VIOLIN_TRIM_Q = 5.0

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 1234
# Ink is measured at print resolution; the result is converted to figure
# fractions, so the placement it drives is independent of the output dpi.
MEASURE_DPI = 320
INK_THRESHOLD = 190  # darker than any structure grey used as a fill or rule

# Log ticks. A labelled 1-2-3-5-7 ladder plus an unlabelled 1..9 ladder, both
# complete in every decade the axis covers, so reference density is a property
# of the scale rather than of which decade an arm happens to land in and the
# two panels the reader must compare carry comparable numbers of labels.
LOG_MAJOR_MANTISSA = (1, 2, 3, 5, 7)
LOG_MINOR_MANTISSA = (1, 2, 3, 4, 5, 6, 7, 8, 9)
LOG_DECADES = range(0, 6)

# --- deterministic layout (figure fractions) --------------------------------
FIG_HEIGHT = 5.25
AX_LEFT, AX_RIGHT = 0.090, 0.985
BLOCK_GAP = 0.088
BLOCK_WIDTH = (AX_RIGHT - AX_LEFT - BLOCK_GAP) / 2.0
ARM_BOTTOM, ARM_TOP = 0.422, 0.808
DELTA_BOTTOM, DELTA_TOP = 0.154, 0.3445
# The difference axis is broken: a stub for the null, a visible gap, then the
# data. Both are cut from the block, so the block's own left and right edges --
# the ones the legend, the header strips and the arms panels also use -- are
# unchanged.
NULL_FRAC = 0.085
BREAK_FRAC = 0.065
# Header strip, in axes fractions of the arms panel: ~7 pt of gap first, so its
# bottom border reads as its own rule rather than doubling the axes top spine.
STRIP_GAP = 0.048
STRIP_HEIGHT = 0.103
LEGEND_TOP = 0.984
CAPTION_Y = 0.087
NOTE_BOTTOM = 0.015
# Baseline of the framed result callout, in axes fractions of its strip: high
# enough that the frame clears the bootstrap ridge by more than any interline
# gap on the page, low enough to clear its own top spine.
RESULT_Y = 0.60

# Both panels clear the drawn data by the same 6% of the drawn span at every
# end; with no linear-to-zero clamp anywhere, that is the only limit rule in
# the figure.
PAD_FRAC = 0.06
# A drawn tick must clear either limit by this fraction of the axis span, so no
# gridline is laid on top of a spine.
TICK_CLEARANCE = 0.035


def _frame(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LW)
        spine.set_color(SPINE_COLOR)


def _log_ladder(mantissas) -> list[float]:
    return [mantissa * 10.0 ** decade
            for decade in LOG_DECADES for mantissa in mantissas]


def _bootstrap_mean_distribution(values: np.ndarray) -> np.ndarray:
    """The resampled means behind the interval, drawn from the same recipe.

    style.bootstrap_ci returns only the two percentiles; the difference axis
    also shows the distribution they are cut from, so the resampling is
    repeated here with the identical generator and checked against the helper's
    own output. If the two ever diverge the figure refuses to build rather than
    drawing a body that does not belong to its interval.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_N, dtype=float)
    for index in range(BOOTSTRAP_N):
        estimates[index] = float(np.mean(
            rng.choice(values, size=values.size, replace=True)))
    reference = bootstrap_ci(values, np.mean, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    local = tuple(np.percentile(estimates, [2.5, 97.5]))
    if not np.allclose(reference, local):
        raise ValueError("local bootstrap does not reproduce style.bootstrap_ci")
    return estimates


class _DeltaMarkHandler(HandlerBase):
    """Legend key for the paired-difference mark, drawn with its own numbers.

    The key is the mark and nothing else: the same capped interval at the same
    line width, the same solid diamond with no casing. The bootstrap body is
    not packed in here -- it has a key of its own -- so the caps and the centre
    mark keep their separation instead of fusing into one blob at print size.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x0 = -xdescent
        y = -ydescent + height * 0.5
        low_x, high_x = x0 + width * 0.06, x0 + width * 0.94
        cap = height * 0.36
        artists = [
            Line2D([low_x, high_x], [y, y], color=MARK_COLOR, linewidth=CI_LW,
                   solid_capstyle="butt"),
            Line2D([low_x, low_x], [y - cap, y + cap], color=MARK_COLOR,
                   linewidth=CI_LW, solid_capstyle="butt"),
            Line2D([high_x, high_x], [y - cap, y + cap], color=MARK_COLOR,
                   linewidth=CI_LW, solid_capstyle="butt"),
            Line2D([x0 + width * 0.5], [y], linestyle="none",
                   marker=DELTA_MARKER, markerfacecolor=MARK_COLOR,
                   markeredgecolor=MARK_COLOR, markeredgewidth=0.0,
                   markersize=DELTA_MARKER_PT),
        ]
        for artist in artists:
            artist.set_transform(trans)
        return artists


def _violin_body(ax, values: np.ndarray, position: float, width: float,
                 arm: str) -> tuple[float, float]:
    """One violin: an opaque body with its own outline, plus a median rule.

    Opaque rather than translucent because a see-through body over a gridline
    reads as a shading gradient in the data that is not there. The density is
    estimated in log space, the space both axes are drawn in, so neither panel
    reports a linear-bandwidth kernel as if it were log density -- and because
    a log-space body and an arithmetic mean do not have to agree, the median is
    drawn too, so the reader can see the mean sitting above the mass of a
    skewed arm as a fact about the arm rather than as a mismatch between two
    plotting conventions.

    Returns the trimmed support, in log space, so the caller can set limits to
    exactly the range that carries a drawn body.
    """
    sample = np.log10(values)
    density = GaussianKDE(sample)
    low, high = np.percentile(sample, [VIOLIN_TRIM_Q, 100.0 - VIOLIN_TRIM_Q])
    grid = np.linspace(low, high, 512)
    half = density.evaluate(grid)
    half = half / density.evaluate(
        np.linspace(sample.min(), sample.max(), 512)).max() * (width / 2.0)
    ax.fill_betweenx(np.power(10.0, grid), position - half, position + half,
                     facecolor=ARM_HUE[arm],
                     edgecolor=ARM_EDGE[arm], linewidth=0.7, zorder=2.5)

    middle = float(np.median(sample))
    # Inset from the contour so the median rule stops short of the body's own
    # outline instead of running into it. Nothing else is drawn at this x: the
    # arm mean is dodged clear of the body, so the median needs no white casing
    # and the legend can show the median as the plain rule it actually is.
    reach = 0.88 * float(np.interp(middle, grid, half))
    ax.plot([position - reach, position + reach],
            [float(np.power(10.0, middle))] * 2,
            color=MARK_COLOR, linewidth=MEDIAN_LW, solid_capstyle="butt",
            zorder=4.5)
    return float(low), float(high)


def _arms_panel(ax, subset: dict[str, np.ndarray], metric: str) -> dict:
    """Draw one arms panel: two densities, their medians and their means."""
    direct = subset[f"{metric}_direct"]
    gateway = subset[f"{metric}_gateway"]
    positions = np.array([0.0, 1.0])

    # Closed frame: gridlines and the header strip both terminate on a spine
    # instead of hanging in mid-air at an open right edge.
    _frame(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR)
    # The two x positions are named by their labels; an inward tick adds no
    # information and puts a stroke on the bottom spine directly beneath each
    # violin, which is where a body edge would otherwise run into it.
    ax.tick_params(axis="x", length=0.0, pad=5.0)

    drawn = [_violin_body(ax, values, position, VIOLIN_WIDTH, arm)
             for position, values, arm
             in zip(positions, (direct, gateway), ("direct", "gateway"))]

    # The arm mean is dodged clear of its own violin instead of being cased in
    # white: a marker at the body centre lands on the median rule whenever the
    # two nearly coincide, and a white casing then punches a hole through the
    # rule to hide the collision rather than removing it. Each mean is dodged
    # outward, away from the other arm, so the marker nearest a violin is
    # always that violin's own.
    means = np.array([direct.mean(), gateway.mean()])
    ax.plot(positions + np.array([-MEAN_DODGE, MEAN_DODGE]), means,
            linestyle="none", marker="o",
            markerfacecolor=MARK_COLOR, markeredgecolor=MARK_COLOR,
            markeredgewidth=0.0, markersize=MARKER_PT, zorder=5,
            clip_on=False)

    ax.set_yscale("log")

    # Group means printed under their own column. Below the axes is the one
    # region guaranteed to hold no marks, so the readout needs no opaque halo
    # and cannot erase a violin contour or a gridline; the clearance is
    # identical in both panels by construction.
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{ARM_LABEL[arm]}\nmean {value:.0f} ms"
         for arm, value in zip(("direct", "gateway"), means)],
        fontsize=LABEL_PT,
    )
    ax.set_xlim(*ARM_XLIM)

    return {
        "direct_mean": float(means[0]),
        "gateway_mean": float(means[1]),
        "support": (min(bounds[0] for bounds in drawn),
                    max(bounds[1] for bounds in drawn)),
        "observed_range": (float(min(direct.min(), gateway.min())),
                           float(max(direct.max(), gateway.max()))),
    }


def _apply_log_limits(ax, support: tuple[float, float], span: float) -> int:
    """Give both panels the same number of decades, centred on their own data.

    A fixed vertical distance then means the same ratio in either panel, so the
    two multiplicative effects can be read against each other instead of
    against two different scale treatments. Each window is centred on its own
    trimmed support, so the shared span costs neither panel its clearance.
    """
    centre = 0.5 * (support[0] + support[1])
    lower, upper = centre - 0.5 * span, centre + 0.5 * span
    ax.set_ylim(10.0 ** lower, 10.0 ** upper)
    # A rung that would land within TICK_CLEARANCE of a limit is dropped rather
    # than drawn on top of the spine it sits beside.
    margin = TICK_CLEARANCE * span

    def _inside(value: float) -> bool:
        return lower + margin <= np.log10(value) <= upper - margin

    majors = [value for value in _log_ladder(LOG_MAJOR_MANTISSA) if _inside(value)]
    set_log_axis_plain(ax, "y", majors, fmt=lambda value: f"{value:g}")
    minors = [value for value in _log_ladder(LOG_MINOR_MANTISSA)
              if _inside(value) and value not in majors]
    ax.yaxis.set_minor_locator(FixedLocator(minors))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", direction="in", length=1.8,
                   width=0.5, color=SPINE_COLOR)
    # The unlabelled rungs are carried across the panel as well as onto the
    # spine: on a 1-2-3-5-7 label ladder the widest unlabelled interval is a
    # factor of 1.4, so no band of either panel is left without a reference.
    ax.yaxis.grid(True, which="minor", color=GRID_COLOR,
                  linewidth=0.3, alpha=0.42)
    return len(majors)


def _null_stub(ax) -> None:
    """The broken-axis stub that carries the null the difference is judged against.

    A difference axis whose limits are set by a bootstrap distribution never
    contains zero, so without this stub the reader has to take "the interval
    excludes no-change" on trust. The stub shows the null, the break shows that
    the intervening range is elided, and the distance from the null to the mark
    is then something the reader can see rather than infer.
    """
    _frame(ax)
    ax.set_ylim(*DELTA_YLIM)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["0"], fontsize=LABEL_PT)
    ax.tick_params(axis="x", labelsize=LABEL_PT)
    ax.plot([0.0, 0.0], [DELTA_YLIM[0], DELTA_YLIM[1]], color=MARK_COLOR,
            linewidth=0.9, linestyle=(0.0, (3.0, 2.0)), zorder=3)


def _break_marks(fig, x_center: float, y_bottom: float, y_top: float) -> None:
    """The two slashes that say the axis is cut between the null and the data.

    Sized to the gap, not to the glyph: the pair spans a little over half the
    break so both slashes keep clear white on either side of them rather than
    running into the stub's spine or the data axis's.
    """
    height = y_top - y_bottom
    reach = 0.0022 + 0.0055
    clearance = 0.5 * BREAK_FRAC * BLOCK_WIDTH - reach
    if clearance < 2.5 / 72.0 / IEEE_DOUBLE_WIDTH:
        raise ValueError("break marks do not clear the spines beside them")
    for offset in (-0.0022, 0.0022):
        fig.add_artist(Line2D(
            [x_center + offset - 0.0055, x_center + offset + 0.0055],
            [y_bottom + 0.42 * height, y_bottom + 0.58 * height],
            transform=fig.transFigure, color=SPINE_COLOR, linewidth=0.8,
            solid_capstyle="butt", zorder=6, clip_on=False))


def _delta_panel(ax, delta: np.ndarray) -> dict:
    """Draw the paired-difference axis: the estimate the figure claims.

    One estimand, one interval. The interval is drawn with end caps and the
    diamond carries no white casing, so the interval reads as one object with a
    measurable extent rather than as three fragments separated by a halo. Its
    numbers are not printed loose beside the ridge; they go in the one framed
    callout the page carries.
    """
    _frame(ax)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_COLOR)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0.0)
    ax.tick_params(axis="x", labelsize=LABEL_PT)

    estimates = _bootstrap_mean_distribution(delta)
    low, high = bootstrap_ci(delta, np.mean, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    mean = float(delta.mean())

    grid = np.linspace(estimates.min(), estimates.max(), 512)
    half = GaussianKDE(estimates).evaluate(grid)
    half = half / half.max() * DELTA_BODY_HALF
    ax.fill_between(grid, -half, half, facecolor=DELTA_BODY_FACE,
                    edgecolor=DELTA_BODY_EDGE, linewidth=0.7, zorder=2)

    ax.plot([low, high], [0.0, 0.0], color=MARK_COLOR, linewidth=CI_LW,
            solid_capstyle="butt", zorder=4)
    for bound in (low, high):
        ax.plot([bound, bound], [-CAP_HALF, CAP_HALF], color=MARK_COLOR,
                linewidth=CI_LW, solid_capstyle="butt", zorder=4)
    ax.plot([mean], [0.0], linestyle="none", marker=DELTA_MARKER,
            markerfacecolor=MARK_COLOR, markeredgecolor=MARK_COLOR,
            markeredgewidth=0.0, markersize=DELTA_MARKER_PT, zorder=5)

    ax.set_ylim(*DELTA_YLIM)
    span = float(estimates.max() - estimates.min())
    ax.set_xlim(estimates.min() - 0.10 * span, estimates.max() + 0.10 * span)
    return {"delta_mean": mean, "delta_mean_ci": (float(low), float(high))}


def _ink_bbox(fig, region: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Figure-fraction bounding box of the dark ink inside `region`.

    Measured off a raster rather than off a text's layout box: the layout box
    carries the font's ascent, descent and side bearings, so padding computed
    from it lands asymmetric on the page even when the numbers say it is even.
    What a reader sees is ink, so ink is what is measured.
    """
    previous = fig.dpi
    fig.set_dpi(MEASURE_DPI)
    fig.canvas.draw()
    raster = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    fig.set_dpi(previous)
    fig.canvas.draw()
    height, width = raster.shape[:2]
    col0 = max(0, int(np.floor(region[0] * width)))
    col1 = min(width, int(np.ceil(region[2] * width)))
    row0 = max(0, int(np.floor((1.0 - region[3]) * height)))
    row1 = min(height, int(np.ceil((1.0 - region[1]) * height)))
    mask = raster[row0:row1, col0:col1].min(axis=2) < INK_THRESHOLD
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("no ink found in the measured region")
    return ((col0 + cols.min()) / width,
            1.0 - (row0 + rows.max() + 1) / height,
            (col0 + cols.max() + 1) / width,
            1.0 - (row0 + rows.min()) / height)


def _result_callout(fig, ax, summary: dict) -> None:
    """The framed headline: the figure's finding, at the figure's largest size.

    The only framed, enlarged element on the page is the result. A caveat set
    in a coloured box while the finding is set at tick-label size inverts the
    hierarchy the reader is asked to take away, so the frame goes here and the
    caveat is demoted to one line of footnote.
    """
    mean = summary["delta_mean"]
    low, high = summary["delta_mean_ci"]
    gap = 0.012
    value = ax.text(0.5 - gap, RESULT_Y, f"{mean:+.0f} ms",
                    transform=ax.transAxes, ha="right", va="baseline",
                    fontsize=RESULT_PT, fontweight="bold", color=MARK_COLOR,
                    zorder=7)
    interval = ax.text(0.5 + gap, RESULT_Y, f"95% CI {low:.0f}–{high:.0f}",
                       transform=ax.transAxes, ha="left", va="baseline",
                       fontsize=NOTE_PT, color=MARK_COLOR, zorder=7)
    def _measure() -> tuple[float, ...]:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [artist.get_window_extent(renderer)
                 for artist in (value, interval)]
        return _ink_bbox(fig, (
            min(box.x0 for box in boxes) / fig.bbox.width - 0.006,
            min(box.y0 for box in boxes) / fig.bbox.height - 0.006,
            max(box.x1 for box in boxes) / fig.bbox.width + 0.006,
            max(box.y1 for box in boxes) / fig.bbox.height + 0.006))

    # The two parts have different widths, so splitting them at the axes centre
    # does not centre the pair. The composite is measured and then shifted, so
    # the frame lands on the strip's own centre line rather than near it.
    for _ in range(3):
        ink = _measure()
        frame = ax.get_window_extent()
        shift = ((0.5 * (frame.x0 + frame.x1) / fig.bbox.width)
                 - 0.5 * (ink[0] + ink[2])) * fig.bbox.width / frame.width
        if abs(shift) < 1e-4:
            break
        for artist in (value, interval):
            artist.set_x(artist.get_position()[0] + shift)
    ink = _measure()
    pad_x = FRAME_PAD_PT / 72.0 / fig.get_figwidth()
    pad_y = FRAME_PAD_PT / 72.0 / fig.get_figheight()
    rect = (ink[0] - pad_x, ink[1] - pad_y, ink[2] + pad_x, ink[3] + pad_y)
    # Added to the axes, not to the figure: a figure-level artist paints over
    # every axes regardless of zorder, which would bury the numbers the frame
    # exists to promote.
    ax.add_artist(Rectangle(
        (rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
        transform=fig.transFigure, clip_on=False, facecolor=STRUCTURE[2],
        edgecolor=FRAME_EDGE, linewidth=FRAME_LW, zorder=6.5))

    # The callout is centred on its own strip rather than on the estimate, so a
    # skewed bootstrap cannot push it through a spine; the fit is asserted
    # rather than assumed.
    frame = ax.get_window_extent()
    margin = 3.0 * fig.dpi / 72.0
    if (rect[0] * fig.bbox.width < frame.x0 + margin
            or rect[2] * fig.bbox.width > frame.x1 - margin
            or rect[3] * fig.bbox.height > frame.y1 - margin):
        raise ValueError("result callout does not fit inside its strip")


def _header(ax, title: str):
    """Light-grey strip carrying the panel's label."""
    bottom = 1.0 + STRIP_GAP
    ax.add_patch(Rectangle((0.0, bottom), 1.0, STRIP_HEIGHT, transform=ax.transAxes,
                           clip_on=False, facecolor=STRIP_FACE, edgecolor=FRAME_EDGE,
                           linewidth=FRAME_LW, zorder=3))
    return ax.text(0.5, bottom + STRIP_HEIGHT * 0.5, title, transform=ax.transAxes,
                   ha="center", va="center", fontsize=LABEL_PT, fontweight="bold",
                   zorder=4)


def _check_headers_fit(fig, headers: list) -> None:
    """A header that outgrows its strip is shortened by hand, never shrunk."""
    renderer = fig.canvas.get_renderer()
    for ax, label in headers:
        strip = ax.get_window_extent().width
        if label.get_window_extent(renderer).width > strip - 8.0:
            raise ValueError(f"header text wider than its strip: {label.get_text()!r}")


def _block_rects(index: int) -> tuple[list[float], list[float], list[float]]:
    left = AX_LEFT + index * (BLOCK_WIDTH + BLOCK_GAP)
    height = DELTA_TOP - DELTA_BOTTOM
    null_width = NULL_FRAC * BLOCK_WIDTH
    break_width = BREAK_FRAC * BLOCK_WIDTH
    arms = [left, ARM_BOTTOM, BLOCK_WIDTH, ARM_TOP - ARM_BOTTOM]
    null = [left, DELTA_BOTTOM, null_width, height]
    delta = [left + null_width + break_width, DELTA_BOTTOM,
             BLOCK_WIDTH - null_width - break_width, height]
    return arms, null, delta


def build_paper_figure(pairs: dict):
    fig = plt.figure(figsize=(IEEE_DOUBLE_WIDTH, FIG_HEIGHT))
    arms_axes, null_axes, delta_axes = [], [], []
    for index in range(2):
        arms_rect, null_rect, delta_rect = _block_rects(index)
        arms_axes.append(fig.add_axes(arms_rect))
        null_axes.append(fig.add_axes(null_rect))
        delta_axes.append(fig.add_axes(delta_rect))

    ttft = _arms_panel(arms_axes[0], pairs["all"], "ttft")
    ttlt = _arms_panel(arms_axes[1], pairs["matched"], "ttlt")
    summaries_by_panel = (("ttft", ttft), ("ttlt", ttlt))
    # One span for both panels: the wider requirement wins, so neither panel is
    # cropped tighter than the 6% clearance rule allows.
    shared_span = (1.0 + 2.0 * PAD_FRAC) * max(
        summary["support"][1] - summary["support"][0]
        for _, summary in summaries_by_panel)
    for ax, (_, summary) in zip(arms_axes, summaries_by_panel):
        summary["n_labelled_ticks"] = _apply_log_limits(
            ax, summary["support"], shared_span)
    for ax in null_axes:
        _null_stub(ax)
    ttft.update(_delta_panel(delta_axes[0], pairs["all"]["ttft_delta"]))
    ttlt.update(_delta_panel(delta_axes[1], pairs["matched"]["ttlt_delta"]))
    summaries = {"ttft": ttft, "ttlt": ttlt}
    arms_axes[0].set_ylabel("TTFT (ms, log scale)")
    arms_axes[1].set_ylabel("TTLT (ms, log scale)")

    for index, metric in enumerate(("ttft", "ttlt")):
        left = AX_LEFT + index * (BLOCK_WIDTH + BLOCK_GAP)
        fig.text(left + BLOCK_WIDTH * 0.5, CAPTION_Y,
                 f"$\\Delta$mean {metric.upper()} (ms)", ha="center",
                 va="bottom", fontsize=LABEL_PT)

    paired_n = int(pairs["all"]["ttft_direct"].size)
    matched_n = int(pairs["matched"]["ttlt_direct"].size)
    # A header is a label, not a sentence: it names the panel and the selection
    # rule behind it in as few words as that takes. The counts those rules
    # yielded are numbers, so they belong in the disclosure block below with
    # the other numbers a reader may want to check, not in a strip heading.
    headers = [
        (arms_axes[0], _header(arms_axes[0], "(a) Paired TTFT")),
        (arms_axes[1], _header(arms_axes[1], "(b) Matched TTLT")),
    ]

    # Colour-to-condition mapping is stated, not implied: one swatch per arm,
    # one swatch for the bootstrap body so no reader has to map that grey onto
    # the Direct arm, and every mark key drawn with the same numbers as the
    # mark it names. Three columns of two rows rather than one row of six, so
    # the composite interval key has room to read as an interval.
    delta_handle = Line2D([], [], linestyle="none")
    handles = [
        Patch(facecolor=ARM_HUE["direct"],
              edgecolor=ARM_EDGE["direct"], linewidth=0.7),
        Patch(facecolor=ARM_HUE["gateway"],
              edgecolor=ARM_EDGE["gateway"], linewidth=0.7),
        Line2D([], [], color=MARK_COLOR, linewidth=MEDIAN_LW,
               solid_capstyle="butt"),
        Line2D([], [], linestyle="none", marker="o", markerfacecolor=MARK_COLOR,
               markeredgecolor=MARK_COLOR, markeredgewidth=0.0,
               markersize=MARKER_PT),
        Patch(facecolor=DELTA_BODY_FACE, edgecolor=DELTA_BODY_EDGE,
              linewidth=0.7),
        delta_handle,
    ]
    labels = [
        "Direct density", "Gateway density",
        "arm median", "arm mean (offset)",
        f"{BOOTSTRAP_N}× bootstrap $\\Delta$mean density",
        "paired $\\Delta$mean ±95% CI",
    ]
    # Pinned to the panel block rather than centred on it: the legend frame,
    # both header strips and all four blocks then share one left edge and one
    # right edge, so no framed band on the page has a width of its own.
    legend = fig.legend(
        handles=handles, labels=labels, loc="upper left",
        bbox_to_anchor=(AX_LEFT, LEGEND_TOP, AX_RIGHT - AX_LEFT, 0.0),
        bbox_transform=fig.transFigure, mode="expand", ncols=3, fontsize=NOTE_PT,
        frameon=True, framealpha=1.0, edgecolor=FRAME_EDGE, borderpad=0.6,
        borderaxespad=0.0, handlelength=2.6, handleheight=1.2,
        handletextpad=0.6, columnspacing=1.2, labelspacing=0.7,
        handler_map={delta_handle: _DeltaMarkHandler()},
    )
    legend.get_frame().set_linewidth(FRAME_LW)
    legend.get_frame().set_boxstyle("square", pad=0.0)

    fig.canvas.draw()
    _check_headers_fit(fig, headers)
    for index, metric in enumerate(("ttft", "ttlt")):
        _result_callout(fig, delta_axes[index], summaries[metric])
        _break_marks(fig,
                     AX_LEFT + index * (BLOCK_WIDTH + BLOCK_GAP)
                     + (NULL_FRAC + BREAK_FRAC * 0.5) * BLOCK_WIDTH,
                     DELTA_BOTTOM, DELTA_TOP)

    # Disclosure block: two footnote lines on one grid line, the trim rule that
    # governs both panels and the exclusion the reader must hold against panel
    # (b). The caveat keeps the house's caveat colour and loses its frame: it
    # is a limit on the finding, not the finding.
    dropped_n = int(pairs["dropped"])
    fig.text(AX_LEFT, NOTE_BOTTOM + 0.032,
             f"(a) {paired_n} paired requests, (b) {matched_n} equal-token "
             "pairs; violin bodies and y-axes trimmed to "
             f"p{VIOLIN_TRIM_Q:g}–p{100 - VIOLIN_TRIM_Q:g} of each arm.",
             ha="left", va="bottom", fontsize=NOTE_PT, color=STRUCTURE[3])
    fig.text(AX_LEFT, NOTE_BOTTOM,
             f"(b) {dropped_n} of {paired_n} pairs dropped for unequal output "
             "tokens; direction of the residual bias is unknown.",
             ha="left", va="bottom", fontsize=NOTE_PT, color=CAVEAT_COLOR)

    all_delta = pairs["all"]["ttlt_delta"]
    matched_delta = pairs["matched"]["ttlt_delta"]
    dropped_delta = float((all_delta.sum() - matched_delta.sum()) / dropped_n)
    return fig, summaries, dropped_delta


def main() -> None:
    pairs = load_overhead_pairs()
    fig, summaries, dropped_delta = build_paper_figure(pairs)
    save(fig, "overhead.pdf")
    plt.close(fig)
    record_provenance("overhead.pdf", [Path(DATA_PATH)])

    for metric, subset in (("ttft", "all"), ("ttlt", "matched")):
        s = summaries[metric]
        direct = pairs[subset][f"{metric}_direct"]
        gateway = pairs[subset][f"{metric}_gateway"]
        # Both ratios are reported here and neither is drawn: the ratio of the
        # two arm means is an unpaired estimand with no interval, and the
        # paired ratio is a different number (its own geometric mean), so
        # printing either beside the paired difference would put two estimands
        # in one result line.
        log_ratio = np.log(gateway / direct)
        paired_low, paired_high = bootstrap_ci(log_ratio, np.mean,
                                               n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        print(f"{metric.upper()} direct_mean_ms={s['direct_mean']:.1f} "
              f"gateway_mean_ms={s['gateway_mean']:.1f} "
              f"paired_delta_mean_ms={s['delta_mean']:.1f} "
              f"CI95=[{s['delta_mean_ci'][0]:.1f}, {s['delta_mean_ci'][1]:.1f}] "
              f"| not drawn: unpaired_mean_ratio="
              f"{s['gateway_mean'] / s['direct_mean']:.2f} "
              f"paired_geometric_ratio={np.exp(log_ratio.mean()):.2f} "
              f"CI95=[{np.exp(paired_low):.2f}, {np.exp(paired_high):.2f}]")
    # For the LaTeX caption: the trim is disclosed on the canvas, the exact
    # range the trim hides is disclosed here so it costs no canvas prose.
    print("caption: full observed range "
          + ", ".join(f"({panel}) {summaries[key]['observed_range'][0]:.0f}–"
                      f"{summaries[key]['observed_range'][1]:.0f} ms"
                      for panel, key in (("a", "ttft"), ("b", "ttlt")))
          + "; labelled log ticks "
          + ", ".join(f"({panel}) {summaries[key]['n_labelled_ticks']}"
                      for panel, key in (("a", "ttft"), ("b", "ttlt"))))
    print(f"TTLT matched pairs dropped: {pairs['dropped']} "
          f"(their delta_mean_ms={dropped_delta:.1f}; not drawn — the arms "
          f"emitted unequal output tokens, so this delta is confounded)")


if __name__ == "__main__":
    main()
