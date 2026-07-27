"""Figure 4 - ranking quality by predictor input family.

A horizontal dot-and-interval (forest) plot rather than bars. Kendall's tau is
an estimate with an interval, not a quantity accumulated from zero: bars spend
most of their ink on the uninteresting distance from zero, and truncating them
to fix that exaggerates the differences.

Layout contract (one claim, one panel, no in-plot prose):

* Every arm the artifact scored is drawn. The row set is derived from the
  artifact's own ``results`` keys, so a new arm cannot be dropped by a stale
  tuple in this file, and the reader is never asked to trust that the five or
  six rows shown are all of them.
* Two uncertainty units are drawn, and each one is named for what it actually
  is. The capped bar is the bootstrap interval the artifact carries, and that
  interval is computed over ONE seed (``ci95_seed17``: t1_strata.py resamples
  sessions of ``common.SEEDS[0]``), while the dot is the mean over all three
  seeds. Those are two estimators, so the key says so - the interval is
  labelled "seed 17 only" and the dot "mean of 3 seeds". Nothing here pretends
  the dot is the centre of the bar; where it is not, the key explains why.
* The thin bar in the lane below each row is the min-max across seeds. Rows
  whose model is deterministic given the split have a zero-wide seed range,
  drawn as a mark no heavier than the bar it replaces: a degenerate range is
  the absence of information and must not out-ink the ranges that carry some.
* Individual seeds are never plotted as separate ticks. With three seeds the
  ticks collide at this scale, and for the deterministic baselines they are
  coincident by construction.
* No vertical rules of any kind inside the panel. A rule on a quantitative axis
  reads as a threshold, so the two statements the geometry cannot make (which
  gap clears the pre-registered bars, and which comparisons are not separable)
  are carried by the header strip, out of the data region.
* Both directions of the separability caveat are stated. The artifact's frozen
  criteria (``pre_registered_criteria``) are read from the JSON, evaluated
  here, and printed on the strip for BOTH the arm that clears them and the arm
  that does not, so no comparison is silently implied to be separable by being
  the one the caveat leaves out. The vermillion line is that caveat, and it
  sits inside the strip's frame rather than orphaned under the key.
* One value-label column, right-aligned on the panel's right edge, beyond the
  last tick. Values are printed to two decimals: the intervals on this panel
  overlap at the third, so a third decimal would invite a ranking the figure
  itself says it cannot support.
* The x axis is truncated (tau is bounded and no row is near zero); the break
  is drawn on the spine, the spine stops at the last tick, and the axis label
  is centred on that drawn spine rather than on the padded axes.
* Ink weight follows the argument: the proposed input and its ablation carry
  the heavier marks, and the baseline greys are lighter than either, so no
  baseline row out-inks a system it loses to. Greys are assigned by rank of the
  plotted value, and arms with the same value get the same grey.
* One spacing unit (5 pt) sets every frame pad, the key's handle gap and the
  gap between blocks; the value column uses twice that. Frames share the
  panel's left and right edge, so the strip, the key and the plot form one
  column with no ragged side.
"""

from __future__ import annotations

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from _common import (
    COLOR,
    IEEE_SINGLE_WIDTH,
    LABEL,
    OFFLINE,
    OKABE_ITO,
    load_json,
    record_provenance,
    save,
)
from style import EXION

T1 = OFFLINE / "t1-strata.json"
# The generator that produced the artifact. The resample count that defines the
# intervals is a property of that code, not of the JSON it wrote, so it is read
# from the source rather than restated here as a literal.
T1_SRC = OFFLINE / "t1_strata.py"
# The proposed input and its schema-text ablation. Every other arm the artifact
# scored is a baseline; that set is derived from the data, never listed here.
OURS = ("bert_prompt_schema", "bert_prompt_only")
# Short row labels for arms _common does not name. The artifact's own label for
# this arm ("schema-hash categorical (E1a)") carries an internal ticket code.
EXTRA_LABEL = {"schema_hash_categorical": "Schema-hash categorical"}
PITCH = 0.62          # row pitch within a group (tight forest rhythm)
GROUP_GAP = 0.40      # extra separation between the BERT and baseline groups
SEED_LANE = 0.22      # seed-range lane, below the row centre and clear of it
BANNER_FRAC = 0.200   # figure fraction reserved at the top for the claim strip
KEY_FRAC = 0.150      # figure fraction reserved at the bottom for the key
# Frames, header strips and gridlines are structure, not data: they take the
# EXION structure greys, so nothing that is merely scaffolding can be mistaken
# for a series.
FRAME_EDGE = EXION["structure"][3]
STRIP_FILL = EXION["structure"][0]
GRID_COLOR = EXION["structure"][0]
# The caveat is the single vermillion mark in the figure: the palette reserves
# that hue for degraded or overstatement semantics, and "these comparisons are
# not separable" is exactly the guard against overstating the ranking the rows
# appear to give.
CAVEAT = COLOR["overstate"]
# The scalar/identity baselines are structure, not a method family: they take
# the EXION structure greys, lightest at the top of the y axis and darkening
# downward. Arms that tie share a step, because they are the same number. Every
# step is lighter than the family ramp it sits under, so no baseline row
# out-inks a system that beats it.
BASELINE_GREYS = tuple(EXION["structure"][1:])
# Key glyphs are annotation, not data: they are drawn in the annotation ink, so
# no key mark is a clone of one plotted series' colour.
KEY_INK = OKABE_ITO["black"]
BODY_PT = 8.0
HEAD_PT = 10.0        # a real size step, so weight is not the only hierarchy
PAD_PT = 5.0          # frame pad = key handle gap = gap between blocks
COLUMN_GAP_PT = 2.0 * PAD_PT   # the one doubled gap: value column off the data
XMIN = 0.350          # left end of the truncated axis, left of every interval
XTICKS = (0.4, 0.5, 0.6, 0.7)
OURS_CI_LW = 2.2      # inferential mark, proposed family: the heaviest ink
BASE_CI_LW = 1.0      # inferential mark, baselines: lighter than the blues
OURS_MS = 5.2
BASE_MS = 3.4
SEED_LW = 1.0         # descriptive mark: lighter than the interval above it
ZERO_SPREAD_MS = 1.4  # degenerate seed range: no heavier than a range bar
# The whole set is one family (style.py: DejaVu Sans, with a custom mathtext
# set mapped onto it). DejaVu carries a real bold file, so the headline needs no
# fallback family of its own - the Arial that used to stand in for Helvetica's
# missing bold is exactly the second family the cross-figure audit flagged.
HEADLINE_FAMILY = ["DejaVu Sans"]
# Frames are placed from ink measured on a raster; unhinted metrics make that
# raster agree with the vector backend that actually writes the figure.
mpl.rcParams["text.hinting"] = "none"


def row_label(name: str) -> str:
    if name in LABEL:
        return LABEL[name]
    if name in EXTRA_LABEL:
        return EXTRA_LABEL[name]
    raise KeyError(f"no short row label for {name}")


def seed_deltas(results: dict, better: str, worse: str) -> list[float]:
    """Paired per-seed differences, over seeds present in both arms."""
    left = results[better]["all"]["per_seed_tau_b"]
    right = results[worse]["all"]["per_seed_tau_b"]
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError(f"no shared seeds between {better} and {worse}")
    return [float(left[s]) - float(right[s]) for s in seeds]


def interval_of(cell: dict) -> tuple[str, float, float]:
    """The artifact's bootstrap interval, with the seed it was computed on.

    The artifact names the interval after that seed (``ci95_seed17``), and the
    key on the figure has to say the same thing, so the seed is parsed from the
    key rather than assumed, and checked against the per-seed table.
    """
    keys = [key for key in cell if key.startswith("ci95")]
    if len(keys) != 1:
        raise ValueError(f"expected exactly one ci95 field, found {keys}")
    match = re.fullmatch(r"ci95_seed(\d+)", keys[0])
    if match is None:
        raise ValueError(f"cannot tell which seed {keys[0]} was computed on")
    seed = match.group(1)
    if seed not in cell["per_seed_tau_b"]:
        raise ValueError(f"interval seed {seed} is not a scored seed")
    low, high = (float(value) for value in cell[keys[0]])
    return seed, low, high


def material_bar(payload: dict) -> float:
    """The pre-registered material-effect bar, read from the artifact."""
    criteria = payload["pre_registered_criteria"]["criteria_text"]
    if not criteria.get("frozen_before_results"):
        raise ValueError("criteria were not frozen before results")
    if "CI separation" not in criteria["primary"]:
        raise ValueError(f"unexpected primary criterion: {criteria['primary']}")
    match = re.search(r">=\s*([0-9.]+)", criteria["secondary"])
    if match is None:
        raise ValueError(f"no threshold in: {criteria['secondary']}")
    return float(match.group(1))


def bootstrap_resamples(path) -> int:
    """Replicate count of the CI in the artifact, read from its generator.

    The count is the parameter that defines the interval, so it is parsed from
    the code that produced the numbers and the call sites are checked for an
    override rather than being trusted to use the default.
    """
    source = path.read_text(encoding="utf-8")
    signature = re.search(r"def bootstrap_ci\(([^()]*)\)", source)
    if signature is None:
        raise ValueError(f"no bootstrap_ci definition in {path}")
    count = re.search(r"iterations\s*=\s*(\d+)", signature.group(1))
    if count is None:
        raise ValueError("bootstrap_ci has no iterations parameter")
    calls = re.findall(r"(?<!def )bootstrap_ci\(([^()]*)\)", source)
    if not calls:
        raise ValueError("bootstrap_ci is never called")
    if any("iterations" in call for call in calls):
        raise ValueError("a call site overrides the resample count")
    return int(count.group(1))


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    record = payload["baseline_of_record"]["model"]
    n_boot = bootstrap_resamples(T1_SRC)
    bar = material_bar(payload)

    def mean_of(name: str) -> float:
        return float(results[name]["all"]["mean_tau_b"])

    # Every scored arm is a row. The baseline family is whatever the artifact
    # holds that is not one of ours, so an arm cannot be dropped here silently.
    missing = [name for name in OURS if name not in results]
    if missing:
        raise ValueError(f"proposed arms absent from the artifact: {missing}")
    ours = tuple(sorted(OURS, key=mean_of, reverse=True))
    baselines = tuple(sorted(sorted(set(results) - set(OURS)),
                             key=mean_of, reverse=True))
    rows = ours + baselines
    if set(rows) != set(results):
        raise ValueError("row set is not the artifact's model set")

    # Greys by rank of the plotted value; equal values get equal ink.
    steps = sorted({f"{mean_of(name):.12f}" for name in baselines}, reverse=True)
    if len(steps) > len(BASELINE_GREYS):
        raise ValueError(f"{len(steps)} distinct baseline values, "
                         f"{len(BASELINE_GREYS)} greys")
    grey_of = dict(zip(steps, BASELINE_GREYS))
    # One family ramp, not one hue per row: ramp position is how much of the
    # ordering signal the input owns, so the proposed input takes the darkest
    # step and its schema-text ablation a lighter one of the same ramp.
    colour_of = {ours[0]: EXION["family"][3], ours[1]: EXION["family"][1]}
    colour_of.update({name: grey_of[f"{mean_of(name):.12f}"]
                      for name in baselines})
    weight_of = {name: (OURS_CI_LW if name in ours else BASE_CI_LW)
                 for name in rows}
    size_of = {name: (OURS_MS if name in ours else BASE_MS) for name in rows}
    # The strip's delta lines are written against the top row, so the top row
    # is checked against the data rather than assumed.
    if max(rows, key=mean_of) != rows[0] or rows[0] != "bert_prompt_schema":
        raise ValueError("top row is no longer the prompt+schema input")

    seed_sets = {name: set(results[name]["all"]["per_seed_tau_b"]) for name in rows}
    seeds = set.intersection(*seed_sets.values())
    if any(seed_sets[name] != seeds for name in rows):
        raise ValueError("rows do not share one seed set")
    n_seeds = len(seeds)

    intervals = {name: interval_of(results[name]["all"]) for name in rows}
    ci_seeds = {seed for seed, _, _ in intervals.values()}
    if len(ci_seeds) != 1:
        raise ValueError(f"intervals mix seeds: {sorted(ci_seeds)}")
    ci_seed = ci_seeds.pop()

    # The frames below are padded against rendered *ink*, so the raster the
    # measurement runs on is drawn fine enough for that to be exact.
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_WIDTH, 3.45), dpi=200,
                           layout="constrained")
    # The strip's text column runs from the row-label edge to the panel edge, so
    # every point of side margin is a point the claim lines do not get. The rect
    # keeps only the margin the frame's own pad needs (the frame is drawn one
    # PAD_PT outside the text, and savefig crops to that frame anyway).
    fig.get_layout_engine().set(
        rect=(0.008, KEY_FRAC, 0.984, 1.0 - KEY_FRAC - BANNER_FRAC))
    # One type size for every glyph the panel carries: the row label, the value
    # label and the tick label sit on the same rows as each other, so they are
    # set at the body size the strip and the key already use.
    ax.tick_params(axis="both", labelsize=BODY_PT)

    # Manual y positions: tight pitch inside a group, a gap between groups.
    ys, cursor = [], 0.0
    for index, name in enumerate(rows):
        if index == len(ours):
            cursor += GROUP_GAP
        ys.append(cursor)
        cursor += PITCH
    ys = np.asarray(ys)

    for name, row_y in zip(rows, ys):
        cell = results[name]["all"]
        mean = float(cell["mean_tau_b"])
        _, low, high = intervals[name]
        colour = colour_of[name]
        # The bar is the artifact's seed-CI and the dot is the multi-seed mean,
        # so the bar is drawn as a span between its own ends rather than as an
        # error bar hung off the dot: the geometry must not claim the dot is
        # the centre of an interval that was not computed around it.
        ax.plot([low, high], [row_y, row_y], color=colour,
                linewidth=weight_of[name], solid_capstyle="butt", zorder=3)
        cap = 2.6
        for end in (low, high):
            ax.plot([end], [row_y], marker="|", markersize=cap,
                    markeredgewidth=weight_of[name], color=colour, zorder=3)
        # The marker is kept smaller than the shortest interval on the panel so
        # neither end of a bar is ever occluded by the dot.
        ax.plot([mean], [row_y], marker="o", markersize=size_of[name],
                markeredgecolor="white", markeredgewidth=0.5,
                color=colour, zorder=4)
        # Seed lane: one bar per row spanning the min-max of the seed means.
        # Its own lane, so it can never touch the mean marker above it, and
        # lighter than the interval, so the descriptive range cannot outweigh
        # the inferential one.
        per_seed = [float(v) for v in cell["per_seed_tau_b"].values()]
        seed_low, seed_high = min(per_seed), max(per_seed)
        if seed_high - seed_low > 1e-12:
            ax.plot([seed_low, seed_high], [row_y + SEED_LANE] * 2,
                    color=colour, linewidth=SEED_LW, solid_capstyle="butt",
                    zorder=3)
        else:
            # Deterministic given the split: the range is a point. It is drawn
            # at the weight of the bars it replaces, not as a tall tick - an
            # absent range must not be the most salient mark in the lane.
            ax.plot([mean], [row_y + SEED_LANE], marker="|",
                    markersize=ZERO_SPREAD_MS, markeredgewidth=SEED_LW,
                    color=colour, zorder=3)

    baseline = mean_of(record)
    proposed = mean_of(rows[0])
    ablation = mean_of(rows[1])

    # Row labels left-aligned on a common edge; no y tick marks.
    ax.set_yticks(ys)
    ax.set_yticklabels([row_label(name) for name in rows])
    for label in ax.get_yticklabels():
        label.set_ha("left")
    ax.set_ylim(ys[-1] + 0.55, -0.45)

    n_test = int(payload["split_sizes"]["test"])
    # The row-label column takes ~100 pt of a 252 pt canvas, so the drawn spine
    # this label is centred on is short. "test split" is shortened to "test":
    # the split is named once, the sample size is unchanged, and the label now
    # clears the value column instead of running under it.
    ax.set_xlabel(f"Kendall $\\tau_b$ (test, $n$ = {n_test})",
                  fontsize=BODY_PT)
    ax.set_xlim(XMIN, 0.78)
    ax.set_xticks(list(XTICKS))
    # Gridlines are the lightest structure step, and lighter still than the
    # palest baseline row, so the faintest series is never confused with the
    # scaffolding it crosses.
    ax.xaxis.grid(True, color=GRID_COLOR)
    ax.set_axisbelow(True)

    # Left-align row labels on a common edge: pad derives from the widest
    # label's rendered extent plus a gap that keeps the longest label off the
    # spine, so the label column is neither ragged nor locally cramped.
    fig.canvas.draw()
    widest_px = max(lbl.get_window_extent().width
                    for lbl in ax.get_yticklabels())
    ax.tick_params(axis="y", length=0, pad=widest_px * 72.0 / fig.dpi + PAD_PT)
    fig.canvas.draw()

    px_per_pt = fig.dpi / 72.0

    def data_pt(points: float) -> float:
        """Convert a length in points to data-x units."""
        unit = (ax.transData.transform((1.0, 0.0))[0]
                - ax.transData.transform((0.0, 0.0))[0])
        return points * px_per_pt / unit

    # --- value-label column -------------------------------------------------
    # One column for all rows, right-aligned on the panel edge and placed
    # beyond the last tick, so the labels share one grammar and no gridline
    # crosses them. Two decimals: every interval on this panel is wider than a
    # third decimal, so printing one would rank rows the figure cannot rank.
    blended = blended_transform_factory(ax.transAxes, ax.transData)
    value_labels = []
    for name, row_y in zip(rows, ys):
        value_labels.append(ax.text(
            1.0, row_y, f"{mean_of(name):.2f}", ha="right",
            va="center_baseline", fontsize=BODY_PT, color=OKABE_ITO["black"],
            transform=blended, zorder=6))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    column_px = max(lbl.get_window_extent(renderer).width
                    for lbl in value_labels)
    column_gap_px = COLUMN_GAP_PT * px_per_pt
    for _ in range(3):
        axes_px = ax.get_window_extent(renderer).width
        share = (axes_px - column_px - column_gap_px) / axes_px
        if share <= 0.0:
            raise RuntimeError("value column does not fit the panel")
        ax.set_xlim(XMIN, XMIN + (XTICKS[-1] - XMIN) / share)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    # Vertical rhythm: one gap unit below the last seed lane, so the distance
    # from the bottom mark to the spine matches the gaps between blocks.
    for _ in range(3):
        unit_py = abs(ax.transData.transform((0.0, 1.0))[1]
                      - ax.transData.transform((0.0, 0.0))[1])
        ax.set_ylim(ys[-1] + SEED_LANE + PAD_PT * px_per_pt / unit_py,
                    ys[0] - (PAD_PT + OURS_MS / 2.0) * px_per_pt / unit_py)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    # The axes geometry is now final. Freezing the layout engine keeps it that
    # way through the frame placement below and through savefig, so the
    # measured positions are the positions that get written.
    fig.set_layout_engine("none")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # --- truncated axis: break glyph on the spine, spine stops at last tick --
    axes_box = ax.get_window_extent(renderer)
    break_x = XMIN + data_pt(3.0)
    break_step = data_pt(3.2)
    spine_start = break_x + break_step + data_pt(2.6)
    if spine_start >= min(low for _, low, _ in intervals.values()) - data_pt(2.0):
        raise RuntimeError("axis break collides with the leftmost interval")
    ax.spines["bottom"].set_bounds(spine_start, XTICKS[-1])
    ax.spines["left"].set_visible(False)
    slash_dy = 3.0 * px_per_pt / axes_box.height
    slash_dx = data_pt(1.9)
    axis_trans = ax.get_xaxis_transform()
    for offset in (0.0, break_step):
        ax.plot([break_x + offset - slash_dx, break_x + offset + slash_dx],
                [-slash_dy, slash_dy], transform=axis_trans, clip_on=False,
                color=mpl.rcParams["axes.edgecolor"],
                linewidth=mpl.rcParams["axes.linewidth"],
                solid_capstyle="butt", zorder=3)

    # The axis label belongs to the axis that is drawn, not to the padded axes
    # the value column forced: it is centred on the spine between the break and
    # the last tick.
    lo, hi = ax.get_xlim()
    label_x = ((spine_start + XTICKS[-1]) / 2.0 - lo) / (hi - lo)
    # Only the horizontal placement is being changed. The axis label's own
    # anchor y is kept: matplotlib holds it in display pixels while the label
    # is auto-placed, so it is converted rather than re-derived from the ink.
    label_anchor_y = ax.xaxis.label.get_position()[1]
    label_y = (label_anchor_y - axes_box.y0) / axes_box.height
    ax.xaxis.set_label_coords(label_x, label_y)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # The value column must clear the data it labels, not merely miss it.
    right_edge_px = max(ax.transData.transform((high, 0.0))[0]
                        for _, _, high in intervals.values())
    column_left_px = min(lbl.get_window_extent(renderer).x0
                         for lbl in value_labels)
    if column_left_px - right_edge_px < PAD_PT * px_per_pt:
        raise RuntimeError("value column crowds an interval")
    if ax.transData.transform((XTICKS[-1], 0.0))[0] > column_left_px:
        raise RuntimeError("a gridline reaches the value column")

    # --- claim strip and key ------------------------------------------------
    width_px, height_px = fig.bbox.width, fig.bbox.height
    pad_px = PAD_PT * px_per_pt
    gap_px = PAD_PT * px_per_pt
    pad_x = pad_px / width_px
    line_y = 11.0 * px_per_pt / height_px
    head_line_y = 13.5 * px_per_pt / height_px
    text_left = min(lbl.get_window_extent(renderer).x0
                    for lbl in ax.get_yticklabels()) / width_px
    frame_left = text_left - pad_x
    if frame_left < 0.0:
        raise RuntimeError("the frame runs off the left edge")
    # Every framed element ends where the panel ends, so the strip, the key and
    # the plot share one right edge as well as one left edge.
    frame_right = ax.get_window_extent(renderer).x1 / width_px
    inner_px = (frame_right - pad_x - text_left) * width_px

    def fits(artist) -> bool:
        return artist.get_window_extent(renderer).width <= inner_px

    def ink_box(boxes):
        """Rendered-ink extent inside the union of ``boxes`` (display units).

        Text layout boxes carry the glyphs' side bearings, so padding a frame
        against them leaves visibly different gaps on the two sides. The frames
        here are padded against the ink itself, which is what a reader (and a
        pixel-counting reviewer) actually sees.
        """
        buffer = np.asarray(fig.canvas.buffer_rgba())
        mask = buffer[:, :, :3].mean(axis=2) < 250
        x0 = max(int(min(b.x0 for b in boxes)) - 6, 0)
        x1 = min(int(max(b.x1 for b in boxes)) + 6, mask.shape[1])
        y0 = max(int(min(b.y0 for b in boxes)) - 6, 0)
        y1 = min(int(max(b.y1 for b in boxes)) + 6, mask.shape[0])
        window = mask[mask.shape[0] - y1:mask.shape[0] - y0, x0:x1]
        if not window.any():
            raise RuntimeError("no ink found where a frame was requested")
        lines_ = np.where(window.any(axis=1))[0]
        return y1 - lines_.max() - 1, y1 - lines_.min()

    def ink_of(artists):
        return ink_box([(getattr(a, "get_bbox_patch", lambda: None)() or a)
                        .get_window_extent(renderer) for a in artists])

    def shift(artists, delta):
        """Move a block by ``delta`` figure fractions."""
        for artist in artists:
            if isinstance(artist, Line2D):
                artist.set_ydata([y + delta for y in artist.get_ydata()])
            else:
                x, y = artist.get_position()
                artist.set_position((x, y + delta))

    def frame(ink_y0, ink_y1, facecolor="white") -> Rectangle:
        """One frame per block: square, house edge, panel width.

        The frame spans the panel's own left and right edge; the vertical pad
        equals the horizontal one in pixels, so the gap a reader measures
        between the frame and its nearest glyph is the same on every side.
        """
        y0 = (ink_y0 - pad_px) / height_px
        y1 = (ink_y1 + pad_px) / height_px
        patch = Rectangle((frame_left, y0), frame_right - frame_left, y1 - y0,
                          transform=fig.transFigure, facecolor=facecolor,
                          edgecolor=FRAME_EDGE, linewidth=0.6, zorder=5)
        fig.add_artist(patch)
        return patch

    total = seed_deltas(results, rows[0], record)
    pair_deltas = seed_deltas(results, rows[0], rows[1])
    agree = sum(1 for d in pair_deltas if d > 0)
    if agree != n_seeds:
        raise ValueError("the per-seed gap no longer holds for every seed")
    # A reader checks a headline delta by subtracting the two labels on the
    # panel. The deltas are therefore quoted at the precision at which that
    # subtraction reproduces them exactly, and the check is run here.
    shown = {name: float(f"{mean_of(name):.2f}") for name in rows}
    for gap, worse in ((proposed - baseline, record),
                       (proposed - ablation, rows[1])):
        if f"{gap:.2f}" != f"{shown[rows[0]] - shown[worse]:.2f}":
            raise RuntimeError("headline delta disagrees with the labels")

    # Both pre-registered criteria, evaluated here on the artifact's own
    # intervals, for the arm that clears them and the arm that does not.
    def separated(left: str, right: str) -> bool:
        return intervals[left][1] > intervals[right][2]

    if not (separated(rows[0], record) and proposed - baseline >= bar):
        raise ValueError("the baseline-of-record gap no longer clears the bars")
    if separated(rows[0], rows[1]) or proposed - ablation >= bar:
        raise ValueError("the ablation gap now clears a bar; redraw the claim")
    # The other statement the panel must not lose: the baselines are not
    # separable. Every pair of baseline intervals must overlap and every
    # baseline mean must sit inside every other baseline's interval.
    pairs = [(i, j) for i in range(len(baselines))
             for j in range(len(baselines)) if i != j]
    if any(separated(baselines[i], baselines[j])
           or not (intervals[baselines[j]][1] <= mean_of(baselines[i])
                   <= intervals[baselines[j]][2])
           for i, j in pairs):
        raise ValueError("baseline intervals now separate; redraw the claim")

    # Provisional placement at the top of the canvas, clear of the panel, so
    # the ink measurement below sees only the strip's own glyphs.
    banner_top = 0.995
    # The header is a label, never a finding: it names what the panel is about
    # and nothing more. Every result lives in the framed callout below it.
    head = fig.text(text_left, banner_top, "Ranking quality by input",
                    fontsize=HEAD_PT, fontweight="bold",
                    fontfamily=HEADLINE_FAMILY, ha="left", va="top",
                    color=OKABE_ITO["black"], zorder=6)
    # The claim line's right half would otherwise be dead space in a box whose
    # other lines run nearly full width, so the reading direction of the metric
    # is set there - it is the one thing a forest plot of a signed correlation
    # cannot show.
    tag = fig.text(frame_right - pad_x, banner_top, "higher is better",
                   fontsize=BODY_PT, ha="right", va="top",
                   color=OKABE_ITO["black"], zorder=6)
    # One artist per line, at a leading this file sets: multi-line strings are
    # laid out from the renderer's own font metrics, which differ between the
    # raster the frame is measured on and the vector file that is written.
    lines = [
        (f"+{proposed - baseline:.2f} vs {row_label(record)}: CIs separate,"
         f" clears pre-set {bar:g}", OKABE_ITO["black"]),
        (f"CIs overlap: BERT pair (+{proposed - ablation:.2f} < {bar:g})"
         f" and {len(baselines)} grey rows", CAVEAT),
    ]
    banner_lines = [head, tag]
    for index, (line, colour) in enumerate(lines):
        banner_lines.append(fig.text(
            text_left, banner_top - head_line_y - line_y * index, line,
            fontsize=BODY_PT, ha="left", va="top", color=colour, zorder=6))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ink_y0, ink_y1 = ink_of(banner_lines)
    delta = (axes_box.y1 + gap_px + pad_px - ink_y0) / height_px
    if ink_y1 + delta * height_px + pad_px > height_px:
        raise RuntimeError("the claim strip does not fit above the panel")
    shift(banner_lines, delta)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    banner = frame(ink_y0 + delta * height_px, ink_y1 + delta * height_px,
                   facecolor=STRIP_FILL)

    # The key names the two units and, for the interval, the estimator it was
    # actually computed on: the dot is a three-seed mean and the bar is a
    # single-seed bootstrap, so the key never calls them one thing.
    handle_px = 17.0 * px_per_pt
    key_text_left = text_left + (handle_px + pad_px) / width_px
    key_top = KEY_FRAC - 0.010
    key_rows = [
        f"dot: mean of {n_seeds} seeds;  bar: 95% CI, seed {ci_seed} only",
        f"lane: min–max over {n_seeds} seeds; tick = zero spread",
    ]
    key_texts, glyphs = [], []
    for index, text in enumerate(key_rows):
        y = key_top - line_y * index
        key_texts.append(fig.text(key_text_left, y, text, fontsize=BODY_PT,
                                  ha="left", va="top",
                                  color=OKABE_ITO["black"], zorder=6))
        mid = y - line_y * 0.34
        x0 = text_left
        x1 = text_left + handle_px / width_px
        if index == 0:
            # Dot and bar are drawn apart in the handle, not concentric: the
            # dot is not the bar's centre and the key must not suggest it is.
            glyphs.append(Line2D([x0], [mid], transform=fig.transFigure,
                                 marker="o", markersize=OURS_MS,
                                 markeredgecolor="white", markeredgewidth=0.5,
                                 color=KEY_INK, zorder=6))
            glyphs.append(Line2D([x0 + (x1 - x0) * 0.42, x1], [mid, mid],
                                 transform=fig.transFigure, color=KEY_INK,
                                 linewidth=OURS_CI_LW, solid_capstyle="butt",
                                 zorder=6))
        else:
            # Both seed-lane glyphs are shown, so the degenerate one is read
            # from the key rather than guessed at on the panel.
            glyphs.append(Line2D([x0, x0 + (x1 - x0) * 0.62], [mid, mid],
                                 transform=fig.transFigure, color=KEY_INK,
                                 linewidth=SEED_LW, solid_capstyle="butt",
                                 zorder=6))
            glyphs.append(Line2D([x1], [mid], transform=fig.transFigure,
                                 marker="|", markersize=ZERO_SPREAD_MS,
                                 markeredgewidth=SEED_LW,
                                 color=KEY_INK, zorder=6))
    for glyph in glyphs:
        fig.add_artist(glyph)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    key_ink_y0, key_ink_y1 = ink_of(key_texts + glyphs)
    label_ink_y0, _ = ink_box([ax.xaxis.label.get_window_extent(renderer)])
    key_delta = (label_ink_y0 - gap_px - pad_px - key_ink_y1) / height_px
    if key_ink_y0 + key_delta * height_px - pad_px < 0.0:
        raise RuntimeError("the key does not fit below the panel")
    shift(key_texts + glyphs, key_delta)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    key_frame = frame(key_ink_y0 + key_delta * height_px,
                      key_ink_y1 + key_delta * height_px,
                      facecolor=STRIP_FILL)

    # Collision guard: every annotation on this panel must keep a visible gap
    # from every other one, measured on the rendered extents rather than by
    # eye, and no line of prose may run past the frame it sits in.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def extent(artist):
        patch = getattr(artist, "get_bbox_patch", lambda: None)()
        if patch is not None:
            return patch.get_window_extent(renderer)
        return artist.get_window_extent(renderer)

    for name, artist in ([("headline", head)]
                         + [(f"banner {i}", t)
                            for i, t in enumerate(banner_lines[2:])]
                         + [(f"key {i}", t) for i, t in enumerate(key_texts)]):
        if not fits(artist):
            over = (artist.get_window_extent(renderer).width - inner_px) / px_per_pt
            raise RuntimeError(f"'{name}' runs past the frame edge by {over:.1f} pt "
                               f"(inner width {inner_px / px_per_pt:.1f} pt): "
                               f"{artist.get_text()!r}")
    if head.get_window_extent(renderer).x1 + pad_px > tag.get_window_extent(
            renderer).x0:
        raise RuntimeError("the headline crowds the reading-direction tag")

    boxed = ([(f"value {t.get_text()}", t) for t in value_labels]
             + [("banner", banner), ("key", key_frame)]
             + [("x label", ax.xaxis.label)]
             + [(f"row {t.get_text()}", t) for t in ax.get_yticklabels()]
             + [(f"tick {t.get_text()}", t) for t in ax.get_xticklabels()])
    min_gap = 2.0 * px_per_pt
    for index, (name_a, art_a) in enumerate(boxed):
        box_a = extent(art_a)
        for name_b, art_b in boxed[index + 1:]:
            box_b = extent(art_b)
            if (box_a.x0 < box_b.x1 - min_gap and box_b.x0 < box_a.x1 - min_gap
                    and box_a.y0 < box_b.y1 - min_gap
                    and box_b.y0 < box_a.y1 - min_gap):
                raise RuntimeError(f"overlap: '{name_a}' and '{name_b}'")
    # The axis label must also stay clear of the value column beside it.
    label_box = ax.xaxis.label.get_window_extent(renderer)
    if label_box.x1 > column_left_px - min_gap:
        raise RuntimeError(
            "the axis label reaches the value column by "
            f"{(label_box.x1 - column_left_px + min_gap) / px_per_pt:.1f} pt "
            f"(label width {label_box.width / px_per_pt:.1f} pt)")

    # Both framed elements and the panel must end on one pixel column.
    edges = [banner.get_window_extent(renderer).x1,
             key_frame.get_window_extent(renderer).x1,
             ax.get_window_extent(renderer).x1,
             max(lbl.get_window_extent(renderer).x1 for lbl in value_labels)]
    if max(edges) - min(edges) > 1.5:
        raise RuntimeError(f"right edges are ragged: {edges}")

    # The three vertical gaps a reader sees between blocks must be one gap:
    # strip to panel, panel to key, and the bottom mark to the spine.
    unit_py = abs(ax.transData.transform((0.0, 1.0))[1]
                  - ax.transData.transform((0.0, 0.0))[1])
    measured = {
        "strip to panel": banner.get_window_extent(renderer).y0 - axes_box.y1,
        "label to key": label_ink_y0 - key_frame.get_window_extent(renderer).y1,
        "last lane to spine": (ax.transData.transform(
            (0.0, ys[-1] + SEED_LANE))[1] - axes_box.y0),
    }
    if max(measured.values()) - min(measured.values()) > 1.5:
        raise RuntimeError(f"vertical rhythm is uneven: {measured}")

    # The seed lane must clear the interval it belongs to by more than the two
    # glyphs' own half-heights, so the two units can never fuse into one mark.
    clearance = SEED_LANE * unit_py - (OURS_MS / 2.0 + SEED_LW / 2.0) * px_per_pt
    if clearance < min_gap:
        raise RuntimeError("seed lane crowds the interval above it")

    save(fig, "ranking.pdf")
    record_provenance("ranking.pdf", [T1, T1_SRC])
    print(f"baseline of record {record}={baseline:.4f} proposed={proposed:.4f}")
    print(f"row order {rows}")
    print(f"grey ramp {dict(zip(baselines, (colour_of[n] for n in baselines)))}")
    print(f"interval = seed {ci_seed} bootstrap, {n_boot} session resamples; "
          f"dot = mean of {n_seeds} seeds; test n={n_test}")
    print(f"pre-registered material bar {bar:g}")
    print(f"delta vs record {proposed - baseline:.4f} "
          f"seed range [{min(total):.4f},{max(total):.4f}] "
          f"CIs separate={separated(rows[0], record)}")
    print(f"delta vs ablation {proposed - ablation:.4f} "
          f"seed range [{min(pair_deltas):.4f},{max(pair_deltas):.4f}] "
          f"{agree}/{n_seeds} seeds agree, "
          f"CIs separate={separated(rows[0], rows[1])}")
    print(f"vertical gaps {measured}")


if __name__ == "__main__":
    main()
