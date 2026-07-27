"""Figure 6 - Cold-Start Transfer, models-by-strata grid.

Three aligned forest panels (Overall, S3, S4) over a fixed row order and one
shared x-scale, so a reader compares by position in both directions: down a
panel ranks models within a stratum, across a row tracks one model as the tool
vocabulary becomes unseen. Grouped bars were rejected for the same reason as in
Figure 4: these are estimates with intervals, not quantities accumulated from
zero.

Layout rules this figure holds to, each the fix for a specific review finding:

* The header strip is a framed box whose left/right edges are the panel's own
  left/right edges - it heads exactly the frame it sits on, and it carries all
  four borders rather than reading as a floating fill. It lives above the
  plotting area, not inside it, so the panel interior holds data only and its
  top and bottom padding are equal by construction (half a row each).
* No numeral is drawn inside the plotting area. The tau_b point estimates are a
  separate column, headed and ruled off with a vertical hairline, outside the
  axes; they cannot be mistaken for a mark on the Kendall axis.
* The x-range brackets the data rather than a round number, and each row's dead
  space is filled by a dotted row rule that stops short of the whisker on both
  sides - so it can never be read as a bar from zero, and never crosses a
  glyph. Vertical gridlines are dropped: at this range every candidate rule
  lands within a marker radius of some estimate in some panel, which is exactly
  the inconsistency the review flagged.
* The axis is truncated (tau_b is defined on [-1, 1]); the break is marked on
  the spine of every panel and stated in the footnote.
* Both contrasts the figure exposes are reported, not just the flattering one:
  the claim against the baseline of record, and the schema ablation against
  prompt-only. The ablation interval reaches or spans zero in all three strata,
  so each ablation cell carries the reserved degraded-marking rule.
* The contrasts live in one full-width band under the panels, two rows deep -
  one row per contrast, one cell per stratum aligned under the panel it belongs
  to - rather than in seven text rows inside each panel's own header strip. The
  strips are back to what a strip is for: a label. The band reads across, which
  is the direction the contrast is actually compared in, and it costs a third
  of the height the three stacked strips did.
* The stratum definitions and their sizes are not repeated here. Figure 2 draws
  the stratification itself - all four strata, their n, which of them are
  under-powered and what the gate does with each - and a second copy of that
  table inside this footnote was six lines of height buying nothing. What is
  local to this figure and not derivable from Figure 2 (that panel (a) is one
  pooled estimate over every test request, withheld strata included, and so is
  not the mean of (b) and (c)) is stated in the caption.
* The footnote keeps only what labels a mark on this plate: the interval level,
  the meaning of the orange rule, and the axis truncation. Method, resample
  count and seed handling are in the caption.

A bar, a blank slot or a point at zero would each be read as a value; a named
row with an explicit withheld marker cannot be.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _common import (
    IEEE_DOUBLE_WIDTH,
    OFFLINE,
    OKABE_ITO,
    load_json,
    record_provenance,
    save,
)
from style import EXION

T1 = OFFLINE / "t1-strata.json"

# The page geometry is written out in inches and the axes are placed by hand:
# every margin, gutter and column width below is therefore a stated quantity
# that the audit at the end of main() re-measures on the rendered canvas.
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0.0})

FIG_W = IEEE_DOUBLE_WIDTH
# Double-column plate cap, from FIGURE-SPEC section 1. The plate used to run
# 4.36 in because the contrasts were set as seven text rows inside each of the
# three header strips (1.16 in) and the strata were re-tabulated in a six-line
# footnote (1.17 in): 2.33 in of the 4.36 was text that either belonged in one
# band read across, or already existed as Figure 2. Both are gone; the type
# ladder is untouched.
FIG_H = 2.76
HEIGHT_CAP = 2.80        # FIGURE-SPEC section 1, double-column plate
MARGIN = 0.07            # identical on all four sides
# Every column below was re-measured after the set moved to one family
# (DejaVu Sans) on the 10/9/8 ladder. DejaVu sets wider than Helvetica at the
# same point size, so the slots grew rather than the type shrinking:
#   row names   widest is "BERT prompt+schema", 1.410 in at 9 pt.
#   band labels widest is "delta tau_b vs tuned LightGBM", 1.452 in at 9 pt
#               starting one PAD inside the band frame - this, not the row
#               names, is what sets LABEL_W now, and it leaves the 6 px the
#               audit demands before the first panel.
#   tau_b cell  "0.43" is 0.279 in at 9 pt; the cell must also clear the
#               hairline that rules it off, so VALUE_W >= 0.279 + RULE_GAP.
# The gutter pays for part of it: the numeric column already separates two
# stratum blocks, so 0.10 in of white between them is enough.
LABEL_W = 1.62           # row-name column, and the band's label column
GUTTER = 0.10            # between stratum blocks
VALUE_W = 0.36           # tau_b numeric column
STRAT_W = (FIG_W - 2 * MARGIN - LABEL_W - 2 * GUTTER) / 3
PANEL_W = STRAT_W - VALUE_W
RULE_GAP = 0.05          # panel edge -> numeric-column hairline
PAD = 0.05               # text inset inside any framed box

# The strip is a label again: one line, one row of text.
STRIP_H = 0.22
GAP_HEAD = 0.24          # holds the numeric column's head
PLOT_H = 1.15
# The x tick labels and the axis title share one row: the title is right
# aligned in the label column, at the height of the ticks it names, instead of
# owning a centred row of its own under the panels. Same reading order, 0.17 in
# less plate.
XTICK_H = 0.22
GAP_BAND = 0.08
BAND_H = 0.48            # two contrast rows, PAD top and bottom
GAP_NOTE = 0.07
NOTE_H = 0.16            # one 8 pt line

Y_STRIP = MARGIN
Y_PLOT = Y_STRIP + STRIP_H + GAP_HEAD
Y_XTICK = Y_PLOT + PLOT_H
Y_BAND = Y_XTICK + XTICK_H + GAP_BAND
Y_NOTE = Y_BAND + BAND_H + GAP_NOTE

TEXT_X = 0.040           # strip title inset, as a fraction of the strip
# Band geometry, in inches. The cell's number is left aligned on the panel it
# belongs to and its interval is right aligned on that panel block's right
# edge, so the band's columns line up with the panels above without either
# element being centred on a string whose width changes with the data.
BAND_LABEL_X = MARGIN + PAD
# Tighter than PAD vertically, and measured rather than chosen: the band's
# tallest glyph box is the 9 pt mathtext label at 0.170 in, so a 0.21 in row
# pitch leaves the 0.04 in channel the overlap sweep demands. At the 0.05 in
# inset the pitch fell to 0.19 in and the sweep failed by 0.08 px.
BAND_PAD_Y = 0.03
CELL_VALUE_PAD = 0.06    # cell left edge -> point estimate (clears the rule)
CELL_CI_PAD = 0.08       # interval right edge -> cell right edge
MARK_W = 0.014           # width of the reserved degraded rule

# Top-to-bottom: the three scalar/identity baselines take the shared structure
# greys lightest-first, then the method family ramp light-to-dark, so grey/blue
# depth and y position agree. The two lightest structure greys are unreadable as
# a filled mark on white, so a baseline marker is drawn as a light fill inside a
# structure[3] outline and its whiskers take that same outline colour: the fill
# still carries the top-to-bottom ordering, the outline carries the visibility.
INK = {
    "schema_hash_lookup": EXION["structure"][0],
    "lightgbm_scalar": EXION["structure"][2],
    "lightgbm_grid": EXION["structure"][3],
    "bert_prompt_only": EXION["family"][1],
    "bert_prompt_schema": EXION["family"][3],
}
BASELINE_EDGE = EXION["structure"][3]
ROWS = ("schema_hash_lookup", "lightgbm_scalar", "lightgbm_grid",
        "bert_prompt_only", "bert_prompt_schema")
# Local row names, deliberately not the shared LABEL map: "grid" and "fixed"
# collide with the tuning-grid sense, and the contrast strings below have to
# name their operands without borrowing an arithmetic sign.
ROW_LABEL = {
    "schema_hash_lookup": "Schema-hash lookup",
    "lightgbm_scalar": "LightGBM (default)",
    "lightgbm_grid": "LightGBM (tuned)",
    "bert_prompt_only": "BERT prompt-only",
    "bert_prompt_schema": "BERT prompt+schema",
}
# Panel heads are short labels: at 9 pt in DejaVu the frame they sit in is
# 1.35 in of usable text, and "(b) Some tools unseen" sets 1.59 in. The pair
# below keeps the contrast the panels are ordered by - partly, then fully -
# and the footnote carries the full stratum definitions.
COLUMNS = (
    ("all", "(a) Pooled overall"),
    ("S3", "(b) Partly unseen"),
    ("S4", "(c) Fully unseen"),
)
WITHHELD = ("S1", "S2")
CLAIM, BASELINE, ABLATION = "bert_prompt_schema", "lightgbm_grid", "bert_prompt_only"
# Same constraint, same fix: each phrase is prefixed by its own delta symbol
# and has to fit the strip, so the operands are named in the shortest form
# that is still unique in this figure (one tuned LightGBM row, one
# prompt-only row).
CONTRASTS = (
    (BASELINE, "vs tuned LightGBM"),
    (ABLATION, "vs prompt-only"),
)

X_LOW, X_HIGH = 0.322, 0.705
# Ticks start at 0.40, not at the low end: the break glyph owns the first
# 4% of the spine, and a 0.35 label would sit on top of it.
X_TICKS = (0.40, 0.50, 0.60)
BREAK_X, BREAK_HALF, BREAK_LEAN = 0.021, 0.010, 0.008
HEADER_FACE = EXION["structure"][0]
FRAME = EXION["structure"][3]
RULE = EXION["structure"][2]
DEGRADED = OKABE_ITO["vermillion"]   # reserved: degraded / unresolved
Z_NORMAL = 1.959963985               # two-sided 95%
BOOTSTRAP_RESAMPLES = 1000           # t1_strata.bootstrap_ci(iterations=...)
SEED_OF_RECORD = 17
N_TRAIN_SEEDS = 3

def footnote() -> str:
    """One line: only what names a mark that is actually drawn on this plate.

    The stratum definitions and their sizes used to be re-tabulated here. They
    are Figure 2's whole subject, drawn there with the reporting rule and the
    gate's decision alongside, so the copy was four lines of duplicated height.
    Method, resample count, seed handling and the pooling qualifier for panel
    (a) moved into the caption, which is where FIGURE-SPEC section 3 puts the
    qualifier that stops a reader over-reading a figure.
    """
    return (
        "Whiskers and brackets are 95% CIs.  Orange rule: $\\Delta\\tau_b$ "
        "interval reaches or spans 0.  x-axis truncated; $\\tau_b$ spans "
        "[−1, 1]."
    )


def signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "−")


def delta_with_ci(claim: dict, base: dict) -> tuple[float, float, float]:
    """Contrast of two independent-assumed bootstrap CIs, on the tau_b scale.

    Each cell publishes a session-clustered bootstrap 95% CI; its half-width
    gives that cell's standard error. The rows are paired, so adding the two
    variances overstates the variance of the difference whenever the two
    models' errors are positively correlated - the interval is therefore the
    conservative one, and the figure says so in the footnote.
    """
    mean_c, mean_b = float(claim["mean_tau_b"]), float(base["mean_tau_b"])
    lo_c, hi_c = (float(v) for v in claim["ci95_seed17"])
    lo_b, hi_b = (float(v) for v in base["ci95_seed17"])
    se = math.hypot((hi_c - lo_c) / (2 * Z_NORMAL), (hi_b - lo_b) / (2 * Z_NORMAL))
    delta = mean_c - mean_b
    half = Z_NORMAL * se
    return delta, delta - half, delta + half


def unresolved(low: float, high: float) -> bool:
    """True when the interval covers 0, or touches it at display precision."""
    return low <= 0.0 <= high or round(low, 2) == 0.0 or round(high, 2) == 0.0


def rect(x_in: float, top_in: float, w_in: float, h_in: float) -> list[float]:
    return [x_in / FIG_W, 1.0 - (top_in + h_in) / FIG_H,
            w_in / FIG_W, h_in / FIG_H]


def row_y(index: int) -> float:
    """Figure-fraction y of a data row (rows are evenly spaced in the panel)."""
    return 1.0 - (Y_PLOT + (index + 0.5) / len(ROWS) * PLOT_H) / FIG_H


def _overlap(first, second, pad: float = 0.0) -> bool:
    return (first.x0 < second.x1 + pad and second.x0 < first.x1 + pad
            and first.y0 < second.y1 + pad and second.y0 < first.y1 + pad)


def audit(checks: list[tuple[str, bool]]) -> None:
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise AssertionError("layout audit failed: " + "; ".join(failed))
    print(f"layout audit: {len(checks)} geometric checks passed")


def draw_break(ax) -> None:
    """Mark the truncated x-axis on the spine it truncates."""
    x, half, lean = BREAK_X, BREAK_HALF, BREAK_LEAN
    # The rule begins after the break, not before it: no orphan stub is left
    # between the panel edge and the glyph.
    ax.add_patch(Rectangle((-0.006, -0.016), x + half + lean + 0.006, 0.032,
                           transform=ax.transAxes, facecolor="white",
                           edgecolor="none", zorder=5, clip_on=False))
    for offset in (-half, half):
        ax.plot([x + offset - lean, x + offset + lean], [-0.026, 0.026],
                transform=ax.transAxes, color=OKABE_ITO["black"], lw=0.7,
                zorder=6, clip_on=False, solid_capstyle="butt")


def strip_text(strip, x, y, text, size, weight="normal"):
    return strip.text(x, y, text, transform=strip.transAxes, ha="left",
                      va="center", fontsize=size, fontweight=weight,
                      color=OKABE_ITO["black"], zorder=4)


def fx(inches: float) -> float:
    return inches / FIG_W


def fy(inches_from_top: float) -> float:
    return 1.0 - inches_from_top / FIG_H


def band_row_y(slot: int) -> float:
    """Figure-fraction y of a contrast row inside the band."""
    inner = BAND_H - 2 * BAND_PAD_Y
    return fy(Y_BAND + BAND_PAD_Y + (slot + 0.5) * inner / len(CONTRASTS))


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    threshold = int(payload["stratum_definition"]["tau_reporting_threshold"])
    sizes = payload["stratum_definition"]["sizes"]
    total = int(sizes["all"])
    withheld_n = sum(int(sizes[s]) for s in WITHHELD)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    tracked: list[dict] = []
    gap = 0.0135          # whisker-to-row-rule clearance, data units

    for column, (stratum, title) in enumerate(COLUMNS):
        left = MARGIN + LABEL_W + column * (STRAT_W + GUTTER)
        ax = fig.add_axes(rect(left, Y_PLOT, PANEL_W, PLOT_H))
        strip = fig.add_axes(rect(left, Y_STRIP, PANEL_W, STRIP_H))

        marks, values = [], []
        for index, name in enumerate(ROWS):
            cell = results[name][stratum]
            mean = float(cell["mean_tau_b"])
            low, high = (float(v) for v in cell["ci95_seed17"])
            colour = INK[name]
            # Family arms are solid; baseline arms are a light structure fill
            # inside the darkest structure grey, which also draws the whiskers.
            stroke = colour if name in (CLAIM, ABLATION) else BASELINE_EDGE
            # Row rule in the two segments the data does not occupy: it fills
            # the corridor, it starts at neither zero nor the marker, and it
            # crosses no glyph.
            for a, b in ((X_LOW, low - gap), (high + gap, X_HIGH)):
                if b - a > 0.004:
                    ax.plot([a, b], [index, index], color=RULE, lw=0.5,
                            linestyle=(0, (1, 2.5)), zorder=0)
            ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]],
                        fmt="o", markersize=5.0, linewidth=1.1, capsize=2.5,
                        color=stroke, markerfacecolor=colour,
                        markeredgecolor=stroke, markeredgewidth=0.8, zorder=3)
            marks.append((low, high, index))
            values.append(fig.text(
                (left + STRAT_W) / FIG_W, row_y(index), f"{mean:.2f}",
                ha="right", va="center", fontsize=9,
                color=OKABE_ITO["black"]))

        ax.set_xlim(X_LOW, X_HIGH)
        ax.set_ylim(len(ROWS) - 0.5, -0.5)     # equal top and bottom padding
        ax.set_xticks(list(X_TICKS))
        ax.set_xticklabels([f"{t:.2f}" for t in X_TICKS], fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=2.6, pad=3.0)
        draw_break(ax)

        # --- numeric column: headed, and ruled off from the plotting area ----
        rule_x = (left + PANEL_W + RULE_GAP) / FIG_W
        head_y = fy(Y_PLOT - 0.035)
        fig.add_artist(Line2D([rule_x, rule_x],
                              [fy(Y_PLOT + PLOT_H), head_y],
                              transform=fig.transFigure, color=FRAME, lw=0.6))
        fig.add_artist(Line2D([rule_x, (left + STRAT_W) / FIG_W],
                              [head_y, head_y], transform=fig.transFigure,
                              color=FRAME, lw=0.6))
        # The offset is stated in inches, not in figure fraction: the plate
        # shrank by a third and a fractional offset would have shrunk with it.
        head = fig.text((left + STRAT_W) / FIG_W, head_y + 0.05 / FIG_H,
                        "$\\tau_b$", ha="right", va="bottom", fontsize=9,
                        color=OKABE_ITO["black"])

        # --- header strip ---------------------------------------------------
        strip.set_xticks([])
        strip.set_yticks([])
        strip.set_facecolor(HEADER_FACE)
        for spine in strip.spines.values():
            spine.set_visible(True)
            spine.set_color(FRAME)
            spine.set_linewidth(0.6)

        # The strip carries a label and nothing else. The contrasts it used to
        # hold are one band under the panels; the sizes it used to repeat are
        # Figure 2's subject.
        head_lines = [strip_text(strip, TEXT_X, 0.5, title, 9, "bold")]

        deltas = [(phrase,) + delta_with_ci(results[CLAIM][stratum],
                                            results[other][stratum])
                  for other, phrase in CONTRASTS]

        tracked.append({"ax": ax, "strip": strip, "values": values,
                        "head": head_lines, "column_head": head,
                        "marks": marks, "left": left, "deltas": deltas,
                        "stratum": stratum})

    # --- row names, left-aligned to the same edge as the footnote -----------
    labels = [fig.text(MARGIN / FIG_W, row_y(index), ROW_LABEL[name],
                       ha="left", va="center", fontsize=9,
                       color=OKABE_ITO["black"])
              for index, name in enumerate(ROWS)]

    panel_left = (MARGIN + LABEL_W) / FIG_W

    # The axis title sits in the label column at the height of the tick labels
    # it names, so it costs no row of its own. It is the bottom entry of the
    # same column the row names occupy, which is where a reader scanning the
    # left edge downwards arrives at the axis anyway.
    xlabel = fig.text(fx(MARGIN + LABEL_W - 0.10), fy(Y_XTICK + XTICK_H / 2),
                      "Kendall $\\tau_b$", ha="right", va="center",
                      fontsize=10, color=OKABE_ITO["black"])

    # --- contrast band: both contrasts, one row each, read across strata -----
    band = Rectangle((fx(MARGIN), fy(Y_BAND + BAND_H)),
                     fx(FIG_W - 2 * MARGIN), BAND_H / FIG_H,
                     transform=fig.transFigure, facecolor=HEADER_FACE,
                     edgecolor=FRAME, linewidth=0.6, zorder=2)
    fig.add_artist(band)

    band_texts: list = []
    band_cells: list[tuple[int, object, object]] = []
    for slot, (_, phrase) in enumerate(CONTRASTS):
        y = band_row_y(slot)
        band_texts.append(fig.text(
            fx(BAND_LABEL_X), y, f"$\\Delta\\tau_b$ {phrase}", ha="left",
            va="center", fontsize=9, color=OKABE_ITO["black"], zorder=3))
        for column, entry in enumerate(tracked):
            _, delta, low, high = entry["deltas"][slot]
            cell_x = entry["left"]
            if unresolved(low, high):
                # One condition, one colour, marked once per cell: the reserved
                # vermillion rule hugs the cell's left edge, clear of the type.
                mark_h = 0.115 / FIG_H
                fig.add_artist(Rectangle(
                    (fx(cell_x), y - mark_h / 2), fx(MARK_W), mark_h,
                    transform=fig.transFigure, facecolor=DEGRADED,
                    edgecolor="none", zorder=3))
            value = fig.text(
                fx(cell_x + CELL_VALUE_PAD), y, signed(delta), ha="left",
                va="center", fontsize=10, fontweight="bold",
                color=OKABE_ITO["black"], zorder=3)
            interval = fig.text(
                fx(cell_x + STRAT_W - CELL_CI_PAD), y,
                f"[{signed(low)}, {signed(high)}]", ha="right", va="center",
                fontsize=8, color=OKABE_ITO["black"], zorder=3)
            band_texts += [value, interval]
            band_cells.append((column, value, interval))

    note = fig.text(MARGIN / FIG_W, MARGIN / FIG_H, footnote(),
                    ha="left", va="bottom", fontsize=8, linespacing=1.42,
                    color=EXION["structure"][3])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Leader from each row name to the panel it names, drawn only in the space
    # the name leaves free, so the label column reads as a table row.
    for label in labels:
        box = label.get_window_extent(renderer)
        x0 = fig.transFigure.inverted().transform((box.x1, 0))[0] + 0.008
        if panel_left - x0 > 0.01:
            fig.add_artist(Line2D([x0, panel_left - 0.004],
                                  [label.get_position()[1]] * 2,
                                  transform=fig.transFigure, color=RULE,
                                  lw=0.5, linestyle=(0, (1, 2.5))))

    fig.canvas.draw()
    checks: list[tuple[str, bool]] = []
    dpi = fig.dpi
    px = dpi / 72.0
    clear = 2.0
    mark_pad = 3.6 * px          # marker radius plus cap half-height

    for entry in tracked:
        ax, strip, stratum = entry["ax"], entry["strip"], entry["stratum"]
        panel = ax.get_window_extent(renderer)
        strip_box = strip.get_window_extent(renderer)

        # 1. the strip heads exactly the frame it sits on
        checks.append((f"{stratum}: strip left edge on the panel frame",
                       abs(strip_box.x0 - panel.x0) <= 0.5))
        checks.append((f"{stratum}: strip right edge on the panel frame",
                       abs(strip_box.x1 - panel.x1) <= 0.5))

        # 2. interior padding equal top and bottom
        first = ax.transData.transform((X_LOW, 0))[1]
        last = ax.transData.transform((X_LOW, len(ROWS) - 1))[1]
        checks.append((f"{stratum}: equal top/bottom interior padding",
                       abs((panel.y1 - first) - (last - panel.y0)) <= 0.5))

        # 3. the data spans the panel
        spans = [(ax.transData.transform((high, 0))[0]
                  - ax.transData.transform((low, 0))[0])
                 for low, high, _ in entry["marks"]]
        left_px = min(ax.transData.transform((low, 0))[0]
                      for low, _, _ in entry["marks"])
        right_px = max(ax.transData.transform((high, 0))[0]
                       for _, high, _ in entry["marks"])
        checks.append((f"{stratum}: data occupies the panel width",
                       (right_px - left_px) >= 0.60 * panel.width))
        checks.append((f"{stratum}: every interval is visible",
                       min(spans) >= 4.0))

        marks = []
        for low, high, index in entry["marks"]:
            x0 = ax.transData.transform((low, index))[0]
            x1 = ax.transData.transform((high, index))[0]
            y_px = ax.transData.transform((low, index))[1]
            marks.append((x0 - mark_pad, x1 + mark_pad,
                          y_px - mark_pad, y_px + mark_pad))
            checks.append((f"{stratum}: row {index} inside the x-range",
                           x0 >= panel.x0 + 1.0 and x1 <= panel.x1 - 1.0))

        # 4. no numeral inside the plotting area; the column is ruled off
        rule_x = (entry["left"] + PANEL_W + RULE_GAP) * dpi
        for text in entry["values"] + [entry["column_head"]]:
            box = text.get_window_extent(renderer)
            checks.append((f"{stratum}: value {text.get_text()} outside the axes",
                           box.x0 >= panel.x1 + clear))
            checks.append((f"{stratum}: value {text.get_text()} right of the rule",
                           box.x0 >= rule_x + clear))

        # 5. header strip: everything inside its frame, nothing touching
        for text in entry["head"]:
            box = text.get_window_extent(renderer)
            name = f"{stratum}: strip {text.get_text()[:16]!r}"
            checks.append((f"{name} inside the frame",
                           box.x0 >= strip_box.x0 + 3.0
                           and box.x1 <= strip_box.x1 - 3.0
                           and box.y0 >= strip_box.y0 + 2.0
                           and box.y1 <= strip_box.y1 - 2.0))
        for i in range(len(entry["head"])):
            for j in range(i + 1, len(entry["head"])):
                checks.append((
                    f"{stratum}: strip text {i} vs {j}",
                    not _overlap(entry["head"][i].get_window_extent(renderer),
                                 entry["head"][j].get_window_extent(renderer),
                                 clear)))

        # 6. x tick marks point inward but stop well short of the last row
        tick_top = panel.y0 + 2.6 * px
        bottom_row = ax.transData.transform((X_LOW, len(ROWS) - 1))[1]
        checks.append((f"{stratum}: x ticks clear the bottom row",
                       bottom_row - mark_pad - tick_top >= 2.0 * px))

        # 7. the break glyph owns the left end of the spine on its own
        break_right = panel.x0 + (BREAK_X + BREAK_HALF + BREAK_LEAN) * panel.width
        tick_boxes = [t.get_window_extent(renderer)
                      for t in ax.get_xticklabels()]
        checks.append((f"{stratum}: break glyph clears the first tick label",
                       min(b.x0 for b in tick_boxes) >= break_right + 3.0 * px))
        checks.append((f"{stratum}: last tick label inside the panel",
                       max(b.x1 for b in tick_boxes) <= panel.x1 + 1.0))

    # 8. row names sit in their own column and clear the panels
    first_panel = tracked[0]["ax"].get_window_extent(renderer)
    for label in labels:
        box = label.get_window_extent(renderer)
        checks.append((f"row name {label.get_text()!r} inside the margin",
                       box.x0 >= MARGIN * dpi - 0.5))
        checks.append((f"row name {label.get_text()!r} clears the panel",
                       box.x1 <= first_panel.x0 - 6.0))

    # 9. the axis title lives in the label column, at the tick row, and clears
    #    both the panel it names and the last row name above it
    label_box = xlabel.get_window_extent(renderer)
    checks.append(("axis title inside the label column",
                   label_box.x0 >= MARGIN * dpi - 0.5
                   and label_box.x1 <= first_panel.x0 - 6.0))
    checks.append(("axis title level with the tick labels",
                   label_box.y1 <= first_panel.y0 + 1.0))
    checks.append(("axis title clears the last row name",
                   label_box.y1 <= min(l.get_window_extent(renderer).y0
                                       for l in labels) - 2.0))

    # 10. the contrast band: inside its frame, aligned on the panels above,
    #     nothing touching anything, and clear of the plate above and below
    band_box = band.get_window_extent(renderer)
    checks.append(("band spans the text width",
                   abs(band_box.x0 - MARGIN * dpi) <= 1.0
                   and abs(band_box.x1 - (FIG_W - MARGIN) * dpi) <= 1.0))
    checks.append(("band clears the tick labels above it",
                   band_box.y1 <= min(t.get_window_extent(renderer).y0
                                      for e in tracked
                                      for t in e["ax"].get_xticklabels())
                   - 2.0))
    for text in band_texts:
        box = text.get_window_extent(renderer)
        name = f"band {text.get_text()[:18]!r}"
        checks.append((f"{name} inside the band frame",
                       box.x0 >= band_box.x0 + 2.0
                       and box.x1 <= band_box.x1 - 2.0
                       and box.y0 >= band_box.y0 + 1.0
                       and box.y1 <= band_box.y1 - 1.0))
    for i in range(len(band_texts)):
        for j in range(i + 1, len(band_texts)):
            checks.append((
                f"band text {i} vs {j}",
                not _overlap(band_texts[i].get_window_extent(renderer),
                             band_texts[j].get_window_extent(renderer),
                             clear)))
    for column, value, interval in band_cells:
        panel = tracked[column]["ax"].get_window_extent(renderer)
        value_box = value.get_window_extent(renderer)
        interval_box = interval.get_window_extent(renderer)
        # The cell reads as a column of the panel above it: its number starts
        # inside that panel's own left edge and its interval ends before the
        # next stratum block begins.
        checks.append((f"band cell {column} starts on its panel",
                       value_box.x0 >= panel.x0 - 0.5
                       and value_box.x0 <= panel.x0 + CELL_VALUE_PAD * dpi
                       + 2.0))
        checks.append((f"band cell {column} interval inside the block",
                       interval_box.x1 <= (tracked[column]["left"] + STRAT_W)
                       * dpi + 0.5))

    note_box = note.get_window_extent(renderer)
    checks.append(("footnote shares the row-name left edge",
                   abs(note_box.x0 - min(l.get_window_extent(renderer).x0
                                         for l in labels)) <= 1.0))
    checks.append(("footnote inside the right margin",
                   note_box.x1 <= FIG_W * dpi - MARGIN * dpi + 0.5))
    checks.append(("footnote is one line",
                   note.get_text().count("\n") == 0))
    checks.append(("footnote clears the band",
                   note_box.y1 <= band_box.y0 - 2.0))

    # 11. the plate is inside the double-column height cap
    checks.append((f"plate height {FIG_H:.2f} in within the "
                   f"{HEIGHT_CAP:.2f} in cap", FIG_H <= HEIGHT_CAP + 1e-9))

    # 12. margins: the same on all four sides, within half a pixel
    ink = [note_box, label_box, band_box]
    ink += [t.get_window_extent(renderer) for t in labels]
    ink += [e["strip"].get_window_extent(renderer) for e in tracked]
    ink += [t.get_window_extent(renderer)
            for e in tracked for t in e["values"] + [e["column_head"]]]
    margin_px = MARGIN * dpi
    checks.append(("left margin honoured",
                   abs(min(b.x0 for b in ink) - margin_px) <= 1.0))
    checks.append(("right margin honoured",
                   abs((FIG_W * dpi - max(b.x1 for b in ink)) - margin_px)
                   <= 2.0))
    checks.append(("top margin honoured",
                   abs((FIG_H * dpi - max(b.y1 for b in ink)) - margin_px)
                   <= 1.0))
    checks.append(("bottom margin honoured",
                   abs(min(b.y0 for b in ink) - margin_px) <= 3.0))
    audit(checks)

    print(f"bootstrap: session-clustered, {BOOTSTRAP_RESAMPLES} resamples, "
          f"percentile, seed-{SEED_OF_RECORD} scores; markers are the mean "
          f"over {N_TRAIN_SEEDS} training seeds")
    print(f"withheld (n<{threshold}): "
          + ", ".join(f"{s} n={sizes[s]}" for s in WITHHELD)
          + f"; pooled Overall n={total}")
    for entry in tracked:
        for phrase, delta, low, high in entry["deltas"]:
            flag = " UNRESOLVED" if unresolved(low, high) else ""
            print(f"{entry['stratum']}: delta {phrase} = {delta:+.4f} "
                  f"95% CI [{low:+.4f}, {high:+.4f}]{flag}")

    save(fig, "coldstart.pdf")
    record_provenance("coldstart.pdf", [T1])
    print("cold-start grid written")


if __name__ == "__main__":
    main()
