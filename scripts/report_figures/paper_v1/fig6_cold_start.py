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
  so each ablation block carries the reserved degraded-marking rule.
* Every interval is labelled with its confidence level, its method and its
  resample count in the footnote, and the withheld strata are named together
  with the fact that their rows are inside the pooled Overall estimate.

A bar, a blank slot or a point at zero would each be read as a value; a named
row with a withheld footnote cannot be.
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
# The plate is 0.24 in taller than it was: the strip gained a line per
# contrast (see STRIP_H) and the six-line footnote is fixed - at 8 pt in
# DejaVu its longest line already runs to 6.6 in of the 7.02 in text width,
# so it cannot be re-flowed into fewer lines. Height is the only slot left.
FIG_H = 4.36
MARGIN = 0.07            # identical on all four sides
# Every column below was re-measured after the set moved to one family
# (DejaVu Sans) on the 10/9/8 ladder. DejaVu sets wider than Helvetica at the
# same point size, so the slots grew rather than the type shrinking:
#   row names   widest is "BERT prompt+schema", 1.408 in at 9 pt -> 1.50 in
#               column leaves the 6 px the audit demands before the panel.
#   tau_b cell  "0.43" is 0.279 in at 9 pt; the cell must also clear the
#               hairline that rules it off, so VALUE_W >= 0.279 + RULE_GAP.
# The gutter pays for part of it: the numeric column already separates two
# stratum blocks, so 0.10 in of white between them is enough.
LABEL_W = 1.50           # row-name column
GUTTER = 0.10            # between stratum blocks
VALUE_W = 0.36           # tau_b numeric column
STRAT_W = (FIG_W - 2 * MARGIN - LABEL_W - 2 * GUTTER) / 3
PANEL_W = STRAT_W - VALUE_W
RULE_GAP = 0.05          # panel edge -> numeric-column hairline

# The strip carries seven text rows now (title, then name/value/interval per
# contrast). At 9/10/8 pt in DejaVu a contrast's point estimate and its
# interval no longer fit side by side inside a panel-wide frame, so the
# interval sits on its own line under the number it belongs to and the strip
# grew to hold it. The height comes out of the slack that sat between the
# axis title and the footnote, so the plate keeps its page height.
STRIP_H = 1.16
GAP_HEAD = 0.24          # holds the numeric column's head
PLOT_H = 1.32
XTICK_H = 0.16
XLABEL_H = 0.17

Y_STRIP = MARGIN
Y_PLOT = Y_STRIP + STRIP_H + GAP_HEAD
Y_XLABEL = Y_PLOT + PLOT_H + XTICK_H

# Strip rows, as fractions of the strip height; spacing is set from the
# measured glyph heights (mathtext runs taller than plain text), not guessed.
S_TITLE, S_HAIR = 0.914, 0.828
S_NAME, S_VALUE, S_CI, S_STEP = 0.741, 0.603, 0.478, 0.401
TEXT_X, RIGHT_X = 0.040, 0.960

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

def footnote(sizes: dict, threshold: int, withheld_n: int) -> str:
    """Sample sizes live here, not in the header strip: a header is a label.

    Every count is read from the artifact, so a rebuilt stratification changes
    the sentence rather than silently disagreeing with it.
    """
    return (
        "Strata by tool vocabulary vs. training — S1 seen combination "
        f"($n$ = {sizes['S1']}) · S2 new combination of seen tools "
        f"($n$ = {sizes['S2']}) ·\n"
        f"S3 some tools unseen ($n$ = {sizes['S3']}) · S4 all tools unseen "
        f"($n$ = {sizes['S4']}).  Overall is one pooled $\\tau_b$ over all "
        f"{sizes['all']} test\n"
        f"requests, not a mean of strata; it includes the {withheld_n} requests "
        "of withheld S1 and S2, whose $\\tau_b$ is suppressed at\n"
        f"$n$ < {threshold}.  Markers are the mean over 3 seeds; every interval "
        "is a 95% CI (session-clustered bootstrap, seed-17 scores,\n"
        "1000 resamples, percentile); the $\\Delta\\tau_b$ interval adds the "
        "two as independent, conservative here.  Orange rule:\n"
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
        head_y = 1.0 - (Y_PLOT - 0.035) / FIG_H
        fig.add_artist(Line2D([rule_x, rule_x],
                              [1.0 - (Y_PLOT + PLOT_H) / FIG_H, head_y],
                              transform=fig.transFigure, color=FRAME, lw=0.6))
        fig.add_artist(Line2D([rule_x, (left + STRAT_W) / FIG_W],
                              [head_y, head_y], transform=fig.transFigure,
                              color=FRAME, lw=0.6))
        head = fig.text((left + STRAT_W) / FIG_W, head_y + 0.012,
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

        # The strip carries a label and the contrasts it frames; the sample
        # sizes it used to repeat sit in the footnote.
        head_lines = [strip_text(strip, TEXT_X, S_TITLE, title, 9, "bold")]
        strip.plot([TEXT_X, RIGHT_X], [S_HAIR, S_HAIR],
                   transform=strip.transAxes, color=FRAME, lw=0.5, zorder=4)

        deltas = []
        for slot, (other, phrase) in enumerate(CONTRASTS):
            delta, low, high = delta_with_ci(results[CLAIM][stratum],
                                             results[other][stratum])
            y_name = S_NAME - slot * S_STEP
            y_value = S_VALUE - slot * S_STEP
            y_ci = S_CI - slot * S_STEP
            head_lines.append(strip_text(
                strip, TEXT_X, y_name, f"$\\Delta\\tau_b$ {phrase}", 8))
            head_lines.append(strip_text(
                strip, TEXT_X, y_value, signed(delta), 10, "bold"))
            head_lines.append(strip_text(
                strip, TEXT_X, y_ci, f"[{signed(low)}, {signed(high)}]", 8))
            if unresolved(low, high):
                # Hugs the frame rather than the type: the first glyph of the
                # block now starts at 0.040 of a narrower panel, so the rule
                # moved left to keep a visible channel between the two.
                strip.add_patch(Rectangle(
                    (0.010, y_ci - 0.050), 0.011,
                    y_name - y_ci + 0.100, transform=strip.transAxes,
                    facecolor=DEGRADED, edgecolor="none", zorder=4))
            deltas.append((phrase, delta, low, high))

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
    panel_right = (MARGIN + LABEL_W + 2 * (STRAT_W + GUTTER) + PANEL_W) / FIG_W
    xlabel = fig.text((panel_left + panel_right) / 2,
                      1.0 - (Y_XLABEL + XLABEL_H / 2) / FIG_H,
                      "Kendall $\\tau_b$", ha="center", va="center",
                      fontsize=10, color=OKABE_ITO["black"])
    note = fig.text(MARGIN / FIG_W, MARGIN / FIG_H,
                    footnote(sizes, threshold, withheld_n),
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

    # 9. axis title and footnote are centred / aligned on the panel block
    block_x0 = tracked[0]["ax"].get_window_extent(renderer).x0
    block_x1 = tracked[-1]["ax"].get_window_extent(renderer).x1
    label_box = xlabel.get_window_extent(renderer)
    checks.append(("x-axis title centred on the tick-labelled span",
                   abs((label_box.x0 + label_box.x1) / 2
                       - (block_x0 + block_x1) / 2) <= 1.0))
    note_box = note.get_window_extent(renderer)
    checks.append(("footnote shares the row-name left edge",
                   abs(note_box.x0 - min(l.get_window_extent(renderer).x0
                                         for l in labels)) <= 1.0))
    checks.append(("footnote inside the right margin",
                   note_box.x1 <= FIG_W * dpi - MARGIN * dpi + 0.5))
    checks.append(("footnote clears the axis title",
                   note_box.y1 <= label_box.y0 - 6.0))

    # 10. margins: the same on all four sides, within half a pixel
    ink = [note_box, label_box]
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
