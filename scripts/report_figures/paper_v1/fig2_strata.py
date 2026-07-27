"""Figure 2 - the cold-start stratification itself.

The figure this replaces redrew the pipeline of Figure 1 (same decision-service
stack, same fail-open arrow, its own duplicate legend) and so carried no
information the reader did not already have. What Background actually leaves in
prose is the *construct*: how a request is placed into one of four cold-start
strata, how big each stratum is, how well the shipped Ranker actually orders
inside it, and which of them the Reliability Gate is willing to vouch for.
That is one table-shaped claim, so it is drawn as one: strata as rows on a
single Kendall axis, each row carrying its size, its measured interval and the
gate's decision.

Layout rules, each of them a fix for a specific review finding on this set:

* Rows are ordered S1 -> S4, i.e. by how much of the request's tool vocabulary
  is new, so vertical position carries the construct's own ordering.
* The two under-powered strata are drawn as rows with no interval, not as rows
  with a short interval or a point at zero. A blank slot reads as "missing";
  a named row with an explicit withheld marker reads as "not claimed". Their
  size, their in-row marker and the gate's word all take the reserved
  vermillion, so one condition is marked once, in one colour, three times over.
* No stratum's tau is invented for the figure. T1 withholds tau below the
  reporting threshold, and the figure withholds it too - the artifact and the
  plate cannot disagree.
* The gate's assigned confidence shares the tau axis because it *is* a tau: it
  is the lower bound of the validation CI. Drawing it on the same scale is what
  makes "the gate promises less than it delivers" visible rather than asserted.
* Sizes live in the rows and the header strips carry labels only. The reporting
  rule that produces the withheld rows is *shown*, not restated: a vermillion
  size, a vermillion withheld marker and a vermillion "abstains" already say it
  three times, so the sentence version of it belongs in the caption instead.
* Method detail that qualifies the marks rather than reading them - what the
  partition is, what the gate fit on - is caption text. Only the two facts a
  reader needs to avoid misreading the marks themselves stay in the artwork.
* Full width rather than single column: the four definitions are the payload,
  and at 9 pt the longest of them sets 1.9 in on its own, which leaves a single
  column no room for both a tau axis and a decision column.

Every number is read from the two committed artifacts at build time.
"""

from __future__ import annotations

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
T5 = OFFLINE / "t5-gate.json"

# The model whose realized quality the figure reports: the one that ships.
CLAIM = "bert_prompt_schema"
# The small-stratum rule the gate runs in production, selected in T5.
RULE = "C_abstain"
STRATA = ("S1", "S2", "S3", "S4")

# Row wording. These are labels, not data: the partition itself is defined by
# the tool-set fingerprint, and the artifact's own glossary line is quoted in
# the footnote. Kept contrastive pair-wise so S1/S2 and S3/S4 cannot blur.
DEFINITION = {
    "S1": "Tool set seen in training",
    "S2": "Tools seen, combination new",
    "S3": "Some tools never seen",
    "S4": "No tool seen in training",
}

INK = OKABE_ITO["black"]
FRAME = EXION["structure"][3]
HEADER_FACE = EXION["structure"][0]
ROW_RULE = EXION["structure"][2]
FOOT = EXION["structure"][3]
# The shipped Ranker owns the whole ordering decision in a vouched stratum, so
# it takes the darkest end of the family ramp; the confidence the gate is
# willing to promise is the same quantity held back, so it takes the lightest.
CLAIM_INK = EXION["family"][3]
GATE_INK = EXION["family"][0]
DEGRADED = OKABE_ITO["vermillion"]      # reserved: withheld / abstained

# ---- page geometry, in inches ---------------------------------------------
plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0.0})

FIG_W = IEEE_DOUBLE_WIDTH
FIG_H = 2.62
MARGIN = 0.07

ID_W = 0.30                              # "S1"
DEF_W = 2.00                             # widest definition, 9 pt DejaVu
N_W = 0.60                               # "n = 543", right aligned
GUTTER = 0.12
PANEL_W = 2.03
RULE_GAP = 0.05                          # panel edge -> numeric hairline
VALUE_W = 0.39                           # "0.65"
PAD = 0.05                               # text inset inside any framed box

# Everything that lives inside a framed column starts one PAD in from that
# frame; only the page-level footnote aligns with the frames themselves. The
# axes are the exception by design - a panel is its own frame, so the strip
# above it takes its exact left edge (audited below).
ID_X = MARGIN + PAD
DEF_X = ID_X + ID_W
N_R = DEF_X + DEF_W + N_W                # right edge of the size column
PANEL_X = N_R + GUTTER
VALUE_R = PANEL_X + PANEL_W + RULE_GAP + VALUE_W
GATE_X = VALUE_R + GUTTER
GATE_W = FIG_W - MARGIN - GATE_X

STRIP_H = 0.22
GAP_STRIP = 0.07
PLOT_H = 1.24
XTICK_H = 0.17
XLABEL_H = 0.19
LEGEND_H = 0.28

Y_STRIP = MARGIN
Y_PLOT = Y_STRIP + STRIP_H + GAP_STRIP
Y_XLABEL = Y_PLOT + PLOT_H + XTICK_H
Y_LEGEND = Y_XLABEL + XLABEL_H + 0.08
Y_FOOT = Y_LEGEND + LEGEND_H + 0.07


def fx(inches: float) -> float:
    return inches / FIG_W


def fy(inches_from_top: float) -> float:
    return 1.0 - inches_from_top / FIG_H


def row_y(index: int) -> float:
    return fy(Y_PLOT + (index + 0.5) / len(STRATA) * PLOT_H)


def framed(fig, x_in: float, top_in: float, w_in: float, h_in: float,
           face: str) -> Rectangle:
    patch = Rectangle((fx(x_in), fy(top_in + h_in)), fx(w_in), h_in / FIG_H,
                      transform=fig.transFigure, facecolor=face,
                      edgecolor=FRAME, linewidth=0.6, zorder=2)
    fig.add_artist(patch)
    return patch


def audit(checks: list[tuple[str, bool]]) -> None:
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise AssertionError("audit failed: " + "; ".join(failed))
    print(f"audit: {len(checks)} checks passed")


def overlaps(first, second, pad: float = 0.0) -> bool:
    return (first.x0 < second.x1 - pad and second.x0 < first.x1 - pad
            and first.y0 < second.y1 - pad and second.y0 < first.y1 - pad)


def draw_break(ax) -> None:
    """Mark the truncation on the spine it truncates."""
    x, half, lean = 0.022, 0.010, 0.008
    ax.add_patch(Rectangle((-0.006, -0.018), x + half + lean + 0.006, 0.036,
                           transform=ax.transAxes, facecolor="white",
                           edgecolor="none", zorder=5, clip_on=False))
    for offset in (-half, half):
        ax.plot([x + offset - lean, x + offset + lean], [-0.028, 0.028],
                transform=ax.transAxes, color=INK, lw=0.7, zorder=6,
                clip_on=False, solid_capstyle="butt")


def main() -> None:
    t1 = load_json(T1)
    t5 = load_json(T5)

    definition = t1["stratum_definition"]
    sizes = {s: int(definition["sizes"][s]) for s in STRATA}
    total = int(definition["sizes"]["all"])
    threshold = int(definition["tau_reporting_threshold"])
    toolless = {s: int(definition["toolless_rows_by_stratum"][s]) for s in STRATA}
    vocab_names = int(definition["train_unique_tool_names"])

    cells = {s: t1["results"][CLAIM][s] for s in STRATA}
    withheld = {s for s in STRATA if "tau_withheld" in cells[s]}
    n_seeds = len(cells["S3"]["per_seed_tau_b"])

    gate = {row["stratum"]: float(row["assigned"])
            for row in t5["rule_comparison"][RULE]["per_stratum"]}
    abstained = {s for s in STRATA if gate[s] <= 0.0}
    validation_n = {s: int(t5["validation_fit"][s]["n"]) for s in STRATA}

    # The three ways of saying "under-powered" have to be the same set, or the
    # figure would be asserting a rule the artifacts do not follow.
    below = {s for s in STRATA if sizes[s] < threshold}
    consistency = [
        ("withheld strata are exactly the sub-threshold test strata",
         withheld == below),
        ("gate abstains on exactly the withheld strata", abstained == withheld),
        ("gate's own split agrees on which strata are under-powered",
         {s for s in STRATA if validation_n[s] < threshold} == withheld),
        ("strata partition the test split", sum(sizes.values()) == total),
    ]

    measured = [s for s in STRATA if s not in withheld]
    lows = [float(cells[s]["ci95_seed17"][0]) for s in measured]
    highs = [float(cells[s]["ci95_seed17"][1]) for s in measured]
    span_lo = min(lows + [gate[s] for s in measured])
    span_hi = max(highs)
    x_low = (int((span_lo - 0.025) * 100)) / 100.0
    x_high = (int((span_hi + 0.025) * 100) + 1) / 100.0
    ticks = [t / 100.0 for t in range(0, 201, 5)
             if x_low + 0.02 * (x_high - x_low) < t / 100.0 < x_high]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([fx(PANEL_X), fy(Y_PLOT + PLOT_H),
                       fx(PANEL_W), PLOT_H / FIG_H])
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(len(STRATA) - 0.5, -0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=2.6, pad=3.0)
    draw_break(ax)

    gap = 0.006 * (x_high - x_low) / 0.17    # whisker-to-rule clearance
    texts: list = []

    for index, stratum in enumerate(STRATA):
        y = row_y(index)
        marked = stratum in withheld
        tone = DEGRADED if marked else INK

        texts.append(fig.text(fx(ID_X), y, stratum, ha="left", va="center",
                              fontsize=9, fontweight="bold", color=INK))
        texts.append(fig.text(fx(DEF_X), y, DEFINITION[stratum], ha="left",
                              va="center", fontsize=9, color=INK))
        texts.append(fig.text(fx(N_R), y, f"$n$ = {sizes[stratum]}",
                              ha="right", va="center", fontsize=9, color=tone))

        if marked:
            # No interval, because there is no published estimate: the row is
            # named and its absence is stated, which a blank slot cannot do.
            ax.plot([x_low, x_high], [index, index], color=ROW_RULE, lw=0.5,
                    linestyle=(0, (1, 2.5)), zorder=0)
            texts.append(ax.text(
                (x_low + x_high) / 2, index, "$\\tau_b$ withheld",
                ha="center", va="center", fontsize=9, color=DEGRADED,
                bbox=dict(boxstyle="square,pad=0.16", facecolor="white",
                          edgecolor="none"), zorder=3))
            texts.append(fig.text(fx(GATE_X + PAD), y, "abstains", ha="left",
                                  va="center", fontsize=9, color=DEGRADED))
            continue

        mean = float(cells[stratum]["mean_tau_b"])
        low, high = (float(v) for v in cells[stratum]["ci95_seed17"])
        for a, b in ((x_low, min(low, gate[stratum]) - gap),
                     (high + gap, x_high)):
            if b - a > 0.004:
                ax.plot([a, b], [index, index], color=ROW_RULE, lw=0.5,
                        linestyle=(0, (1, 2.5)), zorder=0)
        ax.plot([gate[stratum]] * 2, [index - 0.17, index + 0.17],
                color=GATE_INK, lw=1.8, solid_capstyle="butt", zorder=2)
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o",
                    markersize=5.0, linewidth=1.1, capsize=2.5,
                    color=CLAIM_INK, markerfacecolor=CLAIM_INK,
                    markeredgecolor=CLAIM_INK, zorder=3)
        texts.append(fig.text(fx(VALUE_R), y, f"{mean:.2f}", ha="right",
                              va="center", fontsize=9, color=INK))
        texts.append(fig.text(fx(GATE_X + PAD), y, "vouches", ha="left",
                              va="center", fontsize=9, color=CLAIM_INK))

    # --- numeric column, ruled off from the plotting area -------------------
    # The rule spans the plotting area and nothing else: it exists so a value
    # in the right-hand column can never be read as a mark on the tau axis.
    # It carries no head of its own - the strip above already names the block,
    # and a second label would put a symbol inside a header strip.
    rule_x = fx(PANEL_X + PANEL_W + RULE_GAP)
    fig.add_artist(Line2D([rule_x, rule_x],
                          [fy(Y_PLOT + PLOT_H), fy(Y_PLOT)],
                          transform=fig.transFigure, color=FRAME, lw=0.6))

    # --- header strips: labels only, never results --------------------------
    heads = [
        (MARGIN, N_R - MARGIN, "Cold-start stratum"),
        (PANEL_X, VALUE_R - PANEL_X, "Realized ranking quality"),
        (GATE_X, GATE_W, "Reliability gate"),
    ]
    strips = []
    for x_in, w_in, label in heads:
        strips.append(framed(fig, x_in, Y_STRIP, w_in, STRIP_H, HEADER_FACE))
        texts.append(fig.text(fx(x_in + PAD), fy(Y_STRIP + STRIP_H / 2), label,
                              ha="left", va="center", fontsize=9,
                              fontweight="bold", color=INK, zorder=3))

    texts.append(fig.text(fx(PANEL_X + PANEL_W / 2),
                          fy(Y_XLABEL + XLABEL_H / 2), "Kendall $\\tau_b$",
                          ha="center", va="center", fontsize=10, color=INK))

    # --- one boxed legend band, outside the axes ----------------------------
    legend = framed(fig, MARGIN, Y_LEGEND, FIG_W - 2 * MARGIN, LEGEND_H,
                    "white")
    y_leg = fy(Y_LEGEND + LEGEND_H / 2)
    entries = [
        ("point", "Realized $\\tau_b$, 95% CI"),
        ("tick", "Confidence the gate assigns"),
        ("mark", "Under-powered: not claimed"),
    ]
    slot = (FIG_W - 2 * MARGIN - 2 * PAD) / len(entries)
    for position, (kind, label) in enumerate(entries):
        x0 = MARGIN + PAD + position * slot
        if kind == "point":
            fig.add_artist(Line2D([fx(x0), fx(x0 + 0.20)], [y_leg] * 2,
                                  transform=fig.transFigure, color=CLAIM_INK,
                                  lw=1.1, marker="", zorder=3))
            fig.add_artist(Line2D([fx(x0 + 0.10)], [y_leg],
                                  transform=fig.transFigure, color=CLAIM_INK,
                                  marker="o", markersize=5.0, zorder=3))
        elif kind == "tick":
            fig.add_artist(Line2D([fx(x0 + 0.10)] * 2,
                                  [y_leg - 0.026, y_leg + 0.026],
                                  transform=fig.transFigure, color=GATE_INK,
                                  lw=1.8, zorder=3))
        else:
            fig.add_artist(Rectangle((fx(x0 + 0.05), y_leg - 0.024),
                                     fx(0.10), 0.048,
                                     transform=fig.transFigure,
                                     facecolor=DEGRADED, edgecolor="none",
                                     zorder=3))
        texts.append(fig.text(fx(x0 + 0.27), y_leg, label, ha="left",
                              va="center", fontsize=9, color=INK, zorder=3))

    # --- footnote: only what is needed to read the marks correctly ----------
    # Everything else the old four-line note carried - the partition rule, the
    # reporting threshold, the gate's validation sizes - qualifies the figure
    # rather than its marks, so it is emitted for the caption instead (below).
    # No math here: a subscript at 8 pt would render at 5.6 pt.
    note = (
        f"Markers are the mean over {n_seeds} seeds; bars are 95% bootstrap "
        "CIs (seed-17 scores).  Axis truncated; full range [−1, 1]."
    )
    texts.append(fig.text(fx(MARGIN), fy(Y_FOOT), note, ha="left", va="top",
                          fontsize=8, linespacing=1.42, color=FOOT))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checks = list(consistency)

    canvas = fig.get_window_extent(renderer)
    for text in texts:
        box = text.get_window_extent(renderer)
        checks.append((f"'{text.get_text()[:24]}' inside the canvas",
                       box.x0 >= canvas.x0 - 0.5 and box.x1 <= canvas.x1 + 0.5
                       and box.y0 >= canvas.y0 - 0.5
                       and box.y1 <= canvas.y1 + 0.5))

    figure_texts = [t for t in texts if t.get_figure() is fig
                    and t.axes is None]
    for i, first in enumerate(figure_texts):
        for second in figure_texts[i + 1:]:
            checks.append((
                f"'{first.get_text()[:16]}' clears '{second.get_text()[:16]}'",
                not overlaps(first.get_window_extent(renderer),
                             second.get_window_extent(renderer), pad=0.5)))

    panel = ax.get_window_extent(renderer)
    for strip, (x_in, w_in, label) in zip(strips, heads):
        box = strip.get_window_extent(renderer)
        checks.append((f"strip '{label}' clears the plotting area",
                       box.y0 > panel.y1 - 0.5 or box.y1 < panel.y0 + 0.5
                       or not overlaps(box, panel)))
    checks.append(("header strip heads exactly the panel it labels",
                   abs(strips[1].get_window_extent(renderer).x0 - panel.x0)
                   <= 0.5))
    checks.append(("legend band clears the plotting area",
                   not overlaps(legend.get_window_extent(renderer), panel)))
    checks.append(("every drawn interval is inside the axis range",
                   min(lows + [gate[s] for s in measured]) > x_low
                   and max(highs) < x_high))
    audit(checks)

    out = save(fig, "strata.pdf")
    record_provenance("strata.pdf", [T1, T5])

    # The prose that left the artwork still has to be a build-time reading of
    # the same artifacts, or the caption and the plate can drift apart.
    small = " and ".join(f"{s} ($n = {sizes[s]}$)" for s in sorted(withheld))
    print(
        "caption: A stratum's Ranking Tau is published, and the gate vouches "
        f"for it, only where $n \\geq {threshold}$, so {small} are withheld "
        "and routed to Fallback.\n"
        f"running text: The strata partition all {total} test requests by "
        "tool-set fingerprint against the training vocabulary "
        f"({vocab_names} tool names); {toolless['S1']} of the S1 requests "
        "carry no tools at all. Intervals are session-clustered bootstrap "
        "CIs, and the gate fits its confidence on validation, where S1 and "
        f"S2 hold {validation_n['S1']} and {validation_n['S2']} rows."
    )
    return out


if __name__ == "__main__":
    main()
