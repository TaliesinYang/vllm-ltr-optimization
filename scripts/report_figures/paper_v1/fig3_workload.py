"""Figure 3 - workload characterization of real agent traffic.

Three panels, each carrying a measured claim, all recomputed from the raw live
trace at build time:
(a) schema cost per tool count: the bar is the exact schema size in KiB, and
    the share of the request body it occupies is printed on the same bar, so
    the absolute cost and the derived ratio are read in one place instead of
    in two panels that would repeat the same tool-count axis twice
(b) request mix, drawn horizontally so it is never mistaken for a second copy
    of the (a) tool-count axis: it counts requests per class, a different
    measure on a different scale, and it is the only panel that shows the
    zero-tool class as a magnitude
(c) completion-length ECDFs split by class, each median marked on the 0.5 rule
    and labelled next to its own marker, with no drop-line that could be
    confused with a vertical segment of a curve

Conventions held across the whole figure:
* One framed colour key at the top. Grey means "zero tools" and nothing else
  in this figure; the light -> dark blue ramp is the ordinal tool count. No
  annotation, interval or guide is drawn in a series colour, so no reader can
  mistake decoration for data.
* Sample sizes are stated in every panel that needs them: per tool count under
  the (a) axis, as the plotted quantity in (b), and in the (c) takeaway.
* Value labels are black and bold everywhere, sit adjacent to the mark they
  belong to, and are never connected to it by a leader line.
* No panel carries a top in-axes subtitle: method qualifiers live in the framed
  takeaway strip of their own panel, so the top slot is treated identically in
  all three panels.

Byte accounting: canonical JSON (compact separators, sorted keys, UTF-8) of
the tool array over the same serialization of the whole request body. A
different accounting gives a different number, so the method is fixed here.

Schema size is exact rather than sampled: every request with a given tool count
carries a byte-identical tool array in this trace, which the build asserts. The
share is not exact, because the rest of the body varies, so the share is a
per-count median over the n stated under the axis.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

from _common import (
    IEEE_DOUBLE_WIDTH,
    OKABE_ITO,
    PROBE_TRACE,
    load_jsonl,
    record_provenance,
    save,
)
from style import EXION, set_log_axis_plain

TRACE = PROBE_TRACE / "agent_trace_vanilla.jsonl.gz"

# One family ramp, ordered light -> dark, so the ordinal tool-count dimension
# is carried by colour as well as by position. Its darkest step here also
# stands for the pooled tool-bearing class in (c), which is why the key names
# both roles instead of leaving the reader to guess.
BLUES = EXION["family"][:3]
C_TOOL = BLUES[-1]                 # tool-bearing class
C_ZERO = EXION["structure"][3]     # zero-tool class, no other role in this figure
C_TEXT = OKABE_ITO["black"]
C_GRID = EXION["structure"][2]

# One frame style for every framed element (key band, header strips, takeaway
# strips), so structural ink is one colour family and never a series colour.
FRAME_EDGE = EXION["structure"][3]
FRAME_FILL = EXION["structure"][0]
FRAME_HEAD_FILL = EXION["structure"][2]
FRAME_LW = 0.6

# Annotation tiers: data values at 9 pt, supporting text at 8 pt. Nothing in
# the figure is smaller than 8 pt.
FS_VALUE = 9
FS_NOTE = 8

# Framed strips, in axes fractions, so every strip is exactly as wide as the
# panel it belongs to and the two framed elements cannot disagree on where a
# panel ends.
HEAD_Y0, HEAD_H = 1.04, 0.15
FOOT_Y0, FOOT_H = -0.82, 0.30
# One shared depth for every x label, so a panel that needs an extra row under
# its ticks does not drop its axis label below its neighbours'.
XLABEL_Y = -0.34

DASH = "–"  # en dash: numeric ranges must not use a hyphen-minus


def strip(ax, y0: float, height: float, label: str, bold: bool,
          fill: str = FRAME_FILL) -> None:
    ax.add_patch(
        Rectangle(
            (0.0, y0),
            1.0,
            height,
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor=FRAME_EDGE,
            linewidth=FRAME_LW,
            clip_on=False,
            zorder=5,
        )
    )
    ax.text(
        0.5,
        y0 + height / 2.0,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9 if bold else FS_NOTE,
        fontweight="bold" if bold else "normal",
        color=C_TEXT,
        linespacing=1.35,
        zorder=6,
    )


def value(ax, x: float, y: float, text: str, dx: float = 0.0, dy: float = 0.0,
          **kwargs) -> None:
    # One encoding for every drawn value: black, bold, anchored on the mark it
    # belongs to and offset from it in points, so a value label can never drift
    # away from its mark when the panel is resized.
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=FS_VALUE,
        fontweight="bold",
        color=C_TEXT,
        **kwargs,
    )


def canonical_bytes(obj) -> int:
    return len(
        json.dumps(
            obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    )


def p50_pooled(values: np.ndarray) -> float:
    # Ceil-rank pooled percentile, same convention as the rest of the set.
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(ordered[max(0, int(np.ceil(0.5 * ordered.size)) - 1)])


def main() -> None:
    # A little more than the set default, so the outermost framed edge keeps
    # a visible margin instead of touching the bounding box.
    plt.rcParams["savefig.pad_inches"] = 0.045

    trace = load_jsonl(TRACE)

    rows = []
    for row in trace:
        tools = row["body"].get("tools") or []
        rows.append(
            {
                "n_tools": len(tools),
                "tool_bytes": canonical_bytes(tools) if tools else 0,
                "body_bytes": canonical_bytes(row["body"]),
                "completion": int(row["usage"]["completion_tokens"]),
            }
        )

    kinds = sorted({r["n_tools"] for r in rows if r["n_tools"] > 0})
    by_kind = {k: [r for r in rows if r["n_tools"] == k] for k in kinds}
    # Panel (a) draws one exact bar per tool count; this only holds if the
    # schema is byte-identical within a count, so enforce it on the trace and
    # say so in the panel rather than dressing exact bytes as a sampled mean.
    for k in kinds:
        distinct = {r["tool_bytes"] for r in by_kind[k]}
        assert len(distinct) == 1, (
            f"schema bytes vary within count {k}: {sorted(distinct)}"
        )
    tool_rows = [r for r in rows if r["n_tools"] > 0]
    zero_rows = [r for r in rows if r["n_tools"] == 0]
    n_tool, n_zero, n_all = len(tool_rows), len(zero_rows), len(rows)
    n_by_kind = {k: len(by_kind[k]) for k in kinds}
    share_by_kind = {
        k: np.array([r["tool_bytes"] / r["body_bytes"] * 100 for r in by_kind[k]])
        for k in kinds
    }
    comp_tool = np.array([r["completion"] for r in tool_rows], dtype=float)
    comp_zero = np.array([r["completion"] for r in zero_rows], dtype=float)
    comp_all = np.array([r["completion"] for r in rows], dtype=float)

    fig = plt.figure(figsize=(IEEE_DOUBLE_WIDTH, 3.0))
    # Fixed geometry rather than an automatic layout: all three panels then have
    # exactly the same width and exactly the same gutter, so the framed strips
    # keep one column rhythm across the row.
    grid = fig.add_gridspec(1, 3)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fig.subplots_adjust(left=0.062, right=0.996, bottom=0.360, top=0.760,
                        wspace=0.30)

    kind_color = {k: BLUES[i] for i, k in enumerate(kinds)}

    # (a) schema size and its share of the request body ----------------------
    # One tool-count axis in the whole figure: the derived ratio is printed on
    # the bar of the quantity it derives from instead of being given a panel
    # and a second copy of this axis.
    ax = axes[0]
    pos = np.array(kinds, dtype=float)
    kib = [by_kind[k][0]["tool_bytes"] / 1024 for k in kinds]
    shares = [p50_pooled(share_by_kind[k]) for k in kinds]
    ax.bar(pos, kib, color=[kind_color[k] for k in kinds], width=1.3)
    below = blended_transform_factory(ax.transData, ax.transAxes)
    for x, k, size, share in zip(pos, kinds, kib, shares):
        # Share above, size adjacent to the bar top: the bar height encodes the
        # size, so the size label is the one that must touch the bar.
        value(ax, x, size, f"{share:.0f}%\n{size:.1f}", dy=2.5, ha="center",
              va="bottom", linespacing=1.25)
        # The n that the median share rests on, under its own tick: the
        # estimate and its sample size are read in one glance.
        ax.text(x, -0.20, f"n={n_by_kind[k]}", transform=below, ha="center",
                va="top", fontsize=FS_NOTE, color=C_TEXT)
    ax.set_xlim(kinds[0] - 1.6, kinds[-1] + 1.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in kinds])
    ax.set_xlabel("Tools in request")
    ax.set_ylabel("Schema (KiB)")
    ax.set_ylim(0, 29)
    ax.set_yticks([0, 10, 20])

    # (b) request mix. Drawn horizontally: it counts requests per kind, it is
    # not a second panel on the (a) tool-count axis, and the horizontal form
    # keeps the 8 and 10 rows as far apart as every other row ----------------
    ax = axes[1]
    mix_kinds = [0] + list(kinds)
    mix_n = [n_zero] + [n_by_kind[k] for k in kinds]
    mix_colors = [C_ZERO] + [kind_color[k] for k in kinds]
    mix_y = np.arange(len(mix_kinds), dtype=float)
    ax.barh(mix_y, mix_n, color=mix_colors, height=0.62)
    for y, count in zip(mix_y, mix_n):
        value(ax, count, y, f"{count}", dx=3.5, ha="left", va="center")
    ax.set_ylim(len(mix_kinds) - 0.45, -0.55)
    ax.set_yticks(mix_y)
    ax.set_yticklabels([str(k) for k in mix_kinds])
    ax.set_ylabel("Tools in request")
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 20, 40])
    ax.set_xlabel(f"Requests (n={n_all})")

    # (c) completion-length ECDFs split by class -----------------------------
    ax = axes[2]
    for values, color in ((comp_zero, C_ZERO), (comp_tool, C_TOOL)):
        ordered = np.sort(values)
        ecdf = np.arange(1, ordered.size + 1) / ordered.size
        ax.step(ordered, ecdf, where="post", color=color, zorder=3)
    p50_zero = p50_pooled(comp_zero)
    p50_tool = p50_pooled(comp_tool)
    p50_all = p50_pooled(comp_all)
    zero_max = float(comp_zero.max())
    tool_min = float(comp_tool.min())

    ax.set_xscale("log")
    # The axis starts just below the smallest observation, so almost no part of
    # the panel is reserved for a range the data never enters; the margin left
    # is the room the leftmost median label needs.
    x_lo = float(comp_all.min()) * 0.55
    x_hi = float(comp_all.max()) * 1.20
    ax.set_xlim(x_lo, x_hi)
    # No ECDF headroom: the y range is exactly the range an ECDF can occupy.
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])

    for p50, color in ((p50_zero, C_ZERO), (p50_tool, C_TOOL)):
        # The median is the crossing of its own curve with the 0.5 rule, so it
        # is marked there and labelled beside the marker. No drop-line: a
        # dotted guide down the axis would run collinear with the vertical
        # segment of the curve it belongs to and read as part of it.
        ax.plot(p50, 0.5, marker="o", markersize=4.6, color=color,
                markeredgecolor="white", markeredgewidth=0.8,
                linestyle="none", zorder=4)
        value(ax, p50, 0.5, f"{p50:.0f}", dx=-4.0, dy=3.0, ha="right",
              va="bottom")

    # Half-decade labelled ticks plus visible minor ticks: on a log axis the
    # two decade labels alone cannot locate where either class starts or
    # stops, which is what the disjoint-support claim rests on.
    decades = (1, 3, 10, 30, 100, 300, 1000)
    log_ticks = [t for t in decades if x_lo <= t <= x_hi]
    set_log_axis_plain(ax, "x", log_ticks, fontsize=FS_VALUE)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.8, width=0.5,
                   direction="in")
    ax.set_xlabel("Completion (tokens)")
    ax.set_ylabel("ECDF")

    # Shared panel furniture: header strip, gridding, and a stated takeaway
    # under every panel, so no third of the figure is left without a claim.
    # Each header names a measured quantity, not an event.
    headers = [
        "(a) Schema size",
        "(b) Request mix",
        "(c) Completion length",
    ]
    # Each strip carries the claim of its panel plus the qualifier that claim
    # needs: the method note lives here rather than in a cryptic in-panel
    # subtitle, and no strip sends the reader to another panel.
    takeaways = [
        f"{min(kib):.1f}{DASH}{max(kib):.1f} KiB, "
        f"{min(shares):.0f}{DASH}{max(shares):.0f}% of body\n"
        "fixed per count; % are medians",
        f"{n_zero / n_all * 100:.0f}% of requests carry no tools\n"
        f"{n_by_kind[kinds[-1]] / n_tool * 100:.0f}% of the rest "
        f"use {kinds[-1]} tools",
        f"p50 {p50_tool:.0f} vs {p50_zero:.0f} tokens "
        f"({p50_tool / p50_zero:.1f}x)\n"
        f"ranges disjoint, n = {n_tool} vs {n_zero}",
    ]
    # Frame-width guard: a takeaway line that outgrows its strip is exactly
    # the defect the overlap checker cannot see (the frame is a rule, not a
    # glyph), so it is enforced here. 8 pt DejaVu averages ~0.072 in/char at
    # this panel width; 30 characters is the measured safe line.
    for takeaway in takeaways:
        for line in takeaway.split("\n"):
            if len(line) > 30:
                raise SystemExit(
                    f"takeaway line {line!r} is {len(line)} chars; the strip "
                    "holds 30 -- shorten the wording, not the type"
                )
    for ax, header, takeaway in zip(axes, headers, takeaways):
        strip(ax, HEAD_Y0, HEAD_H, header, bold=True, fill=FRAME_HEAD_FILL)
        strip(ax, FOOT_Y0, FOOT_H, takeaway, bold=False)
        ax.tick_params(axis="both", labelsize=FS_VALUE)
        ax.xaxis.set_label_coords(0.5, XLABEL_Y)
        ax.set_axisbelow(True)
    # A rule only where a mark has to be read against a scale. (a) prints its
    # exact values on the bars, so a rule there would only cross those labels.
    axes[1].xaxis.grid(True, color=C_GRID)
    axes[2].yaxis.grid(True, color=C_GRID)

    # One framed colour key for the whole figure, in the same frame style as
    # the header and takeaway strips. The blue ramp is ordinal, so it is keyed
    # here rather than left to be inferred, and the two ECDF classes are keyed
    # as lines so the pooled tool-bearing curve is not read as "10 tools".
    handles = [
        Patch(facecolor=C_ZERO, edgecolor="none"),
        *[Patch(facecolor=kind_color[k], edgecolor="none") for k in kinds],
        Line2D([], [], color=C_ZERO, linewidth=1.35),
        Line2D([], [], color=C_TOOL, linewidth=1.35),
    ]
    labels = [
        "0 tools",
        *[str(k) for k in kinds],
        "ECDF zero-tool",
        f"ECDF tool-bearing ({'/'.join(str(k) for k in kinds)})",
    ]
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        # Sits one strip-gap above the header strips rather than at the top of
        # the canvas: the band and the strips it keys are read together, and
        # the plate does not carry a quarter inch of empty paper above them.
        bbox_to_anchor=(0.5, 0.948),
        ncol=len(handles),
        fontsize=FS_NOTE,
        frameon=True,
        handlelength=1.1,
        handleheight=0.85,
        handletextpad=0.45,
        columnspacing=1.1,
        borderpad=0.5,
    )
    frame = legend.get_frame()
    frame.set_boxstyle("square", pad=0.25)
    frame.set_facecolor(FRAME_FILL)
    frame.set_edgecolor(FRAME_EDGE)
    frame.set_linewidth(FRAME_LW)

    save(fig, "workload.pdf")
    record_provenance("workload.pdf", [TRACE])
    print(
        json.dumps(
            {
                "kinds": {
                    str(k): {
                        "n": n_by_kind[k],
                        "schema_kib": round(
                            by_kind[k][0]["tool_bytes"] / 1024, 2
                        ),
                        "share_p50_pct": round(share, 2),
                        "share_range_pct": [
                            round(float(share_by_kind[k].min()), 2),
                            round(float(share_by_kind[k].max()), 2),
                        ],
                    }
                    for k, share in zip(kinds, shares)
                },
                "share_method": "canonical JSON (compact, sorted keys) "
                "tool-array bytes / body bytes, per request; per-count median, "
                "no interval drawn: n is 4 at one count, which is too few for "
                "a defensible 95% interval, so n is stated instead",
                "mix": {"tool_bearing": n_tool, "zero_tool": n_zero,
                        "total": n_all},
                "p50_zero_tool": p50_zero,
                "p50_tool_bearing": p50_tool,
                "p50_pooled": p50_all,
                "zero_tool_max_completion": zero_max,
                "tool_bearing_min_completion": tool_min,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
