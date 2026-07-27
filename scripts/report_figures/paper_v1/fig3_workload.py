"""Figure 3 - workload characterization of real agent traffic.

Four panels, all recomputed from the raw live trace at build time:
(a) schema bytes per call kind (0/5/8/10-tool requests observed in the trace)
(b) per-request schema share vs total request bytes - the spread the paper
    claims, shown as the 50 tool-bearing points themselves
(c) request mix (zero-tool vs tool-bearing counts)
(d) completion-length ECDFs SPLIT by kind - the pooled median describes
    neither population, which is the figure's point

Byte accounting: canonical JSON (compact separators, sorted keys, UTF-8) of
the tool array over the same serialization of the whole request body. A
different accounting gives a different number, so the method is fixed here.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _common import (
    IEEE_DOUBLE_WIDTH,
    OKABE_ITO,
    PROBE_TRACE,
    load_jsonl,
    record_provenance,
    save,
)
from style import set_log_axis_plain

TRACE = PROBE_TRACE / "agent_trace_vanilla.jsonl.gz"

C_TOOL = OKABE_ITO["blue"]       # tool-bearing (the traffic class we study)
C_ZERO = OKABE_ITO["gray"]       # zero-tool
C_NOTE = OKABE_ITO["dark_gray"]  # annotations / reference lines

# Light->dark blue ramp: request kind ordered by tool count (5 -> 8 -> 10).
# The darkest step is the set-wide house blue; marker SHAPES stay distinct
# so the encoding survives grayscale.
KIND_BLUE = {5: "#8CC5E3", 8: "#4198CB", 10: C_TOOL}
KIND_MARKER = {5: "^", 8: "s", 10: "o"}

# One frame style for every callout, per the figure-craft house rules.
FRAME = {
    "boxstyle": "square,pad=0.25",
    "facecolor": "white",
    "edgecolor": "#888888",
    "linewidth": 0.6,
}


def header_strip(ax, label: str) -> None:
    """Identical light-gray framed header band above each panel."""
    ax.add_patch(
        Rectangle(
            (0.0, 1.035),
            1.0,
            0.115,
            transform=ax.transAxes,
            facecolor="#ececec",
            edgecolor="#888888",
            linewidth=0.6,
            clip_on=False,
            zorder=5,
        )
    )
    ax.text(
        0.5,
        1.0925,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=6,
    )


def canonical_bytes(obj) -> int:
    return len(
        json.dumps(
            obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    )


def p50_pooled(values: np.ndarray) -> float:
    # Ceil-rank pooled percentile, same convention as the rest of the set.
    ordered = np.sort(values)
    return float(ordered[max(0, int(np.ceil(0.5 * ordered.size)) - 1)])


def main() -> None:
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

    kinds = sorted({r["n_tools"] for r in rows})
    by_kind = {k: [r for r in rows if r["n_tools"] == k] for k in kinds}
    # Panel (a) draws one bar per kind from a single request; this only
    # holds if the schema is fixed per kind, so enforce it on the trace.
    for k in kinds:
        distinct = {r["tool_bytes"] for r in by_kind[k]}
        assert len(distinct) == 1, (
            f"schema bytes vary within kind {k}: {sorted(distinct)}"
        )
    tool_rows = [r for r in rows if r["n_tools"] > 0]
    zero_rows = [r for r in rows if r["n_tools"] == 0]
    shares = np.array(
        [r["tool_bytes"] / r["body_bytes"] * 100 for r in tool_rows]
    )
    comp_tool = np.array([r["completion"] for r in tool_rows], dtype=float)
    comp_zero = np.array([r["completion"] for r in zero_rows], dtype=float)
    comp_all = np.array([r["completion"] for r in rows], dtype=float)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(IEEE_DOUBLE_WIDTH, 2.7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.02, 1.12, 0.95, 1.43]},
    )

    # Boxed legend band above the figure: the 5/8/10-tools marker key that
    # panels (a) and (b) share, moved out of the (b) plotting area.
    legend = fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=KIND_MARKER[k],
                linestyle="none",
                markersize=4.5,
                markerfacecolor="none",
                markeredgecolor=KIND_BLUE[k],
                markeredgewidth=1.1,
                label=f"{k} tools",
            )
            for k in (5, 8, 10)
        ],
        loc="outside upper center",
        ncols=3,
        frameon=True,
        fontsize=9,
        handletextpad=0.3,
        columnspacing=1.6,
        borderaxespad=0.15,
    )
    legend.get_frame().set_edgecolor("#888888")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")

    # (a) schema bytes per call kind -----------------------------------------
    ax = axes[0]
    xs = np.arange(len(kinds))
    kib = [by_kind[k][0]["tool_bytes"] / 1024 for k in kinds]
    colors = [C_ZERO if k == 0 else KIND_BLUE[k] for k in kinds]
    bars = ax.bar(xs, kib, color=colors, width=0.62)
    for x, bar, k in zip(xs, bars, kinds):
        ax.text(
            x,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}" if k else "0",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            x,
            -3.4,
            f"n={len(by_kind[k])}",
            ha="center",
            va="top",
            fontsize=8,
            color=C_NOTE,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in kinds])
    ax.set_xlabel("Tools in request", labelpad=13)
    ax.set_ylabel("Schema size (KiB)")
    ax.set_ylim(0, max(kib) * 1.22)
    header_strip(ax, "(a) Schema size")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # (b) schema share vs request size, per request --------------------------
    ax = axes[1]
    for k in kinds:
        if k == 0:
            continue
        sub = by_kind[k]
        ax.scatter(
            [r["body_bytes"] / 1024 for r in sub],
            [r["tool_bytes"] / r["body_bytes"] * 100 for r in sub],
            s=16,
            marker=KIND_MARKER[k],
            facecolors="none",
            edgecolors=KIND_BLUE[k],
            linewidths=1.0,
        )
    lo, hi = shares.min(), shares.max()
    # Upper-right quadrant is empty (high-share requests are all
    # small-bodied), so the takeaway callout sits there.
    ax.text(
        0.96,
        0.94,
        f"{lo:.0f}–{hi:.0f}% of\nrequest bytes",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        color=C_NOTE,
        bbox=dict(FRAME),
    )
    ax.set_xlabel("Request body (KiB)")
    ax.set_ylabel("Schema share (%)")
    body_kib_max = max(r["body_bytes"] / 1024 for r in tool_rows)
    ax.set_xlim(0, body_kib_max * 1.1)
    ax.set_xticks(np.arange(0, body_kib_max * 1.1, 30))
    ax.set_ylim(0, shares.max() * 1.12)
    header_strip(ax, "(b) Schema share")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # (c) request mix --------------------------------------------------------
    ax = axes[2]
    n_tool, n_zero = len(tool_rows), len(zero_rows)
    ax.bar([0], [n_tool], color=C_TOOL, width=0.55)
    ax.bar([0], [n_zero], bottom=[n_tool], color=C_ZERO, width=0.55)
    ax.text(0, n_tool / 2, f"{n_tool} tool-bearing", ha="center",
            va="center", fontsize=9, color="white", rotation=90)
    ax.text(
        0,
        n_tool + n_zero / 2,
        "1/3\nzero-tool",
        ha="center",
        va="center",
        fontsize=9,
        color=C_NOTE,
        bbox=dict(FRAME),
    )
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(0, len(rows))
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel("Live requests")
    header_strip(ax, "(c) Request mix")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # (d) completion-length ECDFs split by kind ------------------------------
    ax = axes[3]
    for values, color in ((comp_zero, C_ZERO), (comp_tool, C_TOOL)):
        ordered = np.sort(values)
        ecdf = np.arange(1, ordered.size + 1) / ordered.size
        ax.step(ordered, ecdf, where="post", color=color)
    p50_zero = p50_pooled(comp_zero)
    p50_tool = p50_pooled(comp_tool)
    p50_all = p50_pooled(comp_all)
    # Plain colour-matched line tags; the takeaway itself is one framed
    # callout instead of two separate p50 labels.
    ax.text(0.02, 0.80, "zero-tool", transform=ax.transAxes,
            fontsize=9, color=C_ZERO, ha="left", va="bottom")
    ax.text(0.97, 0.50, "tool-\nbearing", transform=ax.transAxes,
            fontsize=9, color=C_TOOL, ha="right", va="bottom",
            linespacing=1.0)
    for p50, color in ((p50_zero, C_ZERO), (p50_tool, C_TOOL)):
        ax.plot(p50, 0.5, marker="o", markersize=3.4,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, linestyle="none", zorder=4)
    ax.text(
        0.97,
        0.155,
        f"p50:\n{p50_zero:.0f} vs {p50_tool:.0f}",
        transform=ax.transAxes,
        fontsize=9,
        color=C_NOTE,
        ha="right",
        va="bottom",
        bbox=dict(FRAME),
    )
    ax.axvline(p50_all, ymax=0.80, color=C_NOTE, linestyle=":", linewidth=0.8)
    ax.text(
        p50_all * 0.93,
        0.905,
        f"pooled\np50={p50_all:.0f}",
        fontsize=8,
        color=C_NOTE,
        ha="right",
        va="center",
    )
    # Bottom-right corner: the only region no curve, label, or title enters
    # (the tool-bearing curve is already above y=0.05 at these lengths).
    ax.text(
        comp_tool.max() * 1.30,
        0.045,
        f"max={comp_tool.max():.0f}",
        fontsize=8,
        color=C_NOTE,
        ha="right",
        va="bottom",
    )
    ax.set_xscale("log")
    x_hi = comp_all.max() * 1.35
    ax.set_xlim(1, x_hi)
    ax.set_ylim(0, 1.02)
    log_ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000) if t <= x_hi]
    set_log_axis_plain(ax, "x", log_ticks)
    ax.set_xlabel("Completion length (tokens)")
    ax.set_ylabel("ECDF")
    header_strip(ax, "(d) Completion length")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    save(fig, "workload.pdf")
    record_provenance("workload.pdf", [TRACE])
    print(
        json.dumps(
            {
                "kinds": {
                    str(k): {
                        "n": len(by_kind[k]),
                        "schema_kib": round(
                            by_kind[k][0]["tool_bytes"] / 1024, 2
                        ),
                    }
                    for k in kinds
                },
                "schema_share_min_pct": round(float(shares.min()), 2),
                "schema_share_max_pct": round(float(shares.max()), 2),
                "share_method": "canonical JSON (compact, sorted keys) "
                "tool-array bytes / body bytes, per request",
                "mix": {"tool_bearing": n_tool, "zero_tool": n_zero},
                "p50_zero_tool": p50_zero,
                "p50_tool_bearing": p50_tool,
                "p50_pooled": p50_all,
                "max_completion": float(comp_tool.max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
