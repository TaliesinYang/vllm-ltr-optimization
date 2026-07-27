"""fig:overhead -- publishes the gateway overhead figure into the paper.

The plotting lives in scripts/report_figures/fig8_overhead.py and is not
duplicated here. What this wrapper adds is provenance: the measurement behind
this figure comes from the 2026-07-19 rental, NOT from the trace-calibrated
Block-1 session, and that distinction has to be recorded somewhere a reader
can check rather than left implicit in a manually copied file.

E3's text is scoped to match: it claims a cost of placing the Ranker on the
synchronous path, measured against a direct-to-engine baseline, and says
nothing about the Block-1 workload.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from _common import OKABE_ITO, REPO, record_provenance, save

sys.path.insert(0, str(REPO / "scripts" / "report_figures"))

from fig8_overhead import DATA_PATH, build_figure, load_overhead_pairs  # noqa: E402


FRAME_BOX = dict(boxstyle="square,pad=0.25", facecolor="white",
                 edgecolor="#888888", linewidth=0.6)


def _restyle_panel(ax, summary: dict, header: str, use_log_scale: bool) -> None:
    """Paper-only restyle of one panel, applied to the finished axes.

    The shared generator's layout tests pin the report rendition (delta text
    above the axes), so the paper look is imposed here instead of forking the
    plotting code: distributions faded to background texture, means as bold
    filled dots with heavy butt-capped CI whiskers (the inferential layer
    dominates), the delta reframed as a boxed absolute+ratio readout in the
    gap between the two columns, and a framed header strip naming the panel.
    """
    # Background texture: violins faded to ~0.12, paired hairlines a notch
    # fainter, violin bodies lifted over the hairlines.
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_zorder(1.5)
            collection.set_alpha(0.12)
    for line in ax.lines:
        if line.get_alpha() is not None and abs(line.get_alpha() - 0.055) < 1e-9:
            line.set_alpha(0.045)

    # Mean markers: bold filled dots; CI whiskers: heavy, flat (butt) line
    # ends, no cap ticks.
    markerline, caplines, barcols = ax.containers[0]
    markerline.set_markerfacecolor(OKABE_ITO["black"])
    markerline.set_markeredgecolor("white")
    markerline.set_markeredgewidth(0.6)
    markerline.set_markersize(6.5)
    markerline.set_zorder(6)
    for capline in caplines:
        capline.set_visible(False)
    for barcol in barcols:
        barcol.set_linewidth(2.0)
        barcol.set_capstyle("butt")
        barcol.set_zorder(5)

    # Group means printed beside the black dots (point estimates only; the
    # CI stays graphical as the whisker).
    for x_dot, x_text, ha, mean in (
        (0.0, -0.16, "right", summary["direct_mean"]),
        (1.0, 1.16, "left", summary["gateway_mean"]),
    ):
        ax.text(x_text, mean, f"{mean:.0f} ms", ha=ha, va="center", fontsize=8,
                zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))

    # Delta reframed: framed box holding the absolute delta AND the paired
    # ratio (gateway_mean / direct_mean), both computed from the summaries
    # dict. On the linear panel the gap between the columns at the mean
    # waistline is free; on the log panel that waistline is exactly where
    # both violins are widest, so the box goes into the sparse upper-middle
    # region instead.
    ratio = summary["gateway_mean"] / summary["direct_mean"]
    delta_text = next(text for text in ax.texts if text.get_text().startswith("Δmean"))
    delta_text.set_text(f"{summary['delta_mean']:+.0f} ms ({ratio:.1f}×)")
    delta_text.set_fontsize(9)
    if use_log_scale:
        delta_text.set_position((0.5, 0.80))
    else:
        mid = 0.5 * (summary["direct_mean"] + summary["gateway_mean"])
        delta_text.set_transform(ax.transData)
        delta_text.set_position((0.5, mid))
    delta_text.set_ha("center")
    delta_text.set_va("center")
    delta_text.set_zorder(7)
    delta_text.set_bbox(FRAME_BOX)

    # Framed header strip replaces the plain "(a)"/note texts: a light-gray
    # band spanning the panel width with the panel letter, metric, and n.
    for text in list(ax.texts):
        if text is not delta_text and text.get_transform() is not ax.transData:
            text.remove()
    ax.add_patch(Rectangle((0.0, 1.0), 1.0, 0.075, transform=ax.transAxes,
                           clip_on=False, facecolor="#e8e8e8",
                           edgecolor="#888888", linewidth=0.6, zorder=3))
    ax.text(0.5, 1.0375, header, transform=ax.transAxes, ha="center",
            va="center", fontsize=9, fontweight="bold", zorder=4)


def main() -> None:
    pairs = load_overhead_pairs()
    fig, summaries = build_figure(pairs)
    fig.set_size_inches(fig.get_figwidth(), 3.9)
    paired_n = pairs["all"]["ttft_direct"].size
    matched_n = pairs["matched"]["ttft_direct"].size
    _restyle_panel(fig.axes[0], summaries["ttft"],
                   header=f"(a) TTFT – paired n={paired_n}",
                   use_log_scale=False)
    _restyle_panel(fig.axes[1], summaries["ttlt"],
                   header=f"(b) TTLT – matched n={matched_n}, {pairs['dropped']} dropped",
                   use_log_scale=True)
    # The mark semantics (violin body, paired hairline, mean dot with CI
    # whisker) are not decodable from the panels alone; one boxed legend
    # band above both panels covers the pair.
    handles = [
        Patch(facecolor=OKABE_ITO["gray"], alpha=0.30, label="distribution"),
        Line2D([], [], color=OKABE_ITO["gray"], alpha=0.6, linewidth=0.7,
               label="same request, paired"),
        Line2D([], [], marker="o", color=OKABE_ITO["black"], linewidth=2.0,
               markerfacecolor=OKABE_ITO["black"], markeredgecolor="white",
               markersize=6.5, label="mean $\\pm$ 95% CI"),
    ]
    legend = fig.legend(handles=handles, loc="outside upper center", ncols=3,
                        fontsize=8, frameon=True, framealpha=1.0,
                        edgecolor="#888888", borderpad=0.5, handlelength=1.5,
                        columnspacing=1.6)
    legend.get_frame().set_linewidth(0.6)
    save(fig, "overhead.pdf")
    plt.close(fig)
    record_provenance("overhead.pdf", [Path(DATA_PATH)])

    for metric in ("ttft", "ttlt"):
        s = summaries[metric]
        ratio = s["gateway_mean"] / s["direct_mean"]
        print(f"{metric.upper()} direct_mean_ms={s['direct_mean']:.1f} "
              f"gateway_mean_ms={s['gateway_mean']:.1f} "
              f"delta_mean_ms={s['delta_mean']:.1f} ratio={ratio:.2f} "
              f"CI95=[{s['delta_mean_ci'][0]:.1f}, {s['delta_mean_ci'][1]:.1f}]")
    print(f"TTLT matched pairs dropped: {pairs['dropped']}")


if __name__ == "__main__":
    main()
