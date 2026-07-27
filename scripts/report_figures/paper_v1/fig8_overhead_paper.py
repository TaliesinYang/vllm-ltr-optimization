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
from matplotlib.patches import Patch

from _common import OKABE_ITO, REPO, record_provenance, save

sys.path.insert(0, str(REPO / "scripts" / "report_figures"))

from fig8_overhead import DATA_PATH, build_figure, load_overhead_pairs  # noqa: E402


def _restyle_panel(ax, summary: dict, use_log_scale: bool) -> None:
    """Paper-only restyle of one panel, applied to the finished axes.

    The shared generator's layout tests pin the report rendition (delta text
    above the axes), so the paper look is imposed here instead of forking the
    plotting code: violins drawn over the paired lines, means as filled dots
    with butt-capped CI whiskers, and the delta printed in the gap between
    the two columns where the comparison actually happens.
    """
    # Paired lines stay at zorder 1; lift the violin bodies over them so the
    # thin grey lines read as background texture behind the distributions.
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_zorder(1.5)

    # Mean markers: bold filled dots; CI whiskers: flat (butt) line ends,
    # no cap ticks.
    markerline, caplines, barcols = ax.containers[0]
    markerline.set_markerfacecolor(OKABE_ITO["black"])
    markerline.set_markeredgecolor("white")
    markerline.set_markeredgewidth(0.6)
    markerline.set_markersize(5.2)
    markerline.set_zorder(6)
    for capline in caplines:
        capline.set_visible(False)
    for barcol in barcols:
        barcol.set_linewidth(1.6)
        barcol.set_capstyle("butt")
        barcol.set_zorder(5)

    # Move the delta from the panel margin into empty plot area. On the
    # linear panel the gap between the columns at the mean waistline is free;
    # on the log panel that waistline is exactly where both violins are
    # widest, so the annotation goes into the sparse upper-middle region
    # (above the violin bulk, below the outlier points at the column centres).
    delta_text = next(text for text in ax.texts if text.get_text().startswith("Δmean"))
    delta_text.set_text(delta_text.get_text().replace(" ms [", " ms\n["))
    if use_log_scale:
        delta_text.set_position((0.5, 0.80))
    else:
        mid = 0.5 * (summary["direct_mean"] + summary["gateway_mean"])
        delta_text.set_transform(ax.transData)
        delta_text.set_position((0.5, mid))
    delta_text.set_ha("center")
    delta_text.set_va("center")
    delta_text.set_zorder(7)
    delta_text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.2))


def main() -> None:
    pairs = load_overhead_pairs()
    fig, summaries = build_figure(pairs)
    _restyle_panel(fig.axes[0], summaries["ttft"], use_log_scale=False)
    _restyle_panel(fig.axes[1], summaries["ttlt"], use_log_scale=True)
    # The mark semantics (violin body, paired hairline, mean dot with CI
    # whisker) are not decodable from the panels alone; one legend on the
    # linear panel, whose upper-left corner is empty, covers both.
    handles = [
        Patch(facecolor=OKABE_ITO["gray"], alpha=0.25, label="distribution"),
        Line2D([], [], color=OKABE_ITO["gray"], alpha=0.6, linewidth=0.7,
               label="same request, paired"),
        Line2D([], [], marker="o", color=OKABE_ITO["black"], linewidth=1.4,
               markerfacecolor=OKABE_ITO["black"], markeredgecolor="white",
               markersize=5.2, label="mean $\\pm$ 95% CI"),
    ]
    fig.axes[0].legend(handles=handles, loc="upper left", fontsize=8,
                       frameon=True, framealpha=0.9, edgecolor="none",
                       borderpad=0.5, handlelength=1.5, labelspacing=0.5)
    save(fig, "overhead.pdf")
    plt.close(fig)
    record_provenance("overhead.pdf", [Path(DATA_PATH)])

    for metric in ("ttft", "ttlt"):
        s = summaries[metric]
        print(f"{metric.upper()} delta_mean_ms={s['delta_mean']:.1f} "
              f"CI95=[{s['delta_mean_ci'][0]:.1f}, {s['delta_mean_ci'][1]:.1f}]")
    print(f"TTLT matched pairs dropped: {pairs['dropped']}")


if __name__ == "__main__":
    main()
