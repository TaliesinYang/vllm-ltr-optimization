"""Figure 6 - Cold-Start Transfer, models-by-strata grid.

Three aligned forest columns (Overall, S3, S4) over a fixed row order and one
shared x-scale, so a reader compares by position in both directions: down a
column ranks models within a stratum, across a row tracks one model as the
tools become unseen. Grouped bars were rejected for the same reason as in
Figure 4: these are estimates with intervals, not quantities accumulated from
zero.

EXION-style in-figure hierarchy: each panel opens with a light-gray header
strip (stratum + n), the prompt+schema-minus-grid delta sits in a framed
callout at the panel's top right, and the rows the delta references (the two
BERT inputs and the grid baseline) carry their mean tau_b printed beside the
CI end. The colour legend is a boxed band above the grid; the two
under-powered strata are a structured footer band under it. A bar, a blank
slot or a point at zero would each be read as a value; a labelled band cannot
be. Baseline greys are ordered with the y-axis (lightest at top), so grey
depth tracks row position.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _common import (
    COLOR,
    IEEE_DOUBLE_WIDTH,
    LABEL,
    OFFLINE,
    OKABE_ITO,
    load_json,
    record_provenance,
    save,
)

T1 = OFFLINE / "t1-strata.json"
# Top-to-bottom: baselines lightest-to-darkest grey, then the blue family
# light-to-dark, so grey/blue depth and y position agree.
ROWS = ("schema_hash_lookup", "lightgbm_scalar", "lightgbm_grid",
        "bert_prompt_only", "bert_prompt_schema")
KEY = {"bert_prompt_schema": "prompt_schema", "bert_prompt_only": "prompt_only",
       "lightgbm_grid": "lightgbm_grid", "lightgbm_scalar": "lightgbm_scalar",
       "schema_hash_lookup": "schema_hash_lookup"}
# Rows whose means the delta callout references; only these get value labels.
LABELLED = ("bert_prompt_schema", "bert_prompt_only", "lightgbm_grid")
COLUMNS = (("all", "Overall"), ("S3", "partially new"), ("S4", "all-new"))
WITHHELD = (("S1", "seen combination"), ("S2", "new combination"))

FRAME = dict(boxstyle="square,pad=0.25", facecolor="white",
             edgecolor="#888888", linewidth=0.6)
HEADER_FACE = "#e8e8e8"
HEADER_TOP, HEADER_BOTTOM = -1.72, -0.98


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    threshold = int(payload["stratum_definition"]["tau_reporting_threshold"])
    sizes = payload["stratum_definition"]["sizes"]

    fig, axes = plt.subplot_mosaic(
        [["all", "S3", "S4"], ["note", "note", "note"]],
        figsize=(IEEE_DOUBLE_WIDTH, 2.75),
        height_ratios=[1.0, 0.13],
        layout="constrained",
    )
    panels = [axes[s] for s, _ in COLUMNS]
    y = np.arange(len(ROWS))

    for column, (ax, (stratum, gloss)) in enumerate(zip(panels, COLUMNS)):
        for index, name in enumerate(ROWS):
            cell = results[name][stratum]
            mean = float(cell["mean_tau_b"])
            low, high = (float(v) for v in cell["ci95_seed17"])
            colour = COLOR[KEY[name]]
            ax.errorbar(mean, y[index], xerr=[[mean - low], [high - mean]],
                        fmt="o", markersize=5.0, linewidth=1.1, capsize=2.5,
                        color=colour, markerfacecolor=colour,
                        markeredgecolor=colour, zorder=3)
            if name in LABELLED:
                ax.text(high + 0.010, y[index], f"{mean:.2f}", ha="left",
                        va="center", fontsize=8, color=colour, zorder=3)

        # Full-width header strip inside the panel frame.
        ax.axhspan(HEADER_TOP, HEADER_BOTTOM, facecolor=HEADER_FACE,
                   edgecolor="none", zorder=1)
        if stratum == "all":
            head = f"{gloss}  $n{{=}}{sizes[stratum]}$"
        else:
            head = f"{stratum} {gloss}  $n{{=}}{sizes[stratum]}$"
        ax.text(0.5, (HEADER_TOP + HEADER_BOTTOM) / 2, head,
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=9, fontweight="bold", color=OKABE_ITO["black"])

        # Framed delta callout, top right inside the frame area.
        delta = (float(results["bert_prompt_schema"][stratum]["mean_tau_b"])
                 - float(results["lightgbm_grid"][stratum]["mean_tau_b"]))
        prefix = "prompt+schema $-$ grid: " if column == 0 else ""
        ax.text(0.97, -0.42, f"{prefix}{delta:+.3f}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, color=COLOR["prompt_schema"], bbox=FRAME,
                zorder=4)

        ax.set_xlim(0.30, 0.76)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_ylim(len(ROWS) - 0.5, HEADER_TOP)
        ax.set_yticks(y)
        if column == 0:
            ax.set_yticklabels([LABEL[name] for name in ROWS])
        else:
            ax.set_yticklabels([])

    panels[1].set_xlabel("Kendall $\\tau_b$")

    # Rows are named on the shared axis; the boxed band above the grid states
    # the one thing the names do not, which family each colour belongs to.
    legend = fig.legend(handles=[
        Line2D([], [], marker="o", color=COLOR["prompt_schema"], linewidth=1.1,
               markersize=5.0, label="BERT (ours)"),
        Line2D([], [], marker="o", color=COLOR["neutral"], linewidth=1.1,
               markersize=5.0, label="baselines"),
    ], loc="outside upper center", ncol=2, fontsize=8, frameon=True,
        handlelength=1.3, handletextpad=0.4, columnspacing=1.6,
        borderaxespad=0.2)
    legend.get_frame().set_edgecolor("#888888")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")

    # Structured footer band: the two withheld strata, spanning all panels.
    note = axes["note"]
    note.set_facecolor("#f2f2f2")
    note.set_xticks([])
    note.set_yticks([])
    for spine in note.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.6)
    parts = "   $\\cdot$   ".join(
        f"{stratum} {gloss}  $n{{=}}{sizes[stratum]}$"
        for stratum, gloss in WITHHELD
    )
    note.text(0.5, 0.5,
              f"$\\bf{{Withheld}}$ ($n{{<}}{threshold}$ reporting "
              f"threshold):   {parts}",
              transform=note.transAxes, ha="center", va="center", fontsize=8.5,
              color=OKABE_ITO["dark_gray"])

    save(fig, "coldstart.pdf")
    record_provenance("coldstart.pdf", [T1])
    print("cold-start grid written")


if __name__ == "__main__":
    main()
