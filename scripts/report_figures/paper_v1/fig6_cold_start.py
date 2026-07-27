"""Figure 6 - Cold-Start Transfer, models-by-strata grid.

Three aligned forest columns (Overall, S3, S4) over a fixed row order and one
shared x-scale, so a reader compares by position in both directions: down a
column ranks models within a stratum, across a row tracks one model as the
tools become unseen. Grouped bars were rejected for the same reason as in
Figure 4: these are estimates with intervals, not quantities accumulated from
zero.

The reasoning step each column must carry is "prompt+schema clears the grid
baseline here, by this much", so that delta is computed from the artifact and
printed at the top of every column rather than left to visual subtraction.

The two under-powered strata are a single status line under the grid. A bar,
a blank slot or a point at zero would each be read as a value; a sentence
cannot be.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
ROWS = ("lightgbm_scalar", "lightgbm_grid", "schema_hash_lookup",
        "bert_prompt_only", "bert_prompt_schema")
KEY = {"bert_prompt_schema": "prompt_schema", "bert_prompt_only": "prompt_only",
       "lightgbm_grid": "lightgbm_grid", "lightgbm_scalar": "lightgbm_scalar",
       "schema_hash_lookup": "schema_hash_lookup"}
COLUMNS = (("all", "Overall"), ("S3", "partially new tools"),
           ("S4", "all-new tools"))
WITHHELD = (("S1", "seen combination"), ("S2", "new combination"))


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    threshold = int(payload["stratum_definition"]["tau_reporting_threshold"])
    sizes = payload["stratum_definition"]["sizes"]

    fig, axes = plt.subplots(
        1, 3, figsize=(IEEE_DOUBLE_WIDTH, 2.35),
        sharey=True, layout="constrained",
    )
    y = np.arange(len(ROWS))

    for column, (ax, (stratum, gloss)) in enumerate(zip(axes, COLUMNS)):
        for index, name in enumerate(ROWS):
            cell = results[name][stratum]
            mean = float(cell["mean_tau_b"])
            low, high = (float(v) for v in cell["ci95_seed17"])
            colour = COLOR[KEY[name]]
            ax.errorbar(mean, y[index], xerr=[[mean - low], [high - mean]],
                        fmt="o", markersize=5.0, linewidth=1.1, capsize=2.5,
                        color=colour, markerfacecolor=colour,
                        markeredgecolor=colour, zorder=3)
        if stratum == "all":
            title = f"{gloss}  $n{{=}}{sizes[stratum]}$"
        else:
            title = f"{stratum}  {gloss}  $n{{=}}{sizes[stratum]}$"
        ax.set_title(title, loc="left", fontsize=10, pad=16)

        delta = (float(results["bert_prompt_schema"][stratum]["mean_tau_b"])
                 - float(results["lightgbm_grid"][stratum]["mean_tau_b"]))
        prefix = "prompt+schema $-$ grid: " if column == 0 else ""
        ax.text(1.0, 1.03, f"{prefix}{delta:+.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                color=COLOR["prompt_schema"])

        ax.set_xlim(0.30, 0.75)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[name] for name in ROWS])
    axes[0].set_ylim(len(ROWS) - 0.5, -0.5)
    axes[1].set_xlabel("Kendall $\\tau_b$")
    # Rows are named on the shared axis; the legend states the one thing the
    # names do not, which family each colour belongs to.
    from matplotlib.lines import Line2D

    axes[0].legend(handles=[
        Line2D([], [], marker="o", color=COLOR["prompt_schema"], linewidth=1.1,
               markersize=5.0, label="BERT (ours)"),
        Line2D([], [], marker="o", color=COLOR["neutral"], linewidth=1.1,
               markersize=5.0, label="baselines"),
    ], loc="lower left", fontsize=8, frameon=False, handlelength=1.3,
        handletextpad=0.4, labelspacing=0.4, borderaxespad=0.3)

    note = "Withheld: " + " and ".join(
        f"{stratum} {gloss} ($n{{=}}{sizes[stratum]}$)"
        for stratum, gloss in WITHHELD
    ) + f" fall below the $n{{\\geq}}{threshold}$ reporting threshold."
    fig.text(0.53, -0.04, note, ha="center", va="top", fontsize=10,
             color=OKABE_ITO["dark_gray"])

    save(fig, "coldstart.pdf")
    record_provenance("coldstart.pdf", [T1])
    print("cold-start grid written")


if __name__ == "__main__":
    main()
