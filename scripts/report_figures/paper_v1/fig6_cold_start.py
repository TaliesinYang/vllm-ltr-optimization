"""Figure 6 - Cold-Start Transfer by stratum.

Two aligned forest facets, one per reportable stratum, sharing an x-axis and a
fixed row order so a reader compares by position rather than by hunting for a
colour. Grouped bars were rejected for the same reason as in Figure 4: these
are estimates with intervals, not quantities accumulated from zero.

Deliberately no line connecting the two facets. Strata differ in intrinsic
difficulty and label distribution, so only within-stratum comparisons are
sound; a connector would imply a trend across them.

The two under-powered strata occupy a non-numeric status column rather than the
quantitative axis. A bar, a blank slot or a point at zero would each be read as
a value; a status label cannot be.
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
ROWS = ("lightgbm_scalar", "schema_hash_lookup", "lightgbm_grid",
        "bert_prompt_only", "bert_prompt_schema")
KEY = {"bert_prompt_schema": "prompt_schema", "bert_prompt_only": "prompt_only",
       "lightgbm_grid": "lightgbm_grid", "lightgbm_scalar": "lightgbm_scalar",
       "schema_hash_lookup": "schema_hash_lookup"}
REPORTABLE = (("S3", "partially new tools"), ("S4", "all-new tools"))
WITHHELD = (("S1", "seen combination"), ("S2", "new combination"))


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    threshold = int(payload["stratum_definition"]["tau_reporting_threshold"])
    sizes = payload["stratum_definition"]["sizes"]

    fig, axes = plt.subplots(
        1, 3, figsize=(IEEE_DOUBLE_WIDTH, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.66]},
        sharey=True, layout="constrained",
    )
    y = np.arange(len(ROWS))

    for ax, (stratum, gloss) in zip(axes[:2], REPORTABLE):
        for index, name in enumerate(ROWS):
            cell = results[name][stratum]
            mean = float(cell["mean_tau_b"])
            low, high = (float(v) for v in cell["ci95_seed17"])
            colour = COLOR[KEY[name]]
            ax.errorbar(mean, y[index], xerr=[[mean - low], [high - mean]],
                        fmt="o", markersize=5.0, linewidth=1.1, capsize=2.5,
                        color=colour, markerfacecolor=colour,
                        markeredgecolor=colour, zorder=3)
        ax.set_title(f"{stratum}  {gloss}\n$n{{=}}{sizes[stratum]}$",
                     loc="left", fontsize=10, pad=4)
        ax.set_xlim(0.30, 0.75)
        ax.set_xlabel("Kendall $\\tau_b$")
        ax.set_xticks([0.4, 0.5, 0.6, 0.7])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[name] for name in ROWS])
    axes[0].set_ylim(len(ROWS) - 0.5, -0.5)

    status = axes[2]
    status.set_title("withheld", loc="left", fontsize=10)
    status.set_xlim(0, 1)
    status.axis("off")
    for offset, (stratum, gloss) in enumerate(WITHHELD):
        status.text(0.02, 1.2 + offset * 1.1,
                    f"{stratum}  {gloss}\nWITHHELD  $n{{=}}{sizes[stratum]}<{threshold}$",
                    ha="left", va="center", fontsize=10,
                    color=OKABE_ITO["dark_gray"])

    save(fig, "fig6.pdf")
    record_provenance("fig6.pdf", [T1])
    print("cold-start facets written")


if __name__ == "__main__":
    main()
