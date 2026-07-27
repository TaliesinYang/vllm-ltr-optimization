"""Figure 6 - Cold-Start Transfer by stratum.

Grouped bars over the two strata large enough to report. S1 and S2 appear as
greyed slots carrying their sample size only: under the n<100 rule their tau is
withheld, and showing a number there would invite exactly the comparison the
rule exists to prevent.

Deliberately no connecting lines between strata. Strata differ in intrinsic
difficulty and label distribution, so only within-stratum, between-model
comparisons are sound; a line would imply a trend across them.
"""

from __future__ import annotations

import json

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
MODELS = (
    "bert_prompt_schema",
    "bert_prompt_only",
    "lightgbm_grid",
    "schema_hash_lookup",
)
REPORTABLE = ("S3", "S4")
WITHHELD = ("S1", "S2")


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    threshold = int(payload["stratum_definition"]["tau_reporting_threshold"])
    sizes = payload["stratum_definition"]["sizes"]

    fig, (ax, ax_small) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 2.7),
        gridspec_kw={"width_ratios": [2.9, 1.0]},
        constrained_layout=True,
    )

    width = 0.2
    base = np.arange(len(REPORTABLE))
    for offset, name in enumerate(MODELS):
        means, lows, highs = [], [], []
        for stratum in REPORTABLE:
            cell = results[name][stratum]
            mean = float(cell["mean_tau_b"])
            low, high = (float(v) for v in cell["ci95_seed17"])
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        position = base + (offset - (len(MODELS) - 1) / 2) * width
        ax.bar(
            position,
            means,
            width=width,
            color=COLOR[name.replace("bert_", "")],
            label=LABEL[name],
            zorder=2,
        )
        ax.errorbar(
            position,
            means,
            yerr=[lows, highs],
            fmt="none",
            ecolor=OKABE_ITO["black"],
            elinewidth=0.8,
            capsize=2.0,
            zorder=4,
        )

    ax.set_xticks(base)
    ax.set_xticklabels(
        [
            f"{stratum}\npartial-new tools\n(n={sizes[stratum]})"
            if stratum == "S3"
            else f"{stratum}\nall-new tools\n(n={sizes[stratum]})"
            for stratum in REPORTABLE
        ]
    )
    ax.set_ylabel("Kendall $\\tau_b$")
    ax.set_ylim(0, 0.78)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=4,
              frameon=False, columnspacing=1.1, handlelength=1.4)

    # Withheld strata: size only, no tau.
    ax_small.set_xlim(-0.6, len(WITHHELD) - 0.4)
    ax_small.set_ylim(0, 0.78)
    # A full-height solid bar reads as a value, and the largest one on the
    # figure at that: exactly the misreading the n<100 rule exists to prevent.
    # Hatched, unfilled slots read as absence instead.
    for index, stratum in enumerate(WITHHELD):
        ax_small.bar(
            index,
            0.78,
            width=0.62,
            facecolor="none",
            edgecolor=OKABE_ITO["light_gray"],
            hatch="///",
            linewidth=0.8,
            zorder=1,
        )
        ax_small.text(
            index,
            0.40,
            f"$\\tau$ withheld",
            ha="center",
            va="center",
            fontsize=10,
            color=OKABE_ITO["dark_gray"],
            rotation=90,
        )
    ax_small.set_xticks(range(len(WITHHELD)))
    ax_small.set_xticklabels(
        [
            f"S1\n(n={sizes['S1']})",
            f"S2\n(n={sizes['S2']})",
        ]
    )
    ax_small.set_yticks([])
    ax_small.spines["left"].set_visible(False)

    save(fig, "fig6.pdf")
    record_provenance("fig6.pdf", [T1])
    print(
        json.dumps(
            {
                "threshold": threshold,
                "sizes": sizes,
                "reported": {
                    name: {
                        stratum: float(results[name][stratum]["mean_tau_b"])
                        for stratum in REPORTABLE
                    }
                    for name in MODELS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
