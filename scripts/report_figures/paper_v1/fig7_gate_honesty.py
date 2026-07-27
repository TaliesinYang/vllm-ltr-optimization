"""Figure 7 - what the Reliability Gate claims versus what it delivers.

Each rule assigns a confidence per stratum; the realized value is the measured
test tau for that stratum. A point above the diagonal claims more reliability
than it delivers - that region is shaded, because overstatement is the failure
mode the gate exists to prevent. Rule C's abstain (0.0) on the strata it cannot
measure is drawn distinctly: it is a refusal to vouch, not a low estimate.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _common import (
    IEEE_DOUBLE_WIDTH,
    OFFLINE,
    OKABE_ITO,
    load_json,
    record_provenance,
    save,
)

T5 = OFFLINE / "t5-gate.json"
RULES = (
    ("placeholder_0.9", "Placeholder 0.9", OKABE_ITO["vermillion"], "X"),
    ("global_control_no_stratification", "Global (no strata)", OKABE_ITO["orange"], "s"),
    ("C_abstain", "Rule C (shipped)", OKABE_ITO["blue"], "o"),
)


def main() -> None:
    payload = load_json(T5)
    comparison = payload["rule_comparison"]

    fig, (ax, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 2.9),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )

    # --- assigned vs realized, per stratum -----------------------------------
    # A realized-tau x-axis puts S3 and S4 0.0075 apart and their markers
    # overlap, so strata are categorical here. Per stratum: the grey bar is what
    # the Ranker actually delivers, the shaded band above it is the region where
    # a rule would be claiming more than that.
    strata = [row["stratum"] for row in comparison["C_abstain"]["per_stratum"]]
    realized = {
        row["stratum"]: float(row["realized"])
        for row in comparison["C_abstain"]["per_stratum"]
    }
    ceiling = 1.0
    positions = np.arange(len(strata))

    for index, stratum in enumerate(strata):
        value = realized[stratum]
        ax.add_patch(
            plt.Rectangle(
                (index - 0.42, value),
                0.84,
                ceiling - value,
                color=OKABE_ITO["vermillion"],
                alpha=0.12,
                zorder=1,
                linewidth=0,
            )
        )
        ax.bar(
            index, value, width=0.84, color=OKABE_ITO["light_gray"], zorder=2
        )
        ax.plot(
            [index - 0.42, index + 0.42],
            [value, value],
            color=OKABE_ITO["dark_gray"],
            linewidth=1.2,
            zorder=3,
        )

    offsets = np.linspace(-0.22, 0.22, len(RULES))
    for offset, (key, label, color, marker) in zip(offsets, RULES):
        for index, row in enumerate(comparison[key]["per_stratum"]):
            assigned = float(row["assigned"])
            is_abstain = assigned == 0.0
            ax.scatter(
                index + offset,
                assigned,
                s=52 if is_abstain else 38,
                marker=marker,
                facecolor="white" if is_abstain else color,
                edgecolor=color if is_abstain else "white",
                linewidth=1.4 if is_abstain else 0.6,
                zorder=6,
                label=label if index == 0 else None,
            )

    ax.text(
        -0.45,
        0.975,
        "shaded: claimed $>$ delivered",
        fontsize=10,
        color=OKABE_ITO["vermillion"],
        ha="left",
        va="top",
    )
    ax.annotate(
        "Rule C abstains",
        xy=(1 + offsets[2], 0.0),
        xytext=(1.55, 0.135),
        fontsize=10,
        color=OKABE_ITO["blue"],
        ha="left",
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["blue"], "linewidth": 0.8},
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(strata)
    ax.set_xlim(-0.6, len(strata) - 0.4)
    ax.set_ylim(0, ceiling)
    ax.set_xlabel("Cold-Start stratum")
    ax.set_ylabel("Confidence / realized $\\tau_b$")
    ax.set_title("(a) Claimed vs delivered", loc="left")
    ax.legend(loc="lower left", ncol=1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # --- worst overstatement per rule ---------------------------------------
    names, worst, colors = [], [], []
    for key, label, color, _ in RULES:
        names.append(label)
        worst.append(float(comparison[key]["max_overstatement"]))
        colors.append(color)
    bars = ax_bar.barh(np.arange(len(names)), worst, color=colors, height=0.55, zorder=2)
    ax_bar.axvline(0, color=OKABE_ITO["black"], linewidth=0.9, zorder=3)
    for bar, value in zip(bars, worst):
        # Negative bars extend left, so their label goes to the RIGHT of zero;
        # placing it left of the bar collides with the tick labels.
        ax_bar.text(
            value + 0.014 if value >= 0 else 0.014,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    ax_bar.set_yticks(np.arange(len(names)))
    ax_bar.set_yticklabels(names)
    ax_bar.set_xlabel("Worst overstatement\n(assigned $-$ realized $\\tau_b$)")
    ax_bar.set_xlim(-0.09, 0.66)
    ax_bar.set_title("(b) Worst case per rule", loc="left", pad=12)
    ax_bar.xaxis.grid(True)
    ax_bar.set_axisbelow(True)
    ax_bar.annotate(
        "never overstates",
        xy=(worst[2] / 2, 2),
        xytext=(0.19, 2.30),
        fontsize=10,
        color=OKABE_ITO["blue"],
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["blue"], "linewidth": 0.8},
    )

    save(fig, "fig7.pdf")
    record_provenance("fig7.pdf", [T5])
    print(
        json.dumps(
            {
                key: {
                    "max_overstatement": float(comparison[key]["max_overstatement"]),
                    "never_overstates": bool(comparison[key]["never_overstates"]),
                    "per_stratum": {
                        row["stratum"]: {
                            "assigned": float(row["assigned"]),
                            "realized": float(row["realized"]),
                        }
                        for row in comparison[key]["per_stratum"]
                    },
                }
                for key, _, _, _ in RULES
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
