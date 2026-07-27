"""Figure 4 - ranking quality by predictor input family.

A horizontal dot-and-interval (forest) plot rather than bars. Kendall's tau is
an estimate with an interval, not a quantity accumulated from zero: bars spend
most of their ink on the uninteresting distance from zero, and truncating them
to fix that exaggerates the differences. The horizontal layout also removes the
rotated category labels, which were themselves a symptom of the wrong
orientation.

Row order is fixed paper-wide (baselines, then ablation, then the proposed
input) so panels can be compared by position. Colour encodes ownership, never
identity: grey baselines, one accent for the proposed input.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    COLOR,
    IEEE_SINGLE_WIDTH,
    LABEL,
    OFFLINE,
    OKABE_ITO,
    load_json,
    record_provenance,
    save,
)

T1 = OFFLINE / "t1-strata.json"
ROWS = (
    "lightgbm_scalar",
    "schema_hash_lookup",
    "lightgbm_grid",
    "bert_prompt_only",
    "bert_prompt_schema",
)
KEY = {
    "bert_prompt_schema": "prompt_schema",
    "bert_prompt_only": "prompt_only",
    "lightgbm_grid": "lightgbm_grid",
    "lightgbm_scalar": "lightgbm_scalar",
    "schema_hash_lookup": "schema_hash_lookup",
}


def main() -> None:
    payload = load_json(T1)
    results = payload["results"]
    record = payload["baseline_of_record"]["model"]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_WIDTH, 2.30),
                           layout="constrained")

    y = np.arange(len(ROWS))
    for index, name in enumerate(ROWS):
        cell = results[name]["all"]
        mean = float(cell["mean_tau_b"])
        low, high = (float(v) for v in cell["ci95_seed17"])
        colour = COLOR[KEY[name]]
        ax.errorbar(
            mean, y[index],
            xerr=[[mean - low], [high - mean]],
            fmt="o", markersize=5.5, linewidth=1.1, capsize=2.5,
            color=colour, markerfacecolor=colour, markeredgecolor=colour,
            zorder=3,
        )
        for seed_value in cell["per_seed_tau_b"].values():
            ax.plot(float(seed_value), y[index], marker="|", markersize=6,
                    color=colour, alpha=0.6, zorder=2)

    baseline = float(results[record]["all"]["mean_tau_b"])
    proposed = float(results["bert_prompt_schema"]["all"]["mean_tau_b"])
    ax.axvline(baseline, color=OKABE_ITO["dark_gray"], linewidth=0.8,
               linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("", xy=(baseline, -0.62), xytext=(proposed, -0.62),
                arrowprops=dict(arrowstyle="<->", linewidth=0.8,
                                color=OKABE_ITO["black"]))
    ax.text((baseline + proposed) / 2, -0.92,
            f"$\\Delta\\tau_b = +{proposed - baseline:.4f}$",
            ha="center", va="bottom", fontsize=10, color=OKABE_ITO["black"])

    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[name] for name in ROWS])
    ax.set_ylim(len(ROWS) - 0.4, -1.15)
    ax.set_xlabel("Kendall $\\tau_b$ (test split, $n{=}999$)")
    ax.set_xlim(0.33, 0.72)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    save(fig, "ranking.pdf")
    record_provenance("ranking.pdf", [T1])
    print(f"baseline of record {record}={baseline:.4f} proposed={proposed:.4f}")


if __name__ == "__main__":
    main()
