"""Figure 4 - ranking quality by predictor input family.

A horizontal dot-and-interval (forest) plot rather than bars. Kendall's tau is
an estimate with an interval, not a quantity accumulated from zero: bars spend
most of their ink on the uninteresting distance from zero, and truncating them
to fix that exaggerates the differences. The horizontal layout also removes the
rotated category labels, which were themselves a symptom of the wrong
orientation.

Forest-plot convention: the proposed input reads first (top row), its
ablation directly under it, and the baseline group sits below a visual gap.
Colour encodes ownership, never identity: grey baselines, one accent for the
proposed input. The tuned-baseline mean is a labelled reference line, and the
headline delta is printed in the right margin against a bracket spanning the
two rows it compares.
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
# Top-to-bottom: ours, its ablation, then the baseline group after a gap.
ROWS = (
    "bert_prompt_schema",
    "bert_prompt_only",
    "lightgbm_grid",
    "lightgbm_scalar",
    "schema_hash_lookup",
)
PITCH = 0.62          # row pitch within a group (tight forest rhythm)
GROUP_GAP = 0.55      # extra separation between the BERT and baseline groups
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

    fig, ax = plt.subplots(figsize=(3.4, 1.95), layout="constrained")

    # Manual y positions: tight pitch inside a group, a gap between groups.
    ys, cursor = [], 0.0
    for index, name in enumerate(ROWS):
        if index == 2:
            cursor += GROUP_GAP
        ys.append(cursor)
        cursor += PITCH
    ys = np.asarray(ys)

    highs = []
    for name, row_y in zip(ROWS, ys):
        cell = results[name]["all"]
        mean = float(cell["mean_tau_b"])
        low, high = (float(v) for v in cell["ci95_seed17"])
        highs.append(high)
        colour = COLOR[KEY[name]]
        is_ours = name == "bert_prompt_schema"
        err = ax.errorbar(
            mean, row_y,
            xerr=[[mean - low], [high - mean]],
            fmt="o", markersize=6.0 if is_ours else 5.0,
            linewidth=1.4 if is_ours else 1.1, capsize=0,
            color=colour, markerfacecolor=colour, markeredgecolor=colour,
            zorder=3,
        )
        for bar in err[2]:
            bar.set_capstyle("butt")
        # Per-seed ticks sit just below the error-bar line so they stay
        # visible instead of vanishing under the same-colour interval.
        for seed_value in cell["per_seed_tau_b"].values():
            ax.plot(float(seed_value), row_y + 0.17, marker="|",
                    markersize=3.2, markeredgewidth=0.9, color=colour,
                    alpha=0.7, zorder=2)

    baseline = float(results[record]["all"]["mean_tau_b"])
    proposed = float(results["bert_prompt_schema"]["all"]["mean_tau_b"])

    # Reference line at the tuned-baseline mean, labelled once at the top edge.
    ax.axvline(baseline, color=OKABE_ITO["gray"], linewidth=0.7,
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(baseline, 1.02, "tuned LightGBM grid", ha="center", va="bottom",
            fontsize=10, color=OKABE_ITO["dark_gray"],
            transform=ax.get_xaxis_transform())

    # Headline delta in the right margin, bracket spanning ours -> tuned
    # baseline (the two rows the number compares).
    x_br = max(highs) + 0.025
    y_top, y_base = ys[0], ys[ROWS.index(record)]
    tick = 0.012
    ax.plot([x_br, x_br], [y_top, y_base], color=OKABE_ITO["black"],
            linewidth=0.8, solid_capstyle="butt", clip_on=False, zorder=3)
    for row_y in (y_top, y_base):
        ax.plot([x_br - tick, x_br], [row_y, row_y], color=OKABE_ITO["black"],
                linewidth=0.8, solid_capstyle="butt", clip_on=False, zorder=3)
    ax.text(x_br + 0.016, (y_top + y_base) / 2,
            f"$\\Delta\\tau_b$\n$+{proposed - baseline:.2f}$",
            ha="left", va="center", fontsize=10, color=OKABE_ITO["black"],
            linespacing=1.25)

    # Row labels left-aligned on a common edge; no y tick marks.
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[name] for name in ROWS])
    for label in ax.get_yticklabels():
        label.set_ha("left")
    ax.set_ylim(ys[-1] + 0.45, -0.45)
    # Left-align row labels on a common edge: pad derives from the widest
    # label's rendered extent instead of a hand-tuned constant.
    fig.canvas.draw()
    widest_px = max(lbl.get_window_extent().width
                    for lbl in ax.get_yticklabels())
    ax.tick_params(axis="y", length=0, pad=widest_px * 72.0 / fig.dpi + 3)

    n_test = int(payload["split_sizes"]["test"])
    ax.set_xlabel(f"Kendall $\\tau_b$ (test split, $n{{=}}{n_test}$)")
    ax.set_xlim(0.35, x_br + 0.118)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    # The per-seed tick is the one mark a reader cannot decode from the row
    # labels; the legend names it and the summary mark it hangs under.
    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([], [], marker="o", color=OKABE_ITO["black"], linewidth=1.1,
               markersize=5.0, label="mean $\\pm$ 95% CI"),
        Line2D([], [], marker="|", linestyle="none", color=OKABE_ITO["black"],
               markersize=4.0, markeredgewidth=0.9, label="per-seed $\\tau_b$"),
    ], loc="lower right", fontsize=8, frameon=False, handletextpad=0.4,
        labelspacing=0.4, borderaxespad=0.3)

    save(fig, "ranking.pdf")
    record_provenance("ranking.pdf", [T1])
    print(f"baseline of record {record}={baseline:.4f} proposed={proposed:.4f}")


if __name__ == "__main__":
    main()
