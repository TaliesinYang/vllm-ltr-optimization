#!/usr/bin/env python3
"""FIGURE 2 - Predictor ranking quality (Kendall's Tau, in-distribution).

Ground truth (SUMMARY.md Table 2 / RESULTS-E1-gap.md):
  classification (OPT-125M, 10 buckets) = 0.194
  listMLE        (OPT-125M, ranking)    = 0.559
  PARS (ours)    (BERT, margin+delta)   = 0.596
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# --- Shared publication style ---------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 300,
    }
)

# Fixed colorblind-safe palette (Okabe-Ito-ish)
COLOR_GREY = "#7A7A7A"        # neutral / weakest
COLOR_AMBER = "#E8A33D"       # third / mid series
COLOR_VERMILLION = "#C2452D"  # ours / winner

# --- Ground-truth data (DO NOT modify) ------------------------------------
labels = ["classification\n(OPT-125M)", "listMLE\n(OPT-125M)", "PARS (ours)\n(BERT)"]
tau_values = [0.194, 0.559, 0.596]
colors = [COLOR_GREY, COLOR_AMBER, COLOR_VERMILLION]

# Headline number: PARS vs classification ratio
ratio = tau_values[2] / tau_values[0]  # 0.596 / 0.194 = 3.07x

# --- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.5))

x_positions = range(len(tau_values))
bars = ax.bar(
    x_positions,
    tau_values,
    color=colors,
    width=0.62,
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)

# Value labels on top of each bar (3 decimals)
for bar, value in zip(bars, tau_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        value + 0.012,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Axes
ax.set_ylabel("Kendall's Tau (in-dist)")
ax.set_ylim(0.0, 0.7)
ax.set_xticks(list(x_positions))
ax.set_xticklabels(labels)
ax.set_title("Predictor Ranking Quality (in-distribution)")
ax.set_axisbelow(True)
ax.grid(axis="x", visible=False)

# --- Headline callout: ranking >> classification --------------------------
# Curved arrow from the classification bar up to the PARS (winner) bar.
y_low = tau_values[0]
y_high = tau_values[2]
ax.annotate(
    "",
    xy=(1.96, y_high + 0.018),
    xytext=(0.22, y_low + 0.03),
    arrowprops=dict(
        arrowstyle="-|>",
        color=COLOR_VERMILLION,
        lw=2.0,
        connectionstyle="arc3,rad=-0.28",
    ),
)
# Text note placed in the open top-left region (above the short classification bar).
ax.text(
    -0.36,
    0.685,
    "ranking >> classification",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    color=COLOR_VERMILLION,
)
ax.text(
    -0.36,
    0.625,
    f"PARS {ratio:.1f}x higher\nthan classification",
    ha="left",
    va="top",
    fontsize=11,
    fontweight="bold",
    color=COLOR_VERMILLION,
)

fig.tight_layout()

# --- Save both PNG (300dpi) and vector PDF --------------------------------
import os

out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "tau.png")
pdf_path = os.path.join(out_dir, "tau.pdf")

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print(f"Headline ratio PARS/classification = {ratio:.3f}x")
