import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Shared publication style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

BLUE = "#2E5E8C"        # baseline / endpoint bars
VERMILLION = "#C2452D"  # the one positive driver
GREY = "#7A7A7A"        # no-help / negative steps (bar FILL only)
DARK_TEXT = "#444444"   # single secondary dark tone for text/lines (matches gap/baseline)

# ----------------------------------------------------------------------------
# GROUND TRUTH (docs/RESULTS-E1-gap.md Table E2, cross-dist Kendall's Tau).
# Exact numbers only -- no rounding, smoothing, or fabrication.
# ----------------------------------------------------------------------------
BASE = 0.315            # base listMLE (OPT)
AFTER_LOSS = 0.303      # +pairwise margin loss (A1, OPT)
AFTER_BACKBONE = 0.368  # +BERT backbone (A2, no filter)
PARS_FULL = 0.361       # +delta-filter (PARS full)

D_LOSS = -0.012         # margin loss: no help
D_BACKBONE = +0.065     # BERT backbone: DOMINANT driver
D_FILTER = -0.007       # delta-filter: no help

Y_FLOOR = 0.29          # zoom y-axis to make the small steps legible

# ----------------------------------------------------------------------------
# Waterfall geometry: (bottom, top, color) per bar
# ----------------------------------------------------------------------------
xs = [0, 1, 2, 3, 4]
labels = ["base\nlistMLE", "+margin loss\n(A1)", "+BERT backbone\n(A2)",
          "+δ-filter", "PARS\n(full)"]

# bottom, top for each bar
bars = [
    (Y_FLOOR, BASE, BLUE),                  # base absolute
    (AFTER_LOSS, BASE, GREY),               # loss step (drop 0.315 -> 0.303)
    (AFTER_LOSS, AFTER_BACKBONE, VERMILLION),  # backbone step (rise 0.303 -> 0.368)
    (PARS_FULL, AFTER_BACKBONE, GREY),      # filter step (drop 0.368 -> 0.361)
    (Y_FLOOR, PARS_FULL, BLUE),             # PARS-full absolute
]

fig, ax = plt.subplots(figsize=(8.5, 5.4))
width = 0.62
for x, (bottom, top, color) in zip(xs, bars):
    ax.bar(x, top - bottom, bottom=bottom, width=width, color=color,
           edgecolor="black", linewidth=0.6, zorder=3)

# ----------------------------------------------------------------------------
# Connector lines linking each cumulative level to the next bar
# ----------------------------------------------------------------------------
cum_levels = [(0, BASE), (1, AFTER_LOSS), (2, AFTER_BACKBONE), (3, PARS_FULL)]
for x, level in cum_levels:
    ax.plot([x - width / 2, x + 1 + width / 2], [level, level],
            color=DARK_TEXT, linestyle="--", linewidth=1.0, zorder=2)

# ----------------------------------------------------------------------------
# Value labels: absolute on blue endpoints, signed delta on each step
# ----------------------------------------------------------------------------
# absolute endpoint values (inside top of blue bars, white)
ax.text(0, BASE - 0.004, f"{BASE:.3f}", ha="center", va="top",
        color="white", fontweight="bold", fontsize=11, zorder=4)
ax.text(4, PARS_FULL - 0.004, f"{PARS_FULL:.3f}", ha="center", va="top",
        color="white", fontweight="bold", fontsize=11, zorder=4)

# signed deltas above each step bar.
# Grey is kept for the bar FILL only; the numbers use the dark secondary tone
# so they stay legible when shrunk onto a slide (A2/A4 contrast fix).
ax.text(1, BASE + 0.0015, f"{D_LOSS:+.3f}", ha="center", va="bottom",
        color=DARK_TEXT, fontweight="bold", fontsize=11)
ax.text(2, AFTER_BACKBONE + 0.004, f"{D_BACKBONE:+.3f}", ha="center", va="bottom",
        color=VERMILLION, fontweight="bold", fontsize=13)
ax.text(3, AFTER_BACKBONE + 0.0015, f"{D_FILTER:+.3f}", ha="center", va="bottom",
        color=DARK_TEXT, fontweight="bold", fontsize=11)

# ----------------------------------------------------------------------------
# Headline callout (the honest negative-result message)
# ----------------------------------------------------------------------------
ax.annotate(
    "BERT backbone (+0.065) is the only component\n"
    "that helps here; margin loss & δ-filter don't.",
    xy=(1.82, AFTER_BACKBONE), xycoords="data",
    xytext=(0.02, 0.97), textcoords="axes fraction",
    ha="left", va="top", fontsize=11.5,
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#FBF3E6",
              edgecolor=VERMILLION, linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color=VERMILLION, linewidth=1.4),
    zorder=5,
)

# ----------------------------------------------------------------------------
# Axes cosmetics
# ----------------------------------------------------------------------------
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylim(Y_FLOOR, 0.392)
ax.set_ylabel("Cross-dist Kendall's Tau")
ax.set_title("Ablation: what drives PARS's cross-distribution gain?",
             fontweight="bold")
ax.set_axisbelow(True)
ax.grid(axis="x", visible=False)

# ----------------------------------------------------------------------------
# Waterfall colour key (B1/C3): this figure intentionally uses a waterfall
# encoding (blue = absolute level, vermillion = the step that helps, grey =
# a step with no help), which differs from the series-colour meaning in the
# other three figures. The explicit key removes the "vermillion = ours"
# cross-figure misread.
# ----------------------------------------------------------------------------
legend_handles = [
    Patch(facecolor=BLUE, edgecolor="black", linewidth=0.6,
          label="Absolute τ (endpoint)"),
    Patch(facecolor=VERMILLION, edgecolor="black", linewidth=0.6,
          label="Step that helps (driver)"),
    Patch(facecolor=GREY, edgecolor="black", linewidth=0.6,
          label="Step with no help"),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
          fontsize=10, handlelength=1.3, columnspacing=1.8)

fig.tight_layout()
fig.savefig("figures/ablation.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/ablation.pdf", bbox_inches="tight")
print("saved figures/ablation.png (300dpi) + figures/ablation.pdf")
