"""Figure 3 - Generalization gap: in-distribution (LMSYS) vs cross-distribution (ShareGPT).

Ground truth (docs/RESULTS-E1-gap.md Table E1):
    listMLE: Tau_in = 0.559, Tau_cross = 0.315   (gap 0.243)
    PARS   : Tau_in = 0.596, Tau_cross = 0.361   (gap 0.235)

Dumbbell / slope chart: two x-positions (LMSYS in-dist, ShareGPT cross-dist),
two connected lines (listMLE amber, PARS vermillion), markers at each Tau.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared publication style
# ---------------------------------------------------------------------------
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

# Fixed colorblind-safe palette
COLOR_PARS = "#C2452D"     # vermillion (ours / strong)
COLOR_LISTMLE = "#E8A33D"  # amber (mid)

# ---------------------------------------------------------------------------
# Ground-truth data (exact values, no rounding/smoothing)
# ---------------------------------------------------------------------------
X_IN, X_CROSS = 0.0, 1.0

listmle_in, listmle_cross = 0.559, 0.315  # gap 0.243
pars_in, pars_cross = 0.596, 0.361        # gap 0.235

# Gaps as reported in Table E1 (full-precision values, not the rounded
# Tau differences shown above) -- use these verbatim for honesty/consistency.
LISTMLE_GAP, PARS_GAP = 0.243, 0.235

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.0))

# Connecting slope lines (the "drop")
ax.plot([X_IN, X_CROSS], [listmle_in, listmle_cross],
        color=COLOR_LISTMLE, lw=2.5, zorder=2, label="listMLE")
ax.plot([X_IN, X_CROSS], [pars_in, pars_cross],
        color=COLOR_PARS, lw=2.5, zorder=2, label="PARS (ours)")

# Markers at each Tau
marker_kw = dict(s=90, zorder=3, edgecolor="white", linewidth=1.2)
ax.scatter([X_IN, X_CROSS], [listmle_in, listmle_cross], color=COLOR_LISTMLE, **marker_kw)
ax.scatter([X_IN, X_CROSS], [pars_in, pars_cross], color=COLOR_PARS, **marker_kw)

# Value labels at each marker
ax.annotate(f"{listmle_in:.3f}", (X_IN, listmle_in),
            textcoords="offset points", xytext=(-10, 8),
            ha="right", color=COLOR_LISTMLE, fontsize=11, fontweight="bold")
ax.annotate(f"{listmle_cross:.3f}", (X_CROSS, listmle_cross),
            textcoords="offset points", xytext=(10, -16),
            ha="left", color=COLOR_LISTMLE, fontsize=11, fontweight="bold")
ax.annotate(f"{pars_in:.3f}", (X_IN, pars_in),
            textcoords="offset points", xytext=(-10, 8),
            ha="right", color=COLOR_PARS, fontsize=11, fontweight="bold")
ax.annotate(f"{pars_cross:.3f}", (X_CROSS, pars_cross),
            textcoords="offset points", xytext=(10, 8),
            ha="left", color=COLOR_PARS, fontsize=11, fontweight="bold")

# Per-method gap annotation (honest: show both drops), offset to the left of
# the slope lines so they stay clear of the headline callout.
ax.annotate(f"drop {LISTMLE_GAP:.3f}",
            (0.42, 0.42 * (listmle_cross - listmle_in) + listmle_in),
            color=COLOR_LISTMLE, fontsize=9.5, ha="right", va="top",
            xytext=(-6, -2), textcoords="offset points", style="italic")
ax.annotate(f"drop {PARS_GAP:.3f}",
            (0.42, 0.42 * (pars_cross - pars_in) + pars_in),
            color=COLOR_PARS, fontsize=9.5, ha="right", va="bottom",
            xytext=(-6, 2), textcoords="offset points", style="italic")

# ---------------------------------------------------------------------------
# Headline callout near the cross-dist points (placed in open top-right space)
# ---------------------------------------------------------------------------
rel_gain = (pars_cross - listmle_cross) / listmle_cross * 100  # +14.6% -> "+15%"
ax.annotate(
    "+15% relative\n(0.315 → 0.361)",
    xy=(X_CROSS, pars_cross),
    xytext=(1.18, 0.55),
    fontsize=11.5, fontweight="bold", color=COLOR_PARS,
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.4", fc="#FBEDE9", ec=COLOR_PARS, lw=1.2),
    arrowprops=dict(arrowstyle="->", color=COLOR_PARS, lw=1.4),
)

# ---------------------------------------------------------------------------
# Axes cosmetics
# ---------------------------------------------------------------------------
ax.set_xlim(-0.45, 1.45)
ax.set_ylim(0.0, 0.65)
ax.set_xticks([X_IN, X_CROSS])
ax.set_xticklabels(["LMSYS\n(in-distribution)", "ShareGPT\n(cross-distribution)"])
ax.set_ylabel("Kendall's Tau (rank correlation)")
ax.set_title("Generalization gap: in-distribution vs cross-distribution")

ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="lower left")

# Honest note: both still drop ~0.24 -> overfitting persists
fig.text(
    0.5, -0.02,
    "Both methods still drop ~0.24 Tau on unseen data: overfitting persists. "
    "PARS reduces the gap (0.235 vs 0.243) but does not eliminate it.",
    ha="center", va="top", fontsize=9, color="#444444", style="italic",
)

fig.tight_layout()
fig.savefig("figures/gap.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/gap.pdf", bbox_inches="tight")
print("Saved figures/gap.png and figures/gap.pdf")
