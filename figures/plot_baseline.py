#!/usr/bin/env python3
"""Figure 1 - Baseline serving latency: FCFS vs LTR (core reproduction).

Ground truth: baseline-2026-06-22/RESULTS-summary.txt (Llama-3-8B, LMSYS).
Left  panel: TTFT vs request rate (log y) -- LTR slashes queueing delay at load.
Right panel: p99 TPOT vs request rate (log y) -- the honest cost: LTR per-token
latency is WORSE (SJF trade-off). Both shown honestly, no hidden losing number.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Publication style (shared across all figures) -------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,          # body / legend tier (consistent across the set)
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,     # axis labels bumped for half-slide legibility
    "xtick.labelsize": 14,    # tick labels bumped (this is the widest figure)
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

# Okabe-Ito-ish fixed palette (consistent across the figure set)
COLOR_FCFS = "#2E5E8C"  # blue   -> FCFS / baseline
COLOR_LTR = "#C2452D"   # vermillion -> LTR / ours
SECONDARY = "#444444"   # single secondary dark tone (matches gap/ablation)

# ---- Ground truth (EXACT values, do not round/smooth) ----------------------
RATES = [2, 4, 8, 16, 32, 64]  # request rate, queries/s

FCFS_TTFT = [50.5, 95.6, 137.1, 17274.2, 91581.5, 258346.9]      # ms
LTR_TTFT = [103.2, 121.6, 203.1, 6043.7, 48088.8, 157546.3]      # ms

FCFS_P99_TPOT = [28.13, 62.01, 97.94, 145.19, 158.86, 171.19]    # ms
LTR_P99_TPOT = [44.18, 65.48, 159.98, 332.94, 715.43, 1400.29]   # ms

MARKER = "o"
# FCFS = baseline (lighter); LTR = ours -> drawn heavier + on top (emphasis).
LW_FCFS, MS_FCFS = 2.2, 6.5
LW_LTR, MS_LTR = 2.8, 7.0

fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(8.5, 4.2))

# ---- Left panel: TTFT vs rate (log y) --------------------------------------
ax_ttft.plot(RATES, FCFS_TTFT, marker=MARKER, lw=LW_FCFS, ms=MS_FCFS,
             color=COLOR_FCFS, label="FCFS", zorder=2)
ax_ttft.plot(RATES, LTR_TTFT, marker=MARKER, lw=LW_LTR, ms=MS_LTR,
             color=COLOR_LTR, label="LTR", zorder=3)
ax_ttft.set_xscale("log", base=2)
ax_ttft.set_yscale("log")
ax_ttft.set_xticks(RATES)
ax_ttft.set_xticklabels(RATES)
ax_ttft.set_xlabel("Request rate (queries/s)")
ax_ttft.set_ylabel("TTFT (ms)")
ax_ttft.set_title("Time to first token (lower is better)")
ax_ttft.set_ylim(top=1.0e6)
ax_ttft.grid(axis="x", visible=False)  # keep only major-decade y-gridlines
ax_ttft.legend(frameon=False, loc="lower right")

# Headline callout at rate=16: FCFS 17274.2 -> LTR 6043.7  (2.86x lower TTFT)
ratio = FCFS_TTFT[3] / LTR_TTFT[3]  # 17274.2 / 6043.7 = 2.858...
ax_ttft.annotate(
    "",
    xy=(16, LTR_TTFT[3]), xytext=(16, FCFS_TTFT[3]),
    arrowprops=dict(arrowstyle="<->", color=SECONDARY, lw=1.6),
)
ax_ttft.annotate(
    f"{ratio:.2f}x lower TTFT\n@ 16 q/s",
    xy=(16, (FCFS_TTFT[3] * LTR_TTFT[3]) ** 0.5),
    xytext=(3.0, 120000),
    fontsize=11.5, fontweight="bold", color=SECONDARY,
    ha="left", va="center",
    arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.3,
                    connectionstyle="arc3,rad=-0.2"),
)

# ---- Right panel: p99 TPOT vs rate (log y) ---------------------------------
ax_tpot.plot(RATES, FCFS_P99_TPOT, marker=MARKER, lw=LW_FCFS, ms=MS_FCFS,
             color=COLOR_FCFS, label="FCFS", zorder=2)
ax_tpot.plot(RATES, LTR_P99_TPOT, marker=MARKER, lw=LW_LTR, ms=MS_LTR,
             color=COLOR_LTR, label="LTR", zorder=3)
ax_tpot.set_xscale("log", base=2)
ax_tpot.set_yscale("log")
ax_tpot.set_xticks(RATES)
ax_tpot.set_xticklabels(RATES)
ax_tpot.set_xlabel("Request rate (queries/s)")
ax_tpot.set_ylabel("p99 TPOT (ms)")
ax_tpot.set_title("p99 per-token latency (honest cost)")
ax_tpot.grid(axis="x", visible=False)  # keep only major-decade y-gridlines
# Legend in the open low-right gap (under the flat FCFS plateau); upper-left
# is reserved for the WORSE callout so the two never overlap.
ax_tpot.legend(frameon=False, loc="lower right")

# Honesty note on the right panel: LTR is worse here (SJF trade-off).
cost = LTR_P99_TPOT[5] / FCFS_P99_TPOT[5]  # 1400.29 / 171.19 = 8.18x
ax_tpot.annotate(
    f"LTR {cost:.1f}x WORSE\n@ 64 q/s (SJF cost)",
    xy=(64, LTR_P99_TPOT[5]),
    xytext=(9, 900),
    fontsize=11.5, fontweight="bold", color=COLOR_LTR,
    ha="center", va="center",
    arrowprops=dict(arrowstyle="->", color=COLOR_LTR, lw=1.3,
                    connectionstyle="arc3,rad=0.2"),
)

fig.suptitle(
    "Baseline serving latency: FCFS vs LTR  (Llama-3-8B, LMSYS, 2026-06-22)",
    fontweight="bold", y=1.02,
)
fig.tight_layout()

# ---- Save both PNG (PPT) + PDF (vector), same basename ---------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(OUT_DIR, "baseline.png")
pdf_path = os.path.join(OUT_DIR, "baseline.pdf")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"saved {png_path} (300dpi) + {pdf_path}")
print(f"headline: {ratio:.2f}x lower TTFT @ 16 q/s ; "
      f"honest cost: {cost:.1f}x worse p99 TPOT @ 64 q/s")
