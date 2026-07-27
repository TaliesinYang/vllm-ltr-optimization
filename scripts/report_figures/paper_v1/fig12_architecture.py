#!/usr/bin/env python3
"""fig1 (system architecture) + fig2 (decision path).

Conventions taken from systems-conference architecture figures:
  * one enclosing boundary for the system under study, so a reader sees at a
    glance what is contributed and what is environment;
  * contributed components carry the accent colour and a heavier rule while
    pre-existing ones stay muted -- colour encodes ownership, not identity;
  * a numbered request lifecycle along the arrows, so the figure reads in an
    order instead of as a static parts list;
  * measured facts appear as annotations tied to what they describe, never as
    boxes, because a box reads as a component;
  * the degraded path is drawn in the warning colour, the normal path solid.

matplotlib-only (course rule), >=10 pt text, vector output.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ACCENT = "#0072B2"
ACCENT_FILL = "#DCEBF7"
MUTED = "#6E6E6E"
MUTED_FILL = "#F0F0F0"
WARN = "#D55E00"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
OUT = Path(__file__).resolve().parent.parent / "figs"


def component(ax, x, y, w, h, title, sub=None, *, ours=False, fs=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.06",
        facecolor=ACCENT_FILL if ours else MUTED_FILL,
        edgecolor=ACCENT if ours else MUTED,
        linewidth=1.6 if ours else 1.0, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=fs - 1.5, color=MUTED, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)


def flow(ax, p0, p1, *, n=None, color=INK, dashed=False, label=None,
         lx=0.0, ly=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, color=color,
        linewidth=1.3, linestyle=(0, (4, 2)) if dashed else "-", zorder=2))
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    if n is not None:
        ax.text(mx + lx, my + ly, str(n), fontsize=9, color="white",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.18", fc=color, ec="none"))
    if label:
        ax.text(mx + lx, my + ly, label, fontsize=9, color=color,
                ha="center", va="center", zorder=5)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.85))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    ax.add_patch(Rectangle((2.45, 0.45), 5.45, 3.10, facecolor="none",
                           edgecolor=ACCENT, linewidth=0.9,
                           linestyle=(0, (5, 3)), zorder=1))
    ax.text(2.58, 3.68, "serving stack under study", fontsize=9, color=ACCENT,
            style="italic", ha="left", va="center", zorder=4)

    component(ax, 0.15, 2.45, 2.00, 0.85, "Agent clients", "OpenCode")
    component(ax, 2.65, 2.45, 2.10, 0.85, "Gateway", "admission, routing")
    component(ax, 5.30, 2.30, 2.40, 1.10, "Decision service",
              "Ranker + Reliability Gate", ours=True)
    component(ax, 2.65, 0.70, 2.10, 0.85, "vLLM engine", "Qwen3.5-9B")
    component(ax, 5.30, 0.70, 2.40, 0.85, "Queue order", "policy under test",
              ours=True)
    component(ax, 8.30, 2.45, 1.95, 0.85, "Fallback", "arrival order")

    flow(ax, (2.17, 2.88), (2.61, 2.88), n=1, ly=0.30)
    flow(ax, (4.77, 2.88), (5.26, 2.88), n=2, ly=0.30)
    flow(ax, (6.50, 2.26), (6.50, 1.59), n=3, lx=0.32, color=ACCENT)
    flow(ax, (5.26, 1.12), (4.79, 1.12), n=4, ly=0.30, color=ACCENT)
    flow(ax, (3.55, 2.41), (3.55, 1.59), n=5, lx=-0.32)
    flow(ax, (7.74, 2.88), (8.26, 2.88), color=WARN, dashed=True)

    ax.annotate("15 ms budget,\nfail-open on timeout", xy=(8.00, 2.88),
                xytext=(9.10, 1.60), fontsize=9, color=WARN, ha="center",
                va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color=WARN, linewidth=0.7))
    ax.annotate("tool schema: 25\u201367% of\nrequest bytes (measured)",
                xy=(2.35, 2.88), xytext=(1.15, 1.35), fontsize=9, color=MUTED,
                ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "arch.pdf")
    plt.close(fig)


def fig_decision() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    component(ax, 0.15, 1.85, 1.95, 0.85, "Request", "prompt + schema")
    component(ax, 2.45, 1.85, 2.50, 0.85, "Stratum classifier",
              "hash vs vocabulary", ours=True)
    component(ax, 5.35, 2.05, 2.30, 0.85, "Ranker",
              "BERT fp16, batch $\\leq$ 8", ours=True)
    component(ax, 8.05, 2.05, 2.20, 0.85, "Rank order",
              "trusted slots only", ours=True)
    component(ax, 5.35, 0.45, 2.30, 0.85, "Fallback queue", "arrival order")
    component(ax, 8.05, 0.45, 2.20, 0.85, "Served", "slot preserved")

    flow(ax, (2.12, 2.28), (2.46, 2.28))
    flow(ax, (4.97, 2.28), (5.31, 2.28), color=ACCENT)
    ax.text(5.02, 3.06, "S3 / S4", fontsize=9, color=ACCENT, ha="center")
    flow(ax, (7.69, 2.48), (8.01, 2.48), color=ACCENT)
    flow(ax, (3.40, 1.81), (3.40, 1.02), color=WARN)
    flow(ax, (3.42, 0.88), (5.31, 0.88), color=WARN)
    ax.text(4.60, 1.06, "S1 / S2 / zero-tool", fontsize=9, color=WARN,
            ha="center")
    flow(ax, (7.69, 0.88), (8.01, 0.88), color=WARN)

    ax.text(2.45, 1.58, "no Ranker invocation", fontsize=9, color=WARN,
            ha="left", va="center")
    ax.text(0.15, 0.10,
            "Abstained requests keep their arrival slots; trusted requests are "
            "reordered only among themselves.",
            fontsize=9, color=MUTED, style="italic", ha="left", va="bottom")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "decision.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_decision()
    print(f"wrote {OUT/'arch.pdf'} and {OUT/'decision.pdf'}")
