#!/usr/bin/env python3
"""fig1 (system architecture) + fig2 (decision path w/ gate short-circuit).

Ryoo/EXION standard: minimal text, multi-lane, >=10pt fonts, Okabe-Ito,
matplotlib-only (course rule). No data dependencies.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "sky": "#56B4E9",
}
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
OUT = Path(__file__).resolve().parent.parent / "figs"


def box(ax, x, y, w, h, text, fc, ec="black", fs=10, tc="black", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color="black", style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=lw,
                                 linestyle=ls))


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

    # lanes
    for y, label in ((3.3, "Agent clients"), (1.8, "Gateway tier"), (0.3, "Engine")):
        ax.text(0.06, y + 0.55, label, fontsize=10, style="italic",
                color=OKABE["grey"], rotation=90, va="center")
        ax.axhline(y - 0.12, color=OKABE["grey"], lw=0.4, alpha=0.4)

    box(ax, 0.7, 3.35, 1.9, 0.75, "OpenCode\nagents", "#EAF3FA")
    box(ax, 3.1, 3.35, 2.1, 0.75, "utility calls\n(33% zero-tool)", "#F5F5F5")
    box(ax, 6.2, 3.35, 3.1, 0.75, "request = prompt +\ntool schema (27–69% bytes)", "#FDF3E3")

    box(ax, 0.7, 1.85, 2.6, 0.8, "VeloxMesh gateway\n(admission, routing)", "#EAF3FA")
    box(ax, 4.0, 1.85, 2.8, 0.8, "Decision service\nRanker + Reliability Gate", "#E7F5EF")
    box(ax, 7.4, 1.85, 1.9, 0.8, "fail-open\n$\\leq$ 15 ms", "#FBE9E1", ec=OKABE["red"])

    box(ax, 0.7, 0.35, 3.0, 0.8, "vLLM engine\nQwen3.5-9B, paged KV", "#EFEAF6")
    box(ax, 4.3, 0.35, 2.5, 0.8, "scheduler seam\n(policy per arm)", "#F5F5F5")

    arrow(ax, 1.65, 3.32, 1.9, 2.68)                        # clients -> gateway
    arrow(ax, 7.7, 3.32, 5.4, 2.68)                         # payload -> decision
    arrow(ax, 3.32, 2.25, 3.97, 2.25, color=OKABE["blue"])  # gateway -> decision
    arrow(ax, 6.82, 2.25, 7.37, 2.25, color=OKABE["red"], ls="--")  # timeout
    arrow(ax, 2.0, 1.82, 2.1, 1.18)                         # gateway -> engine
    arrow(ax, 5.4, 1.82, 5.5, 1.18, color=OKABE["green"])   # verdict -> scheduler

    fig.tight_layout()
    fig.savefig(OUT / "fig1.pdf")
    plt.close(fig)


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

    box(ax, 0.35, 2.5, 1.9, 0.75, "Request\nfeatures", "#F5F5F5")
    box(ax, 2.85, 2.5, 2.6, 0.75, "Stratum classifier\n(vocabulary hash)", "#EAF3FA")
    box(ax, 6.1, 2.5, 3.4, 0.75, "Trust table\nS3 0.579 / S4 0.623", "#E7F5EF")
    box(ax, 2.85, 0.5, 2.6, 0.75,
        "Abstention table\nS1 / S2 / zero-tool = 0", "#FBE9E1", ec=OKABE["red"])
    box(ax, 6.1, 0.5, 3.4, 0.75, "Fallback queue\n(arrival order)", "#F5F5F5")
    box(ax, 6.1, 1.5, 3.4, 0.75,
        "BERT fp16, micro-batch $\\leq$8\n37.9 ms p50 $\\rightarrow$ rank order",
        "#EFEAF6")

    arrow(ax, 2.27, 2.87, 2.82, 2.87)
    arrow(ax, 5.47, 2.87, 6.07, 2.87, color=OKABE["green"])
    arrow(ax, 4.15, 2.47, 4.15, 1.28, color=OKABE["red"])          # abstain path
    arrow(ax, 5.47, 0.87, 6.07, 0.87, color=OKABE["red"])
    arrow(ax, 7.8, 2.47, 7.8, 2.28, color=OKABE["green"])          # trusted -> BERT
    ax.text(4.25, 1.7, "no Ranker invocation", fontsize=9, color=OKABE["red"])
    ax.text(0.4, 0.15,
            "slot-preserving: trusted requests reorder only among trusted slots",
            fontsize=9, style="italic", color=OKABE["grey"])

    fig.tight_layout()
    fig.savefig(OUT / "fig2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print(f"wrote {OUT/'fig1.pdf'} and {OUT/'fig2.pdf'}")
