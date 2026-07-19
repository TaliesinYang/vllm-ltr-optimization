#!/usr/bin/env python3
"""Matplotlib-only figure registry for the human-authored final report.

Fig.1 and Fig.2 are evidence-neutral system diagrams and can be generated now.
Fig.3--Fig.8 deliberately stop until measured artifacts are supplied; this file
must never fabricate values merely to fill a LaTeX slot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FigureSpec:
    identifier: str
    title: str
    data_sources: tuple[str, ...]
    x_label: str | None
    y_label: str | None


FIGURES = {
    "fig1": FigureSpec(
        "fig1",
        "System architecture and request path",
        ("VeloxMesh gateway pin d49d79d", "scheduler_benchmark/vllm_scheduler.py"),
        None,
        None,
    ),
    "fig2": FigureSpec(
        "fig2",
        "Training-artifact and serving-artifact lineage",
        (
            "configs/training_sources.json",
            "tier2-sample-manifest.json",
            "rank_quantiles.json",
        ),
        None,
        None,
    ),
    "fig3": FigureSpec(
        "fig3",
        "Baseline reproduction across offered load",
        ("deliverables/04-evaluation/baseline-2026-06-22 request/run results",),
        "Offered request rate (requests/s)",
        "Latency or throughput (units from source artifact)",
    ),
    "fig4": FigureSpec(
        "fig4",
        "Predictor rank quality under distribution shift",
        ("runs/offline-evidence-r1/scores.jsonl", "offline-analysis.json"),
        "Evaluation distribution",
        "Kendall tau-b",
    ),
    "fig5": FigureSpec(
        "fig5",
        "Predictor comparison and Tier-2 learning curve",
        ("tier2-matrix-summary.json", "tier2-learning-curve.json"),
        "Predictor or Tier-2 train-pool size",
        "Kendall tau-b",
    ),
    "fig6": FigureSpec(
        "fig6",
        "Reliability coverage and empirical error",
        ("runs/offline-evidence-r1/scores.jsonl", "disagreement-diagnostic.json"),
        "Retained coverage",
        "Empirical error or Kendall tau-b",
    ),
    "fig7": FigureSpec(
        "fig7",
        "Scheduler utility across reliability and load regimes",
        ("simulator result JSON produced by scripts/plot_fig6.py",),
        "Predictor reliability or offered load",
        "Paired scheduler utility",
    ),
    "fig8": FigureSpec(
        "fig8",
        "End-to-end latency, goodput, and Pareto frontier",
        ("runs/matrix/", "runs/matrix-ood/", "runs/matrix/parity.json"),
        "Scheduling policy or normalized load",
        "End-to-end TTLT / goodput (units from run manifest)",
    ),
}


def _matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt, FancyArrowPatch, FancyBboxPatch


def _node(ax, patch_type, x: float, y: float, width: float, label: str) -> None:
    patch = patch_type(
        (x, y),
        width,
        0.24,
        boxstyle="round,pad=0.025",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor="#DCEAF7",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.12, label, ha="center", va="center", fontsize=10)


def _arrow(ax, arrow_type, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        arrow_type(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#333333",
        )
    )


def draw_fig1(output: Path) -> None:
    plt, arrow_type, patch_type = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.1, 2.2), constrained_layout=True)
    nodes = (
        (0.02, "Client"),
        (0.27, "VeloxMesh\nGateway"),
        (0.52, "BERT Decision\nService (CPU)"),
        (0.77, "vLLM 0.24\nScheduler + Qwen"),
    )
    for x, label in nodes:
        _node(ax, patch_type, x, 0.46, 0.19, label)
    for left, right in zip(nodes, nodes[1:]):
        _arrow(ax, arrow_type, (left[0] + 0.19, 0.58), (right[0], 0.58))
    _arrow(ax, arrow_type, (0.77, 0.38), (0.21, 0.38))
    ax.text(0.49, 0.29, "chat SSE response", ha="center", va="center", fontsize=10)
    ax.text(0.395, 0.75, "HTTP decision contract", ha="center", va="center", fontsize=10)
    ax.text(0.65, 0.75, "vllm_xargs (int 0/1)", ha="center", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 0.88)
    ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def draw_fig2(output: Path) -> None:
    plt, arrow_type, patch_type = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.1, 2.5), constrained_layout=True)
    nodes = (
        (0.02, 0.58, "Pinned ToolACE\nsource"),
        (0.27, 0.58, "Tier-1 / Tier-2\nlabel artifacts"),
        (0.52, 0.58, "BERT checkpoints\nseeds 17/42/73"),
        (0.77, 0.58, "Decision service\n+ provenance"),
        (0.52, 0.15, "Rank-quantile\nmanifest"),
        (0.77, 0.15, "Gateway + vLLM\nbenchmark"),
    )
    for x, y, label in nodes:
        _node(ax, patch_type, x, y, 0.19, label)
    for start, end in (
        ((0.21, 0.70), (0.27, 0.70)),
        ((0.46, 0.70), (0.52, 0.70)),
        ((0.71, 0.70), (0.77, 0.70)),
        ((0.615, 0.58), (0.615, 0.39)),
        ((0.71, 0.27), (0.77, 0.27)),
        ((0.865, 0.58), (0.865, 0.39)),
    ):
        _arrow(ax, arrow_type, start, end)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.92)
    ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=sorted(FIGURES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true", help="Print figure evidence inputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for key, spec in FIGURES.items():
            print(f"{key}: {spec.title}")
            for source in spec.data_sources:
                print(f"  source: {source}")
            if spec.x_label:
                print(f"  axes: {spec.x_label} | {spec.y_label}")
        return
    if not args.figure or not args.output:
        raise SystemExit("--figure and --output are required unless --list is used")
    if args.figure == "fig1":
        draw_fig1(args.output)
        return
    if args.figure == "fig2":
        draw_fig2(args.output)
        return
    spec = FIGURES[args.figure]
    sources = "; ".join(spec.data_sources)
    raise SystemExit(
        f"{args.figure} requires a measured-data adapter; refusing placeholder data. "
        f"Required sources: {sources}"
    )


if __name__ == "__main__":
    main()
