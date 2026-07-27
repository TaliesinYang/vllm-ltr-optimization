#!/usr/bin/env python3
"""Build publication-v2 Figures 1--3 from committed evidence.

Figure 1 preserves the implemented request/control path while making the
current placeholder-confidence boundary explicit. Figure 2 is deliberately
limited to offline artifact lineage; it does not infer live serving state from
recorded SHAs. Figure 3 re-plots the single legacy baseline sweep without
inventing repeat-level uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_final_report_figures import (  # noqa: E402
    DEFAULT_BASELINE_SUMMARY,
    DEFAULT_TIER2_LEARNING_CURVE,
    DEFAULT_TIER2_MANIFEST,
    DEFAULT_TIER2_SUMMARY,
    DEFAULT_TRAINING_SOURCES,
    FIG1_CONTROL_LANES,
    FIG2_COMPONENTS,
    FIG2_EDGES,
    parse_baseline_summary,
)
from scripts.report_figures.style import (  # noqa: E402
    IEEE_DOUBLE_WIDTH,
    IEEE_SINGLE_WIDTH,
    OKABE_ITO,
    POLICY_COLOR,
    set_log_axis_plain,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v2"
OFFLINE_SHA256SUMS = (
    REPO_ROOT / "scripts" / "report_figures" / "data" / "offline" / "SHA256SUMS.txt"
)


def _tint(color: str, alpha: float = 0.14) -> tuple[float, float, float, float]:
    return to_rgba(color, alpha)


def _box(
    ax,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    details: Sequence[str],
    edge: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.15,
            edgecolor=edge,
            facecolor=_tint(edge),
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.75,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.36,
        "\n".join(details),
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.12,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    color: str = OKABE_ITO["dark_gray"],
    dashed: bool = False,
    label_offset: tuple[float, float] = (0.0, 0.03),
    connectionstyle: str | None = None,
) -> None:
    props: dict[str, object] = {
        "arrowstyle": "-|>",
        "mutation_scale": 13,
        "linewidth": 1.25,
        "color": color,
        "linestyle": "--" if dashed else "-",
        "shrinkA": 1,
        "shrinkB": 1,
    }
    if connectionstyle:
        props["connectionstyle"] = connectionstyle
    ax.annotate("", xy=end, xytext=start, arrowprops=props)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            midpoint[0] + label_offset[0],
            midpoint[1] + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _short_sha(value: object) -> str:
    if not isinstance(value, str) or len(value) < 8:
        raise ValueError(f"invalid recorded SHA: {value!r}")
    return value[:8]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _require_offline_edge(source: str, target: str) -> None:
    """Keep Fig.2 aligned with the legacy architecture constants."""

    if source not in FIG2_COMPONENTS or target not in FIG2_COMPONENTS:
        raise ValueError(f"unknown Fig.2 component: {source} -> {target}")
    if not any(left == source and right == target for left, right, _ in FIG2_EDGES):
        raise ValueError(f"missing legacy Fig.2 edge: {source} -> {target}")


def build_fig1() -> Figure:
    """Return implemented request/control path with honest reliability scope."""

    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_WIDTH, 4.35))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    learned = POLICY_COLOR["PureLTR"]
    gateway = POLICY_COLOR["GatedHybrid"]
    data = OKABE_ITO["green"]
    neutral = OKABE_ITO["dark_gray"]

    _box(
        ax,
        x=0.02,
        y=0.53,
        width=0.20,
        height=0.27,
        title="Client / Runner",
        details=("chat completions", "Bearer API key", "full history"),
        edge=OKABE_ITO["sky_blue"],
    )
    _box(
        ax,
        x=0.31,
        y=0.53,
        width=0.28,
        height=0.27,
        title="VeloxMesh Gateway\n:9100",
        details=(
            "injects decision verdict",
            "whitelist: ltr_kind, ltr_category",
            "contract failure: fail-open",
        ),
        edge=gateway,
    )
    _box(
        ax,
        x=0.72,
        y=0.53,
        width=0.26,
        height=0.27,
        title="vLLM 0.24\n:8000",
        details=("Qwen3.5-9B", "custom scheduler", "reads sampling extra_args"),
        edge=data,
    )
    _box(
        ax,
        x=0.34,
        y=0.15,
        width=0.38,
        height=0.25,
        title="Decision Service  :9200",
        details=(
            "BERT prompt + schema",
            "confidence = placeholder",
            "no evaluated online OOD detector",
            "rank-quantile mapper + SHA",
        ),
        edge=learned,
    )

    _arrow(
        ax,
        (0.22, 0.665),
        (0.31, 0.665),
        label="HTTP POST\n/v1/chat/completions",
        label_offset=(0.0, 0.17),
    )
    _arrow(
        ax,
        (0.59, 0.665),
        (0.72, 0.665),
        label="upstream chat request\n+ vllm_xargs (int 0/1)",
        label_offset=(0.0, 0.17),
    )
    _arrow(ax, (0.40, 0.53), (0.44, 0.40), color=learned)
    _arrow(ax, (0.65, 0.40), (0.56, 0.53), color=learned)
    _arrow(
        ax,
        (0.72, FIG1_CONTROL_LANES["sse_response"] + 0.03),
        (0.22, FIG1_CONTROL_LANES["sse_response"] + 0.03),
        color=neutral,
        dashed=True,
    )
    ax.text(
        0.26,
        FIG1_CONTROL_LANES["decision_request"] + 0.05,
        "POST /v1/decision",
        ha="center",
        va="center",
        fontsize=10,
        color=learned,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.text(
        0.75,
        FIG1_CONTROL_LANES["decision_response"] + 0.05,
        "verdict + optional estimate\n+ provenance",
        ha="center",
        va="center",
        fontsize=10,
        color=learned,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.text(
        0.47,
        FIG1_CONTROL_LANES["sse_response"] + 0.06,
        "chat SSE response + usage",
        ha="center",
        va="center",
        fontsize=10,
        color=neutral,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.text(0.64, 0.96, "Data plane", color=data, fontsize=10, fontweight="bold", ha="center")
    ax.text(0.18, 0.24, "Control branch", color=learned, fontsize=10, fontweight="bold", ha="center")
    ax.text(
        0.50,
        0.006,
        "Ownership boundary (connected pipeline ≠ sole authorship)\n"
        "Dazhi: predictor/ranker + scheduling integration/analysis · Mingye: gateway infrastructure\n"
        "Yibo: reusable evaluation thread",
        ha="center",
        va="bottom",
        fontsize=10,
        color=neutral,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig


def _offline_lineage() -> dict[str, object]:
    for path in (
        DEFAULT_TRAINING_SOURCES,
        DEFAULT_TIER2_MANIFEST,
        DEFAULT_TIER2_SUMMARY,
        DEFAULT_TIER2_LEARNING_CURVE,
        OFFLINE_SHA256SUMS,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    sources = _json(DEFAULT_TRAINING_SOURCES)
    sample = _json(DEFAULT_TIER2_MANIFEST)
    summary = _json(DEFAULT_TIER2_SUMMARY)
    curve = _json(DEFAULT_TIER2_LEARNING_CURVE)
    toolace = sources.get("toolace")
    backbone = sources.get("bert_backbone")
    tokenizer = sources.get("toolace_label_tokenizer")
    if not all(isinstance(value, dict) for value in (toolace, backbone, tokenizer)):
        raise ValueError("training source declarations are incomplete")
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("tier2 matrix summary has no runs")
    best = max(runs, key=lambda row: float(row["validation_tau"]))
    splits = sample.get("split_counts")
    if not isinstance(splits, dict):
        raise ValueError("tier2 sample manifest has no split_counts")
    exclusions = summary.get("censor_exclusion_counts")
    if not isinstance(exclusions, dict):
        raise ValueError("tier2 matrix summary has no censor_exclusion_counts")
    effective: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        counts = exclusions.get(split)
        if not isinstance(counts, dict) or "eligible" not in counts:
            raise ValueError(f"tier2 matrix summary lacks eligible count for {split}")
        effective[split] = int(counts["eligible"])
    best_run = str(best["run_name"])
    best_label = (
        best_run.removeprefix("bert-")
        .replace("-tier2-", " · ")
        .replace("seed", "seed ")
    )
    return {
        "toolace_revision": _short_sha(toolace["revision"]),
        "tokenizer_revision": _short_sha(tokenizer["revision"]),
        "backbone_revision": _short_sha(backbone["revision"]),
        "sample_count": int(sample["sample_count"]),
        "sample_seed": int(sample["sampling_seed"]),
        "splits": {key: int(splits[key]) for key in ("train", "validation", "test")},
        "effective": effective,
        "sample_manifest_sha": _file_sha(DEFAULT_TIER2_MANIFEST),
        "sample_sha": _short_sha(sample["sample_sha256"]),
        "completed_runs": int(summary["completed_runs"]),
        "expected_runs": int(summary["expected_runs"]),
        "best_run": best_run,
        "best_label": best_label,
        "best_validation_tau": float(best["validation_tau"]),
        "best_test_tau": float(best["test_tau"]),
        "summary_sha": _file_sha(DEFAULT_TIER2_SUMMARY),
        "curve_points": int(curve["completed_points"]),
        "curve_sha": _file_sha(DEFAULT_TIER2_LEARNING_CURVE),
        "sha_ledger_sha": _file_sha(OFFLINE_SHA256SUMS),
    }


def build_fig2() -> Figure:
    """Return offline-only data/model artifact lineage with live file hashes."""

    for edge in (
        ("artifact_store", "label_pipeline"),
        ("label_pipeline", "training_service"),
        ("training_service", "checkpoint_registry"),
        ("training_service", "rank_quantile_mapper"),
    ):
        _require_offline_edge(*edge)
    lineage = _offline_lineage()

    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_WIDTH, 4.75))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    source_color = OKABE_ITO["sky_blue"]
    process_color = OKABE_ITO["orange"]
    learned = POLICY_COLOR["PureLTR"]
    artifact_color = OKABE_ITO["purple"]
    neutral = OKABE_ITO["dark_gray"]

    _box(
        ax,
        x=0.02,
        y=0.57,
        width=0.23,
        height=0.28,
        title="Pinned source\ndeclarations",
        details=(
            f"ToolACE {lineage['toolace_revision']}",
            f"Qwen tokenizer {lineage['tokenizer_revision']}",
            f"BERT backbone {lineage['backbone_revision']}",
        ),
        edge=source_color,
    )
    _box(
        ax,
        x=0.28,
        y=0.54,
        width=0.29,
        height=0.34,
        title="Frozen sample\nmanifest",
        details=(
            f"raw n = {lineage['sample_count']:,} · seed {lineage['sample_seed']}",
            "raw train / val / test",
            f"{lineage['splits']['train']:,} / {lineage['splits']['validation']:,} / {lineage['splits']['test']:,}",
            "analyzed train / val / test",
            f"{lineage['effective']['train']:,} / {lineage['effective']['validation']:,} / {lineage['effective']['test']:,}",
            f"manifest SHA {lineage['sample_manifest_sha']}",
        ),
        edge=process_color,
    )
    _box(
        ax,
        x=0.62,
        y=0.54,
        width=0.35,
        height=0.34,
        title="Training + recorded\nsummaries",
        details=(
            f"matrix {lineage['completed_runs']} / {lineage['expected_runs']} runs",
            "selected by validation τ",
            f"{lineage['best_label']}",
            f"val τ {lineage['best_validation_tau']:.3f} · held-out test τ {lineage['best_test_tau']:.3f}",
            f"matrix SHA {lineage['summary_sha']}",
        ),
        edge=learned,
    )
    _box(
        ax,
        x=0.18,
        y=0.18,
        width=0.28,
        height=0.23,
        title="Checkpoint artifact",
        details=(
            "weights + run metrics",
            "selected by validation τ",
            "serving copy is downstream",
        ),
        edge=artifact_color,
    )
    _box(
        ax,
        x=0.56,
        y=0.18,
        width=0.28,
        height=0.23,
        title="Rank-quantile artifact",
        details=(
            "held-out scores + labels",
            "mapper input produced offline",
            "runtime manifest not loaded",
        ),
        edge=artifact_color,
    )

    _arrow(ax, (0.25, 0.71), (0.28, 0.71))
    _arrow(ax, (0.57, 0.71), (0.62, 0.71))
    _arrow(
        ax,
        (0.72, 0.54),
        (0.39, 0.41),
        label="weights + metrics",
        color=artifact_color,
        label_offset=(-0.015, 0.02),
    )
    _arrow(
        ax,
        (0.83, 0.54),
        (0.70, 0.41),
        label="held-out scores",
        color=artifact_color,
        label_offset=(0.035, 0.0),
    )
    ax.text(
        0.02,
        0.94,
        "Offline artifact lineage only",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=learned,
    )
    ax.text(
        0.98,
        0.94,
        "Recorded provenance does not prove live service state",
        ha="right",
        va="center",
        fontsize=10,
        color=neutral,
    )
    ax.text(
        0.02,
        0.055,
        f"IDs/hashes parsed or computed from committed files at render time · SHA ledger {lineage['sha_ledger_sha']}",
        ha="left",
        va="bottom",
        fontsize=10,
        color=neutral,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig


def _series_index(rates: Sequence[float], target: float) -> int:
    try:
        return rates.index(target)
    except ValueError as exc:
        raise ValueError(f"baseline summary has no rate {target:g}") from exc


def build_fig3(*, baseline_summary: Path = DEFAULT_BASELINE_SUMMARY) -> Figure:
    """Return legacy FCFS/LTR sweep with no invented repeat-level CI."""

    data = parse_baseline_summary(baseline_summary)
    fig, axes = plt.subplots(2, 1, figsize=(IEEE_SINGLE_WIDTH, 6.35))
    fig.subplots_adjust(
        left=0.22, right=0.98, bottom=0.20, top=0.96, hspace=0.58
    )
    styles = {
        "FCFS": {
            "color": POLICY_COLOR["stock_fcfs"],
            "marker": "o",
            "label": "FCFS",
        },
        "LTR": {
            "color": POLICY_COLOR["PureLTR"],
            "marker": "s",
            "label": "Length-aware (legacy LTR)",
        },
    }
    for method in ("FCFS", "LTR"):
        axes[0].plot(
            data[method]["rate"],
            data[method]["ttft_ms"],
            linewidth=1.8,
            markersize=5.5,
            **styles[method],
        )
        axes[1].plot(
            data[method]["rate"],
            data[method]["p99_tpot_ms"],
            linewidth=1.8,
            markersize=5.5,
            **styles[method],
        )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(1.7, 76)
        ax.set_xticks(data["FCFS"]["rate"])
        ax.set_xticklabels([f"{value:g}" for value in data["FCFS"]["rate"]])
        ax.set_xlabel("Request rate (queries/s)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", frameon=False, handlelength=1.3)
    axes[0].set_ylim(35, 450_000)
    axes[1].set_ylim(22, 2_000)
    set_log_axis_plain(axes[0], "y", [50, 100, 1_000, 10_000, 100_000])
    set_log_axis_plain(axes[1], "y", [25, 50, 100, 250, 500, 1_000])
    axes[0].set_ylabel("Mean TTFT (ms)")
    axes[1].set_ylabel("p99 TPOT (ms)")
    axes[0].set_title("(a) Mean TTFT")
    axes[1].set_title("(b) Tail per-token cost")

    rates = data["FCFS"]["rate"]
    index_16 = _series_index(rates, 16.0)
    index_64 = _series_index(rates, 64.0)
    ttft_reduction = 100.0 * (
        1.0
        - data["LTR"]["ttft_ms"][index_16]
        / data["FCFS"]["ttft_ms"][index_16]
    )
    p99_cost = (
        data["LTR"]["p99_tpot_ms"][index_64]
        / data["FCFS"]["p99_tpot_ms"][index_64]
    )
    axes[0].annotate(
        f"{ttft_reduction:.1f}% lower mean TTFT\nat 16 queries/s",
        xy=(16, data["LTR"]["ttft_ms"][index_16]),
        xytext=(24, 900),
        ha="center",
        fontsize=10,
        color=POLICY_COLOR["PureLTR"],
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": POLICY_COLOR["PureLTR"],
            "linewidth": 1.1,
        },
    )
    axes[1].text(
        0.98,
        0.06,
        f"{p99_cost:.2f}× p99 TPOT cost\n(LTR / FCFS) at 64 queries/s",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=POLICY_COLOR["PureLTR"],
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "Legacy stack: Meta-Llama-3-8B-Instruct\n"
        "LMSYS · vLLM 0.4.1 fork · single sweep\n"
        "No repeated runs · no repeat-level CI",
        ha="center",
        va="bottom",
        fontsize=10,
        color=OKABE_ITO["dark_gray"],
    )
    return fig


def _save_pair(
    fig: Figure, output_dir: Path, stem: str, *, fixed_canvas: bool = False
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fixed_metadata = {"Creator": "publication_v2/figures_01_03.py", "CreationDate": None, "ModDate": None}
    bbox = (
        Bbox.from_bounds(0, 0, *fig.get_size_inches())
        if fixed_canvas
        else "tight"
    )
    fig.savefig(pdf_path, metadata=fixed_metadata, bbox_inches=bbox)
    fig.savefig(
        png_path,
        dpi=300,
        metadata={"Software": "publication_v2/figures_01_03.py"},
        bbox_inches=bbox,
    )
    return pdf_path, png_path


def render_all(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    outputs: list[Path] = []
    for number, builder in ((1, build_fig1), (2, build_fig2), (3, build_fig3)):
        figure = builder()
        try:
            outputs.extend(
                _save_pair(
                    figure,
                    output_dir,
                    f"fig{number}",
                    fixed_canvas=number == 3,
                )
            )
        finally:
            plt.close(figure)
    return outputs


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    for output in render_all(args.output_dir):
        print(output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
