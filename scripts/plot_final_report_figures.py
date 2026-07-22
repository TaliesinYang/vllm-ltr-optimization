#!/usr/bin/env python3
"""Generate the Final Report figures with Matplotlib only.

Reproducible measured inputs:

* Fig.1: ``scripts/server/launch_{gateway,decision,vllm}.sh`` and the gateway
  contract implemented in ``scheduler_benchmark``.
* Fig.2: ``configs/training_sources.json``, ``tier2-sample-manifest.json``,
  ``tier2-final-artifacts.sha256``, and the rank-quantile manifest.
* Fig.3: the measured midterm ``baseline-2026-06-22/RESULTS-summary.txt``.
* Fig.5: ``tier2-matrix-summary.json`` plus the optional
  ``tier1-matrix-summary.json``. Tier-2 uses test tau; Tier-1 is shown in a
  separate validation-tau panel so the two metrics are never mixed. The
  learning-curve subplot reads ``tier2-learning-curve.json`` (seed 42,
  full-context, validation tau-b).

Fig.4/6/7/8 have explicit adapters that print ``PENDING`` and create no file
until their declared measured inputs exist. No placeholder observations are
ever generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
T7_RESULTS = Path("/Volumes/T7 Shield/vllm-ltr-results/extracted")
COURSE_DELIVERABLES = Path(
    "/Volumes/T7 Shield/obsidian/4-Resources/Courses/VPL/FDUClasses/"
    "26VU_CSCI_6806_V1 Computer Sci Gr Capstone Proj/deliverables"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures"
# Offline predictor/baseline evidence is copied into the repo so fig3/fig4 are
# reproducible from a fresh clone (hashes in data/offline/SHA256SUMS.txt). The
# T7_RESULTS/COURSE_DELIVERABLES paths above remain only as original provenance.
OFFLINE_DATA = REPO_ROOT / "scripts" / "report_figures" / "data" / "offline"
DEFAULT_BASELINE_SUMMARY = OFFLINE_DATA / "baseline-2026-06-22-RESULTS-summary.txt"
DEFAULT_TIER1_SUMMARY = OFFLINE_DATA / "tier1-matrix-summary.json"
DEFAULT_TIER2_SUMMARY = OFFLINE_DATA / "tier2-matrix-summary.json"
DEFAULT_TIER2_LEARNING_CURVE = OFFLINE_DATA / "tier2-learning-curve.json"
DEFAULT_TIER2_MANIFEST = OFFLINE_DATA / "tier2-sample-manifest.json"
DEFAULT_TRAINING_SOURCES = REPO_ROOT / "configs" / "training_sources.json"
DEFAULT_RANK_MANIFEST = (
    Path.home()
    / ".cache"
    / "vllm-ltr-optimization"
    / "replay-grpc"
    / "uncalibrated-rank-lookup-v1.json"
)

PUBLICATION_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.24,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FIGURE_SIZES = {
    1: (7.15, 3.9),
    2: (7.15, 6.8),
    3: (3.5, 5.35),
    5: (3.5, 8.25),
}
PREDICTOR_BAR_ORIENTATION = "horizontal"
FIG1_CONTROL_LANES = {
    "decision_request": 0.345,
    "decision_response": 0.375,
    "sse_response": 0.445,
}

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
PURPLE = "#CC79A7"  # Okabe-Ito reddish-purple (style contract requires Okabe-Ito)
GREY = "#666666"
LIGHT_BLUE = "#DCEAF7"
LIGHT_GREEN = "#DDF3EC"
LIGHT_AMBER = "#FAE8C8"
LIGHT_PURPLE = "#EAE2F2"


@dataclass(frozen=True)
class FigureSpec:
    number: int
    title: str
    data_sources: tuple[str, ...]
    x_label: str | None
    y_label: str | None


@dataclass(frozen=True)
class PredictorAggregate:
    key: str
    label: str
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def low(self) -> float:
        return min(self.values)

    @property
    def high(self) -> float:
        return max(self.values)


@dataclass(frozen=True)
class LearningCurve:
    seed: int
    variant: str
    pool_sizes: tuple[int, ...]
    effective_examples: tuple[int, ...]
    validation_tau: tuple[float, ...]


FIG2_COMPONENTS = (
    "artifact_store",
    "label_pipeline",
    "training_service",
    "checkpoint_registry",
    "decision_service",
    "rank_quantile_mapper",
    "gateway",
    "vllm_engine",
    "benchmark_runner",
)

# Static deployment/data relationships. Labels intentionally avoid step ordering.
FIG2_EDGES = (
    ("artifact_store", "label_pipeline", "pinned datasets + replay ledgers"),
    ("label_pipeline", "training_service", "Tier-1 / Tier-2 labels"),
    ("training_service", "checkpoint_registry", "weights + run metrics"),
    ("training_service", "rank_quantile_mapper", "held-out scores + labels"),
    ("checkpoint_registry", "decision_service", "BERT checkpoint"),
    ("rank_quantile_mapper", "decision_service", "rank lookup + provenance"),
    ("benchmark_runner", "gateway", "chat requests + workload metadata"),
    ("gateway", "decision_service", "HTTP /v1/decision + verdict"),
    ("gateway", "vllm_engine", "chat + vllm_xargs"),
    ("vllm_engine", "benchmark_runner", "SSE tokens + usage"),
)
FIG2_EDGE_LABEL_LANES = {"upper_gap": 0.645, "lower_gap": 0.315}


FIGURES = {
    1: FigureSpec(
        1,
        "End-to-end serving architecture",
        (
            "scripts/server/launch_gateway.sh",
            "scripts/server/launch_decision.sh",
            "scripts/server/launch_vllm.sh",
        ),
        None,
        None,
    ),
    2: FigureSpec(
        2,
        "Training and serving system architecture",
        (
            "configs/training_sources.json",
            "tier2-sample-manifest.json",
            "tier2-final-artifacts.sha256",
            "uncalibrated-rank-lookup-v1.json",
        ),
        None,
        None,
    ),
    3: FigureSpec(
        3,
        "FCFS versus length-aware scheduling reproduction",
        (str(DEFAULT_BASELINE_SUMMARY),),
        "Request rate (queries/s)",
        "TTFT / p99 TPOT (ms)",
    ),
    4: FigureSpec(
        4,
        "Distribution-shift predictor quality",
        (
            "runs/offline-evidence-r1/scores.jsonl",
            "runs/offline-evidence-r1/offline-analysis.json",
        ),
        "Evaluation distribution",
        "Kendall tau-b",
    ),
    5: FigureSpec(
        5,
        "Predictor comparison and Tier-2 learning curve",
        (
            str(DEFAULT_TIER2_SUMMARY),
            str(DEFAULT_TIER1_SUMMARY),
            str(DEFAULT_TIER2_LEARNING_CURVE),
        ),
        "Predictor / feature input",
        "Kendall tau-b",
    ),
    6: FigureSpec(
        6,
        "Reliability coverage and empirical error",
        (
            "runs/offline-evidence-r1/scores.jsonl",
            "runs/offline-evidence-r1/disagreement-diagnostic.json",
        ),
        "Retained coverage",
        "Empirical error",
    ),
    7: FigureSpec(
        7,
        "Scheduler regime map",
        ("runs/simulator-final/summary.json",),
        "Predictor reliability / offered load",
        "Paired scheduler utility",
    ),
    8: FigureSpec(
        8,
        "End-to-end serving results",
        ("runs/matrix/", "runs/matrix-ood/", "runs/matrix/parity.json"),
        "Scheduling policy / normalized load",
        "End-to-end latency / goodput",
    ),
}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update(PUBLICATION_STYLE)
    return plt, FancyBboxPatch


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _short_sha(value: object, *, fallback: str = "pending") -> str:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{8,64}", value):
        return value[:8].lower()
    return fallback


def _box(
    ax,
    patch_type,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    details: Sequence[str],
    color: str,
) -> None:
    patch = patch_type(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.15,
        edgecolor="#303030",
        facecolor=color,
    )
    ax.add_patch(patch)
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
        y + height * 0.38,
        "\n".join(details),
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.15,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    color: str = "#303030",
    dashed: bool = False,
    label_offset: tuple[float, float] = (0.0, 0.03),
    connectionstyle: str | None = None,
) -> None:
    arrowprops = {
        "arrowstyle": "-|>",
        "mutation_scale": 13,
        "linewidth": 1.25,
        "color": color,
        "linestyle": "--" if dashed else "-",
        "shrinkA": 1,
        "shrinkB": 1,
    }
    if connectionstyle is not None:
        arrowprops["connectionstyle"] = connectionstyle
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=arrowprops,
    )
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


def draw_fig1(output: Path) -> bool:
    """Draw the measured serving topology from the committed launch scripts."""

    plt, patch_type = _matplotlib()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES[1], constrained_layout=True)

    _box(
        ax,
        patch_type,
        x=0.02,
        y=0.50,
        width=0.20,
        height=0.29,
        title="Client / Runner",
        details=("chat completions", "Bearer API key", "full history"),
        color=LIGHT_BLUE,
    )
    _box(
        ax,
        patch_type,
        x=0.31,
        y=0.50,
        width=0.28,
        height=0.29,
        title="VeloxMesh Gateway\n:9100",
        details=(
            "injects decision verdict",
            "whitelist: kind, category",
            "contract failure: fail-open",
        ),
        color=LIGHT_AMBER,
    )
    _box(
        ax,
        patch_type,
        x=0.72,
        y=0.50,
        width=0.26,
        height=0.29,
        title="vLLM 0.24\n:8000",
        details=(
            "Qwen3.5-9B",
            "custom scheduler",
            "reads sampling extra_args",
        ),
        color=LIGHT_GREEN,
    )
    _box(
        ax,
        patch_type,
        x=0.36,
        y=0.07,
        width=0.33,
        height=0.25,
        title="Decision Service  :9200",
        details=(
            "BERT prompt + schema",
            "OOD / reliability verdict",
            "rank-quantile mapper + SHA",
        ),
        color=LIGHT_PURPLE,
    )

    _arrow(
        ax,
        (0.22, 0.645),
        (0.31, 0.645),
        label="HTTP POST\n/v1/chat/completions",
        label_offset=(0.0, 0.20),
    )
    _arrow(
        ax,
        (0.59, 0.645),
        (0.72, 0.645),
        label="upstream chat request\n+ vllm_xargs (int 0/1)",
        label_offset=(0.0, 0.20),
    )
    _arrow(
        ax,
        (0.40, 0.50),
        (0.43, 0.32),
        color=PURPLE,
    )
    _arrow(
        ax,
        (0.62, 0.32),
        (0.56, 0.50),
        color=PURPLE,
    )
    _arrow(
        ax,
        (0.72, FIG1_CONTROL_LANES["sse_response"]),
        (0.22, FIG1_CONTROL_LANES["sse_response"]),
        color=GREY,
        dashed=True,
    )
    ax.text(
        0.25,
        FIG1_CONTROL_LANES["decision_request"],
        "POST /v1/decision",
        ha="center",
        va="center",
        fontsize=10,
        color=PURPLE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.text(
        0.73,
        FIG1_CONTROL_LANES["decision_response"],
        "verdict + estimate\n+ provenance",
        ha="center",
        va="center",
        fontsize=10,
        color=PURPLE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.text(
        0.47,
        FIG1_CONTROL_LANES["sse_response"] + 0.030,
        "chat SSE response + usage",
        ha="center",
        va="center",
        fontsize=10,
        color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )

    ax.text(
        0.64,
        0.95,
        "Data plane",
        color=GREEN,
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        0.22,
        0.18,
        "Control branch",
        color=PURPLE,
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    _save(fig, output)
    plt.close(fig)
    return True


def draw_fig2(
    output: Path,
    *,
    training_sources: Path = DEFAULT_TRAINING_SOURCES,
    sample_manifest: Path = DEFAULT_TIER2_MANIFEST,
    rank_manifest: Path = DEFAULT_RANK_MANIFEST,
) -> bool:
    """Draw a static component/deployment architecture with pinned provenance."""

    required = (training_sources, sample_manifest, rank_manifest)
    missing = tuple(path for path in required if not path.exists())
    if missing:
        return _pending(2, output, missing)

    sources = _json(training_sources)
    sample = _json(sample_manifest)
    rank = _json(rank_manifest)
    toolace = sources.get("toolace")
    if not isinstance(toolace, dict):
        raise ValueError("training_sources.json has no toolace object")
    toolace_files = toolace.get("files")
    if not isinstance(toolace_files, dict):
        raise ValueError("training_sources.json has no toolace files")

    source_rev = str(toolace.get("revision", "pending"))[:8]
    raw_sha = _short_sha(toolace_files.get("data.json"))
    tier1_sha = _short_sha(sample.get("source_sha256"))
    sample_sha = _short_sha(sample.get("sample_sha256"))
    sample_count = int(sample.get("sample_count", 0))
    checkpoint_sha = _short_sha(rank.get("checkpoint_sha256"))

    plt, patch_type = _matplotlib()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES[2], constrained_layout=True)
    nodes = {
        "artifact_store": (
            0.02,
            0.72,
            0.24,
            0.22,
            "Artifact Store\n(OSS / T7)",
            (
                f"ToolACE {source_rev}",
                "BFCL 61fc0608",
                "Toolathlon 61940341",
            ),
            LIGHT_BLUE,
        ),
        "label_pipeline": (
            0.31,
            0.72,
            0.24,
            0.22,
            "Tier-1 / Tier-2\nLabel Pipeline",
            (
                "Tier-1 n = 13,819",
                f"Tier-2 n = {sample_count:,}",
                "ledger SHA 077eec42",
            ),
            LIGHT_AMBER,
        ),
        "checkpoint_registry": (
            0.02,
            0.37,
            0.24,
            0.22,
            "Checkpoint Registry",
            ("prompt + schema", "selected seed 17", f"SHA {checkpoint_sha}"),
            LIGHT_PURPLE,
        ),
        "training_service": (
            0.31,
            0.37,
            0.24,
            0.22,
            "Training Service",
            ("BERT: 3 x 3", "LightGBM: seed 42", "SHA 6c5abde2"),
            LIGHT_PURPLE,
        ),
        "rank_quantile_mapper": (
            0.31,
            0.07,
            0.24,
            0.20,
            "Rank-Quantile Mapper",
            ("rank lookup v1", "manifest quantiles", "SHA provenance"),
            LIGHT_PURPLE,
        ),
        "benchmark_runner": (
            0.62,
            0.72,
            0.16,
            0.22,
            "Benchmark\nRunner",
            ("workload v2", "metrics", "run manifest"),
            LIGHT_BLUE,
        ),
        "gateway": (
            0.82,
            0.72,
            0.16,
            0.22,
            "VeloxMesh\nGateway",
            ("whitelist", "fail-open", "injects verdict"),
            LIGHT_AMBER,
        ),
        "decision_service": (
            0.62,
            0.37,
            0.16,
            0.22,
            "Decision\nService",
            ("BERT + OOD", "estimate", "provenance"),
            LIGHT_GREEN,
        ),
        "vllm_engine": (
            0.82,
            0.37,
            0.16,
            0.22,
            "vLLM Engine\n:8000",
            ("Qwen3.5-9B", "custom scheduler", "extra_args"),
            LIGHT_GREEN,
        ),
    }
    for values in nodes.values():
        x, y, width, height, title, details, color = values
        _box(
            ax,
            patch_type,
            x=x,
            y=y,
            width=width,
            height=height,
            title=title,
            details=details,
            color=color,
        )

    _arrow(ax, (0.26, 0.83), (0.31, 0.83), label="data", label_offset=(0, 0.035))
    _arrow(ax, (0.43, 0.72), (0.43, 0.59), label="labels", label_offset=(0.045, 0))
    _arrow(ax, (0.31, 0.48), (0.26, 0.48), label="weights", label_offset=(0, 0.035))
    _arrow(ax, (0.43, 0.37), (0.43, 0.27), label="scores", label_offset=(0.045, 0))
    _arrow(
        ax,
        (0.26, 0.55),
        (0.62, 0.55),
        label="checkpoint",
        color=PURPLE,
        label_offset=(0, 0.075),
        connectionstyle="arc3,rad=-0.45",
    )
    _arrow(ax, (0.55, 0.15), (0.68, 0.37), label="quantiles", color=PURPLE, label_offset=(0.025, 0))
    _arrow(ax, (0.78, 0.83), (0.82, 0.83), label="chat", label_offset=(0, 0.035))
    _arrow(ax, (0.87, 0.72), (0.73, 0.59), label="decision", color=PURPLE, label_offset=(0.01, 0.01))
    _arrow(ax, (0.90, 0.72), (0.90, 0.59), label="xargs", label_offset=(0.045, 0))
    _arrow(
        ax,
        (0.98, 0.48),
        (0.70, 0.72),
        label="SSE",
        color=GREY,
        dashed=True,
        label_offset=(0.055, 0.015),
        connectionstyle="arc3,rad=0.45",
    )

    ax.axvline(0.59, color="#BBBBBB", linestyle="--", linewidth=1.0)
    ax.text(0.02, 0.965, "Offline data and model plane", fontsize=10, fontweight="bold", color=BLUE, ha="left")
    ax.text(0.62, 0.965, "Online serving plane", fontsize=10, fontweight="bold", color=GREEN, ha="left")
    ax.text(
        0.02,
        0.015,
        f"Recorded provenance: raw / source / sample SHA {raw_sha} / {tier1_sha} / {sample_sha}",
        fontsize=10,
        color=GREY,
        ha="left",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    _save(fig, output)
    plt.close(fig)
    return True


def parse_baseline_summary(path: Path) -> dict[str, dict[str, list[float]]]:
    """Parse the measured whitespace table used for the midterm reproduction."""

    if not path.exists():
        raise FileNotFoundError(path)
    parsed: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        required = {"method", "rate", "TTFT_ms", "TPOT_ms", "p99_TPOT"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"baseline summary columns must include {sorted(required)}")
        for row in reader:
            method = row["method"]
            record = parsed.setdefault(
                method,
                {"rate": [], "ttft_ms": [], "tpot_ms": [], "p99_tpot_ms": []},
            )
            record["rate"].append(float(row["rate"]))
            record["ttft_ms"].append(float(row["TTFT_ms"]))
            record["tpot_ms"].append(float(row["TPOT_ms"]))
            record["p99_tpot_ms"].append(float(row["p99_TPOT"]))
    if set(parsed) != {"FCFS", "LTR"}:
        raise ValueError(f"expected FCFS and LTR rows, found {sorted(parsed)}")
    for method, record in parsed.items():
        order = sorted(range(len(record["rate"])), key=record["rate"].__getitem__)
        for key in record:
            record[key] = [record[key][index] for index in order]
        if len(set(record["rate"])) != len(record["rate"]):
            raise ValueError(f"duplicate rate for {method}")
    return parsed


def draw_fig3(
    output: Path, *, baseline_summary: Path = DEFAULT_BASELINE_SUMMARY
) -> bool:
    """Plot the measured FCFS/length-aware midterm reproduction without smoothing."""

    if not baseline_summary.exists():
        return _pending(3, output, (baseline_summary,))
    data = parse_baseline_summary(baseline_summary)
    plt, _ = _matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZES[3], constrained_layout=True)
    styles = {
        "FCFS": {"color": BLUE, "marker": "o", "label": "FCFS"},
        "LTR": {
            "color": VERMILLION,
            "marker": "s",
            "label": "Length-aware scheduling",
        },
    }
    for method in ("FCFS", "LTR"):
        axes[0].plot(
            data[method]["rate"],
            data[method]["ttft_ms"],
            linewidth=2.0,
            markersize=5.5,
            **styles[method],
        )
        axes[1].plot(
            data[method]["rate"],
            data[method]["p99_tpot_ms"],
            linewidth=2.0,
            markersize=5.5,
            **styles[method],
        )

    from matplotlib.ticker import FuncFormatter, NullFormatter

    # Plain decimal tick labels keep every glyph at 10pt; the default log
    # formatter renders mathtext exponents (10^3) whose superscript shrinks to 7pt.
    plain_ticks = FuncFormatter(lambda value, _pos: f"{value:g}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(data["FCFS"]["rate"])
        ax.set_xticklabels([f"{value:g}" for value in data["FCFS"]["rate"]])
        ax.set_xlabel("Request rate (queries/s)")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(plain_ticks)
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Time to first token (ms)")
    axes[0].set_title("Queueing latency (lower is better)")
    axes[1].set_ylabel("p99 per-token latency (ms)")
    axes[1].set_title("Tail per-token cost (lower is better)")
    for ax in axes:
        ax.text(
            0.04,
            0.96,
            "FCFS",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=BLUE,
            fontweight="bold",
        )
        ax.text(
            0.04,
            0.86,
            "Length-aware",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=VERMILLION,
            fontweight="bold",
        )

    rates = data["FCFS"]["rate"]
    index_16 = rates.index(16.0)
    ratio = (
        data["FCFS"]["ttft_ms"][index_16]
        / data["LTR"]["ttft_ms"][index_16]
    )
    axes[0].annotate(
        f"{ratio:.2f}x lower\nat 16 queries/s",
        xy=(16, data["LTR"]["ttft_ms"][index_16]),
        xytext=(22, 600),
        ha="center",
        fontsize=10,
        color=VERMILLION,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": VERMILLION, "linewidth": 1.1},
    )
    axes[1].text(
        0.98,
        0.07,
        "8.2x higher p99 TPOT\nat 64 queries/s",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=VERMILLION,
        fontweight="bold",
    )
    _save(fig, output)
    plt.close(fig)
    return True


_BERT_RUN = re.compile(
    r"^bert-(prompt_only|prompt_schema|full_context)-tier[12]-seed(17|42|73)$"
)


def aggregate_predictor_groups(
    summary_path: Path, *, metric: str
) -> tuple[PredictorAggregate, ...]:
    """Aggregate finite predictor metrics; BERT uncertainty is the seed range."""

    payload = _json(summary_path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"summary has no runs list: {summary_path}")
    values: dict[str, list[float]] = {
        "prompt_only": [],
        "prompt_schema": [],
        "full_context": [],
        "lightgbm": [],
    }
    for row in runs:
        if not isinstance(row, dict):
            continue
        run_name = row.get("run_name")
        raw = row.get(metric)
        if not isinstance(run_name, str) or isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        match = _BERT_RUN.fullmatch(run_name)
        if match:
            values[match.group(1)].append(value)
        elif run_name.startswith("lightgbm-structural-"):
            values["lightgbm"].append(value)
    labels = {
        "prompt_only": "BERT\nprompt only",
        "prompt_schema": "BERT\nprompt + schema",
        "full_context": "BERT\nfull context",
        "lightgbm": "LightGBM\nstructural",
    }
    result = tuple(
        PredictorAggregate(key, labels[key], tuple(values[key]))
        for key in labels
        if values[key]
    )
    if not result:
        raise ValueError(f"no finite {metric} values in {summary_path}")
    return result


def load_learning_curve(path: Path) -> LearningCurve:
    """Load measured Tier-2 learning-curve points without interpolation."""

    payload = _json(path)
    seed = payload.get("seed")
    variant = payload.get("variant")
    raw_points = payload.get("points")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"learning curve seed must be an int: {path}")
    if not isinstance(variant, str) or not variant:
        raise ValueError(f"learning curve variant must be a string: {path}")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"learning curve must contain points: {path}")

    points: list[tuple[int, int, float]] = []
    for row in raw_points:
        if not isinstance(row, dict):
            raise ValueError(f"learning curve point must be an object: {path}")
        pool_size = row.get("train_pool_size")
        effective = row.get("effective_train_examples")
        tau = row.get("validation_tau")
        if (
            isinstance(pool_size, bool)
            or not isinstance(pool_size, int)
            or pool_size < 1
            or isinstance(effective, bool)
            or not isinstance(effective, int)
            or effective < 1
            or isinstance(tau, bool)
            or not isinstance(tau, (int, float))
            or not math.isfinite(float(tau))
        ):
            raise ValueError(f"invalid learning curve point: {row!r}")
        points.append((pool_size, effective, float(tau)))
    points.sort(key=lambda item: item[0])
    if len({point[0] for point in points}) != len(points):
        raise ValueError(f"duplicate learning-curve pool size: {path}")
    return LearningCurve(
        seed=seed,
        variant=variant,
        pool_sizes=tuple(point[0] for point in points),
        effective_examples=tuple(point[1] for point in points),
        validation_tau=tuple(point[2] for point in points),
    )


def with_seed_count(
    group: PredictorAggregate, *, expected: int
) -> PredictorAggregate:
    """Put incomplete-seed provenance in the category label, not over the data."""

    if len(group.values) == expected:
        return group
    unit = "seed" if len(group.values) == 1 else "seeds"
    return PredictorAggregate(
        key=group.key,
        label=f"{group.label}\n({len(group.values)} {unit})",
        values=group.values,
    )


def _predictor_panel(ax, groups: Sequence[PredictorAggregate], title: str) -> None:
    colors = (BLUE, VERMILLION, GREEN, GREY)
    ys = list(range(len(groups)))
    means = [group.mean for group in groups]
    lower = [group.mean - group.low for group in groups]
    upper = [group.high - group.mean for group in groups]
    bars = ax.barh(
        ys,
        means,
        height=0.64,
        color=colors[: len(groups)],
        xerr=[lower, upper],
        capsize=4,
        error_kw={"linewidth": 1.2, "capthick": 1.2},
        zorder=2,
    )
    for y, group in zip(ys, groups):
        offsets = (0.0,) if len(group.values) == 1 else tuple(
            -0.15 + index * 0.30 / (len(group.values) - 1)
            for index in range(len(group.values))
        )
        ax.scatter(
            group.values,
            [y + offset for offset in offsets],
            s=23,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.8,
            zorder=3,
        )
    for bar, group in zip(bars, groups):
        ax.text(
            group.high + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{group.mean:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([group.label for group in groups])
    ax.set_xlabel("Kendall tau-b")
    ax.set_ylabel("Predictor / feature input")
    ax.set_xlim(0, 0.75)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


def _learning_curve_panel(ax, curve: LearningCurve) -> None:
    ax.plot(
        curve.pool_sizes,
        curve.validation_tau,
        color=PURPLE,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
    )
    for pool_size, effective, tau in zip(
        curve.pool_sizes, curve.effective_examples, curve.validation_tau
    ):
        first_point = pool_size == min(curve.pool_sizes)
        ax.annotate(
            f"{tau:.3f}\n(n={effective})",
            xy=(pool_size, tau),
            xytext=(14 if first_point else 0,
                    -30 if first_point else (9 if pool_size != max(curve.pool_sizes) else -28)),
            textcoords="offset points",
            ha="left" if first_point else "center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(curve.pool_sizes)
    ax.set_xticklabels([f"{size:,}" for size in curve.pool_sizes])
    ax.set_xlabel("Training-pool target (examples)")
    ax.set_ylabel("Validation Kendall tau-b")
    ax.set_ylim(min(curve.validation_tau) - 0.02, max(curve.validation_tau) + 0.025)
    variant = curve.variant.replace("_", " ")
    ax.set_title(f"Tier-2 data scale: {variant}, seed {curve.seed}")
    ax.grid(axis="x", visible=False)


def draw_fig5(
    output: Path,
    *,
    tier2_summary: Path = DEFAULT_TIER2_SUMMARY,
    tier1_summary: Path = DEFAULT_TIER1_SUMMARY,
    learning_curve: Path = DEFAULT_TIER2_LEARNING_CURVE,
) -> bool:
    """Plot predictor comparisons and the measured Tier-2 learning curve."""

    required = (tier2_summary, learning_curve)
    missing = tuple(path for path in required if not path.exists())
    if missing:
        return _pending(5, output, missing)
    tier2 = aggregate_predictor_groups(tier2_summary, metric="test_tau")
    curve = load_learning_curve(learning_curve)
    include_tier1 = tier1_summary.exists()
    plt, _ = _matplotlib()
    panel_count = 3 if include_tier1 else 2
    fig, axes_value = plt.subplots(
        panel_count,
        1,
        figsize=FIGURE_SIZES[5] if include_tier1 else (3.5, 5.7),
        constrained_layout=True,
    )
    axes = list(axes_value)
    _predictor_panel(axes[0], tier2, "Tier-2 target labels: held-out test")
    if include_tier1:
        tier1 = tuple(
            with_seed_count(group, expected=3)
            for group in aggregate_predictor_groups(
                tier1_summary, metric="validation_tau"
            )
        )
        _predictor_panel(axes[1], tier1, "Tier-1 source labels: validation")
    _learning_curve_panel(axes[-1], curve)
    _save(fig, output)
    plt.close(fig)
    return True


def _pending(number: int, output: Path, sources: Sequence[Path]) -> bool:
    rendered = ", ".join(str(path) for path in sources)
    print(f"PENDING fig{number}: missing measured input(s): {rendered}")
    if output.exists():
        raise RuntimeError(f"refusing to leave stale pending output: {output}")
    return False


def draw_fig4(
    output: Path,
    *,
    sources: Sequence[Path] = (
        REPO_ROOT / "runs/offline-evidence-r1/scores.jsonl",
        REPO_ROOT / "runs/offline-evidence-r1/offline-analysis.json",
    ),
) -> bool:
    """PENDING adapter for the ID/OOD score matrix; never fabricates rows."""

    missing = tuple(path for path in sources if not path.exists())
    return _pending(4, output, missing or tuple(sources))


def draw_fig6(
    output: Path,
    *,
    sources: Sequence[Path] = (
        REPO_ROOT / "runs/offline-evidence-r1/scores.jsonl",
        REPO_ROOT / "runs/offline-evidence-r1/disagreement-diagnostic.json",
    ),
) -> bool:
    """PENDING adapter for reliability coverage; no synthetic confidence curve."""

    missing = tuple(path for path in sources if not path.exists())
    return _pending(6, output, missing or tuple(sources))


def draw_fig7(
    output: Path,
    *,
    sources: Sequence[Path] = (REPO_ROOT / "runs/simulator-final/summary.json",),
) -> bool:
    """PENDING adapter for the final simulator regime map."""

    missing = tuple(path for path in sources if not path.exists())
    return _pending(7, output, missing or tuple(sources))


def draw_fig8(
    output: Path,
    *,
    sources: Sequence[Path] = (
        REPO_ROOT / "runs/matrix",
        REPO_ROOT / "runs/matrix-ood",
        REPO_ROOT / "runs/matrix/parity.json",
    ),
) -> bool:
    """PENDING adapter for the complete gateway-to-vLLM benchmark matrix."""

    missing = tuple(path for path in sources if not path.exists())
    return _pending(8, output, missing or tuple(sources))


def _figure_number(value: str) -> int:
    normalized = value.lower().removeprefix("fig")
    try:
        number = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("figure must be 1..8 or fig1..fig8") from exc
    if number not in FIGURES:
        raise argparse.ArgumentTypeError("figure must be 1..8")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fig", "--figure", type=_figure_number)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--list", action="store_true", help="List evidence inputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, help="Single-figure output override.")
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY
    )
    parser.add_argument("--tier1-summary", type=Path, default=DEFAULT_TIER1_SUMMARY)
    parser.add_argument("--tier2-summary", type=Path, default=DEFAULT_TIER2_SUMMARY)
    parser.add_argument(
        "--tier2-learning-curve", type=Path, default=DEFAULT_TIER2_LEARNING_CURVE
    )
    parser.add_argument("--tier2-manifest", type=Path, default=DEFAULT_TIER2_MANIFEST)
    parser.add_argument(
        "--training-sources", type=Path, default=DEFAULT_TRAINING_SOURCES
    )
    parser.add_argument("--rank-manifest", type=Path, default=DEFAULT_RANK_MANIFEST)
    return parser.parse_args(argv)


def _generate(number: int, args: argparse.Namespace) -> bool:
    output = args.output if args.output and args.fig == number else (
        args.output_dir / f"fig{number}.pdf"
    )
    if number == 1:
        return draw_fig1(output)
    if number == 2:
        return draw_fig2(
            output,
            training_sources=args.training_sources,
            sample_manifest=args.tier2_manifest,
            rank_manifest=args.rank_manifest,
        )
    if number == 3:
        return draw_fig3(output, baseline_summary=args.baseline_summary)
    if number == 4:
        return draw_fig4(output)
    if number == 5:
        return draw_fig5(
            output,
            tier2_summary=args.tier2_summary,
            tier1_summary=args.tier1_summary,
            learning_curve=args.tier2_learning_curve,
        )
    if number == 6:
        return draw_fig6(output)
    if number == 7:
        return draw_fig7(output)
    return draw_fig8(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for number, spec in FIGURES.items():
            print(f"fig{number}: {spec.title}")
            for source in spec.data_sources:
                print(f"  source: {source}")
            if spec.x_label:
                print(f"  axes: {spec.x_label} | {spec.y_label}")
        return 0

    numbers = tuple(FIGURES) if args.all else (args.fig,)
    for number in numbers:
        if _generate(number, args):
            output = args.output if args.output and args.fig == number else (
                args.output_dir / f"fig{number}.pdf"
            )
            print(f"GENERATED fig{number}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
