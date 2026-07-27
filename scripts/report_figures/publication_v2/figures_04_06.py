#!/usr/bin/env python3
"""Render publication-v2 Figures 4--6 from measured repository evidence.

Figure numbering here follows the final report, not the legacy generator:
Fig.4 predictor evidence, Fig.5 MIXED serving, Fig.6 one BFCL-only OOD workload.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Mapping, NamedTuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / "scripts" / "report_figures"
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

from style import (  # noqa: E402
    IEEE_DOUBLE_WIDTH,
    IEEE_SINGLE_WIDTH,
    OKABE_ITO,
    POLICY_COLOR,
    POLICY_LABEL,
    set_log_axis_plain,
)


OFFLINE_DIR = REPORT_ROOT / "data" / "offline"
RENTAL_DIR = REPORT_ROOT / "data" / "rental-20260719T231309Z"
OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v2"

TIER1_SUMMARY = OFFLINE_DIR / "tier1-matrix-summary.json"
TIER2_SUMMARY = OFFLINE_DIR / "tier2-matrix-summary.json"
TIER2_CURVE = OFFLINE_DIR / "tier2-learning-curve.json"

MIXED_POLICY_DIRS = {
    "stock_fcfs": "stock_fcfs.runs",
    "StockFCFSShim": "StockFCFSShim.runs",
    "PureLTR": "PureLTRScheduler.runs",
    "GatedHybrid": "GatedHybridScheduler.runs",
    "TailSafe": "TailSafeScheduler.runs",
    "LTRAging": "LTRAgingScheduler.runs",
    "PromptLengthSJF": "PromptLengthSJFScheduler.runs",
}
OOD_POLICY_DIRS = {
    "StockFCFSShim": "StockFCFSShim.runs",
    "PureLTR": "PureLTRScheduler.runs",
    "GatedHybrid": "GatedHybridScheduler.runs",
    "TailSafe": "TailSafeScheduler.runs",
}
OOD_SUMMARIES = {
    "StockFCFSShim": "StockFCFSShim.json",
    "PureLTR": "PureLTRScheduler.json",
    "GatedHybrid": "GatedHybridScheduler.json",
    "TailSafe": "TailSafeScheduler.json",
}

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 1234
REPEAT_MARKERS = ("o", "s", "D")


class PredictorGroup(NamedTuple):
    key: str
    label: str
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def low(self) -> float:
        return min(self.values)

    @property
    def high(self) -> float:
        return max(self.values)


class MetricSummary(NamedTuple):
    point: float
    interval: tuple[float, float]
    per_repeat: tuple[float, ...]
    improvement: float | None
    improvement_interval: tuple[float, float] | None


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _save(fig: Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(
        pdf,
        metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(png, dpi=300, metadata={"Software": "Matplotlib"})
    return pdf, png


def load_predictor_evidence(
    tier1_path: Path = TIER1_SUMMARY,
    tier2_path: Path = TIER2_SUMMARY,
    curve_path: Path = TIER2_CURVE,
) -> dict:
    """Load Fig.4 evidence while keeping test and validation metrics separate."""

    tier1 = _read_json(tier1_path)
    tier2 = _read_json(tier2_path)
    curve = _read_json(curve_path)
    if tier1.get("completed_runs") != tier1.get("expected_runs"):
        raise ValueError("Tier-1 matrix is incomplete")
    if tier2.get("completed_runs") != tier2.get("expected_runs"):
        raise ValueError("Tier-2 matrix is incomplete")

    grouped: dict[str, list[float]] = {
        "prompt_only": [],
        "prompt_schema": [],
        "full_context": [],
        "lightgbm": [],
    }
    pattern = re.compile(r"^(?:bert-(prompt_only|prompt_schema|full_context)|lightgbm-structural)-tier2-seed\d+$")
    for run in tier2["runs"]:
        name = str(run["run_name"])
        match = pattern.match(name)
        if match is None:
            raise ValueError(f"unexpected Tier-2 run name: {name}")
        key = match.group(1) or "lightgbm"
        value = run.get("test_tau")
        if value is None:
            raise ValueError(f"missing held-out test tau: {name}")
        grouped[key].append(float(value))

    expected = {"prompt_only": 3, "prompt_schema": 3, "full_context": 3, "lightgbm": 1}
    for key, count in expected.items():
        if len(grouped[key]) != count:
            raise ValueError(f"expected {count} test seeds for {key}, found {len(grouped[key])}")

    groups = (
        PredictorGroup("prompt_only", "BERT\nprompt only", tuple(grouped["prompt_only"])),
        PredictorGroup("prompt_schema", "BERT\nprompt + schema", tuple(grouped["prompt_schema"])),
        PredictorGroup("full_context", "BERT\nfull context", tuple(grouped["full_context"])),
        PredictorGroup("lightgbm", "LightGBM\nstructural", tuple(grouped["lightgbm"])),
    )
    points = curve.get("points", [])
    pool_sizes = tuple(int(point["train_pool_size"]) for point in points)
    effective_examples = tuple(int(point["effective_train_examples"]) for point in points)
    validation_tau = tuple(float(point["validation_tau"]) for point in points)
    if curve.get("seed") != 42 or curve.get("variant") != "full_context":
        raise ValueError("learning curve must be seed 42 full_context")
    if pool_sizes != (500, 1000, 2000, 4000):
        raise ValueError(f"unexpected learning-curve pools: {pool_sizes}")
    if effective_examples != (499, 999, 1997, 3997):
        raise ValueError(f"unexpected effective learning-curve sizes: {effective_examples}")

    return {
        "groups": groups,
        "curve_seed": 42,
        "curve_variant": "full_context",
        "pool_sizes": pool_sizes,
        "effective_examples": effective_examples,
        "validation_tau": validation_tau,
        "tier1_runs_audited_not_plotted": int(tier1["completed_runs"]),
    }


def build_fig4(output_dir: Path = OUTPUT_DIR, *, save: bool = True) -> tuple[Figure, dict]:
    evidence = load_predictor_evidence()
    groups: tuple[PredictorGroup, ...] = evidence["groups"]
    fig, (ax_rank, ax_curve) = plt.subplots(
        2,
        1,
        figsize=(IEEE_SINGLE_WIDTH, 4.85),
        gridspec_kw={"height_ratios": [1.18, 1.0]},
        constrained_layout=True,
    )

    x = np.arange(len(groups))
    colors = [OKABE_ITO["blue"]] * 3 + [OKABE_ITO["orange"]]
    for index, group in enumerate(groups):
        marker = "o" if len(group.values) > 1 else "D"
        ax_rank.errorbar(
            x[index],
            group.mean,
            yerr=np.array([[group.mean - group.low], [group.high - group.mean]]),
            fmt=marker,
            color=colors[index],
            ecolor=colors[index],
            markeredgecolor="white",
            markersize=6.2,
            elinewidth=1.5,
            capsize=3.0,
            zorder=3,
        )
        ax_rank.text(x[index], group.high + 0.009, f"{group.mean:.3f}", ha="center", fontsize=10)

    ax_rank.set_xticks(x, ["Prompt\nonly", "Prompt +\nschema", "Full\ncontext", "LightGBM\nstructural"])
    ax_rank.set_ylim(0.39, 0.70)
    ax_rank.set_ylabel("Held-out test tau-b")
    ax_rank.yaxis.grid(True, zorder=0)
    ax_rank.xaxis.grid(False)
    ax_rank.set_title("Tier-2 test · seed min–max\nBERT n=3; LightGBM n=1", pad=4)
    ax_rank.text(-0.15, 1.02, "(a)", transform=ax_rank.transAxes, fontweight="bold", fontsize=10)

    pools = evidence["pool_sizes"]
    effective_examples = evidence["effective_examples"]
    validation = evidence["validation_tau"]
    curve_x = np.arange(len(pools))
    ax_curve.plot(
        curve_x,
        validation,
        color=OKABE_ITO["blue"],
        marker="o",
        markeredgecolor="white",
        markersize=6.2,
        zorder=3,
    )
    for point_x, tau in zip(curve_x, validation):
        ax_curve.text(point_x, tau + 0.0024, f"{tau:.3f}", ha="center", va="bottom", fontsize=10)
    ax_curve.set_xticks(
        curve_x,
        [f"{pool:g}\n(n={effective})" for pool, effective in zip(pools, effective_examples)],
    )
    ax_curve.set_ylim(0.598, 0.644)
    ax_curve.set_xlabel("Nominal training-pool size")
    ax_curve.set_ylabel("Validation tau-b")
    ax_curve.yaxis.grid(True, zorder=0)
    ax_curve.xaxis.grid(False)
    ax_curve.set_title("Separate learning-curve ablation\nfull context · seed 42", pad=4)
    ax_curve.text(-0.18, 1.02, "(b)", transform=ax_curve.transAxes, fontweight="bold", fontsize=10)
    if save:
        _save(fig, output_dir, "fig4")
    return fig, evidence


def _load_run_records(
    data_dir: Path,
    policy_dirs: Mapping[str, str],
) -> tuple[dict[str, dict[int, dict[str, float]]], int, set[str]]:
    records: dict[str, dict[int, dict[str, float]]] = {}
    error_rows = 0
    categories: set[str] = set()
    for policy, directory in policy_dirs.items():
        run_dir = data_dir / directory
        metadata_files = sorted(run_dir.glob("*.json"))
        if len(metadata_files) != 3:
            raise ValueError(f"expected 3 repeats for {policy}, found {len(metadata_files)}")
        by_repeat: dict[int, dict[str, float]] = {}
        for metadata_path in metadata_files:
            metadata = _read_json(metadata_path)
            repeat = int(metadata["repeat"])
            if repeat in by_repeat:
                raise ValueError(f"duplicate repeat {repeat} for {policy}")
            samples_path = metadata_path.with_suffix(".samples.csv")
            sample_map: dict[str, float] = {}
            with samples_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    categories.add(row["category"])
                    if (row.get("error") or "").strip():
                        error_rows += 1
                        continue
                    request_id = row["request_id"]
                    if request_id in sample_map:
                        raise ValueError(f"duplicate request ID in {samples_path}: {request_id}")
                    sample_map[request_id] = float(row["ttlt_ms"])
            by_repeat[repeat] = sample_map
        if sorted(by_repeat) != [1, 2, 3]:
            raise ValueError(f"expected repeats 1/2/3 for {policy}, found {sorted(by_repeat)}")
        records[policy] = by_repeat
    return records, error_rows, categories


def _balanced_policy_arrays(
    records: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    policies = tuple(records)
    common_ids = set.intersection(
        *(set(records[policy][repeat]) for policy in policies for repeat in (1, 2, 3))
    )
    request_ids = tuple(sorted(common_ids))
    if not request_ids:
        raise ValueError("no request IDs complete across policies and repeats")
    arrays = {
        policy: np.asarray(
            [[records[policy][repeat][request_id] for request_id in request_ids] for repeat in (1, 2, 3)],
            dtype=float,
        )
        for policy in policies
    }
    return request_ids, arrays


def _hierarchical_draws(
    n_requests: int,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bootstrap < 2000:
        raise ValueError("hierarchical interval requires at least 2000 resamples")
    rng = np.random.default_rng(seed)
    repeat_draws = rng.integers(0, 3, size=(n_bootstrap, 3))
    request_draws = rng.integers(0, n_requests, size=(n_bootstrap, n_requests))
    return repeat_draws, request_draws


def _bootstrap_samples(
    values: np.ndarray,
    repeat_draws: np.ndarray,
    request_draws: np.ndarray,
) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("expected repeat-by-request matrix with three repeats")
    sampled = values[repeat_draws[:, :, None], request_draws[:, None, :]]
    return sampled.reshape(sampled.shape[0], -1)


def paired_hierarchical_summaries(
    arrays: Mapping[str, np.ndarray],
    baseline: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, dict[str, MetricSummary]], tuple[np.ndarray, np.ndarray]]:
    """Summarize pooled metrics with paired repeat/request-cluster resampling."""

    if baseline not in arrays:
        raise ValueError(f"missing baseline policy: {baseline}")
    shapes = {values.shape for values in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"policy arrays are not paired: {sorted(shapes)}")
    shape = next(iter(shapes))
    if shape[0] != 3:
        raise ValueError(f"expected three repeats, found shape {shape}")
    repeat_draws, request_draws = _hierarchical_draws(
        shape[1], n_bootstrap=n_bootstrap, seed=seed
    )
    boot_samples = {
        policy: _bootstrap_samples(values, repeat_draws, request_draws)
        for policy, values in arrays.items()
    }
    boot_metrics = {
        policy: {
            "mean": np.mean(samples, axis=1),
            "p99": np.percentile(samples, 99, axis=1),
        }
        for policy, samples in boot_samples.items()
    }
    summaries: dict[str, dict[str, MetricSummary]] = {}
    for policy, values in arrays.items():
        summaries[policy] = {}
        for metric, stat in (
            ("mean", np.mean),
            ("p99", lambda sample: np.percentile(sample, 99)),
        ):
            point = float(stat(values.ravel()))
            per_repeat = tuple(float(stat(values[index])) for index in range(3))
            interval = _percentile_interval(boot_metrics[policy][metric])
            if policy == baseline:
                improvement = 0.0
                improvement_interval = (0.0, 0.0)
            else:
                baseline_boot = boot_metrics[baseline][metric]
                improvement_boot = 100.0 * (baseline_boot - boot_metrics[policy][metric]) / baseline_boot
                baseline_point = float(stat(arrays[baseline].ravel()))
                improvement = 100.0 * (baseline_point - point) / baseline_point
                improvement_interval = _percentile_interval(improvement_boot)
            summaries[policy][metric] = MetricSummary(
                point=point,
                interval=interval,
                per_repeat=per_repeat,
                improvement=improvement,
                improvement_interval=improvement_interval,
            )
    return summaries, (repeat_draws, request_draws)


def _survival_band(
    values: np.ndarray,
    grid_ms: np.ndarray,
    draws: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    repeat_draws, request_draws = draws
    samples = _bootstrap_samples(values, repeat_draws, request_draws)
    curves = np.empty((samples.shape[0], grid_ms.size), dtype=float)
    chunk_size = 64
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        curves[start:stop] = np.mean(
            samples[start:stop, :, None] > grid_ms[None, None, :], axis=1
        )
    low, high = np.percentile(curves, [2.5, 97.5], axis=0)
    return low, high


def load_mixed_evidence(data_dir: Path = RENTAL_DIR / "matrix") -> dict:
    records, error_rows, categories = _load_run_records(data_dir, MIXED_POLICY_DIRS)
    request_ids, arrays = _balanced_policy_arrays(records)
    if error_rows != 0:
        raise ValueError(f"MIXED evidence unexpectedly contains {error_rows} error rows")
    if len(request_ids) != 150:
        raise ValueError(f"expected 150 repeated request IDs, found {len(request_ids)}")
    return {
        "records": records,
        "request_ids": request_ids,
        "arrays": arrays,
        "error_rows": error_rows,
        "categories": tuple(sorted(categories)),
    }


def _policy_tick_label(policy: str) -> str:
    labels = {
        "stock_fcfs": "Stock\nFCFS",
        "StockFCFSShim": "FCFS\nshim",
        "PureLTR": "Pure\nLTR",
        "GatedHybrid": "Gated\nhybrid",
        "TailSafe": "Tail\nsafe",
        "LTRAging": "LTR\naging",
        "PromptLengthSJF": "Prompt SJF\n(non-learned)",
    }
    return labels[policy]


def build_fig5(output_dir: Path = OUTPUT_DIR, *, save: bool = True) -> tuple[Figure, dict]:
    evidence = load_mixed_evidence()
    arrays: dict[str, np.ndarray] = evidence["arrays"]
    policies = tuple(arrays)
    summaries, draws = paired_hierarchical_summaries(arrays, "stock_fcfs")
    fig, (ax_ccdf, ax_stats) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 4.55),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.26, top=0.96, wspace=0.28)

    all_values = np.concatenate([values.ravel() for values in arrays.values()])
    grid_ms = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 72)
    line_styles = {
        "stock_fcfs": (0, (5, 2)),
        "StockFCFSShim": (0, (2, 1)),
    }
    for policy in policies:
        values = np.sort(arrays[policy].ravel())
        survival = (values.size - np.arange(values.size)) / values.size
        color = POLICY_COLOR[policy]
        ax_ccdf.step(
            values / 1000.0,
            survival,
            where="post",
            color=color,
            linestyle=line_styles.get(policy, "-"),
            linewidth=1.6 if policy in ("GatedHybrid", "PromptLengthSJF") else 1.25,
            label=POLICY_LABEL[policy],
            zorder=3,
        )
        low, high = _survival_band(arrays[policy], grid_ms, draws)
        ax_ccdf.fill_between(
            grid_ms / 1000.0,
            np.maximum(low, 0.001),
            np.maximum(high, 0.001),
            color=color,
            alpha=0.055,
            linewidth=0,
            zorder=1,
        )
    stock_p99 = summaries["stock_fcfs"]["p99"].point / 1000.0
    prompt_p99 = summaries["PromptLengthSJF"]["p99"].point / 1000.0
    ax_ccdf.axvline(stock_p99, color=POLICY_COLOR["stock_fcfs"], linestyle="--", linewidth=0.9)
    ax_ccdf.axvline(prompt_p99, color=POLICY_COLOR["PromptLengthSJF"], linestyle=":", linewidth=1.1)
    ax_ccdf.text(stock_p99 + 0.25, 0.014, f"stock p99 {stock_p99:.1f}s", rotation=90, va="bottom", fontsize=10)
    ax_ccdf.text(prompt_p99 - 0.25, 0.004, f"Prompt p99 {prompt_p99:.1f}s", rotation=90, va="bottom", ha="right", fontsize=10)
    ax_ccdf.set_yscale("log")
    ax_ccdf.set_ylim(0.002, 1.05)
    ax_ccdf.set_xlim(left=0)
    set_log_axis_plain(ax_ccdf, "y", [0.01, 0.1, 1.0])
    ax_ccdf.set_xlabel("TTLT (s)")
    ax_ccdf.set_ylabel("CCDF, Pr(TTLT > t)")
    ax_ccdf.yaxis.grid(True, which="major")
    ax_ccdf.xaxis.grid(False)
    ax_ccdf.legend(ncol=2, loc="upper right", columnspacing=0.7, handlelength=1.8)
    ax_ccdf.text(
        0.02,
        0.025,
        "150 request IDs × 3 repeats · 0 errors",
        transform=ax_ccdf.transAxes,
        fontsize=10,
    )
    ax_ccdf.text(-0.13, 1.02, "(a)", transform=ax_ccdf.transAxes, fontweight="bold", fontsize=10)

    x = np.arange(len(policies), dtype=float)
    width = 0.34
    metric_specs = (
        ("mean", -width / 2, 0.58, ""),
        ("p99", width / 2, 1.0, "///"),
    )
    for metric, offset, alpha, hatch in metric_specs:
        points = np.array([summaries[policy][metric].point for policy in policies]) / 1000.0
        intervals = np.array([summaries[policy][metric].interval for policy in policies]) / 1000.0
        errors = np.vstack((points - intervals[:, 0], intervals[:, 1] - points))
        ax_stats.bar(
            x + offset,
            points,
            width=width,
            color=[POLICY_COLOR[policy] for policy in policies],
            alpha=alpha,
            hatch=hatch,
            edgecolor="white" if not hatch else OKABE_ITO["black"],
            zorder=2,
        )
        ax_stats.errorbar(
            x + offset,
            points,
            yerr=errors,
            fmt="none",
            ecolor=OKABE_ITO["black"],
            elinewidth=0.75,
            capsize=2.2,
            capthick=0.75,
            zorder=4,
        )
        for repeat_index, marker in enumerate(REPEAT_MARKERS):
            repeat_points = np.array(
                [summaries[policy][metric].per_repeat[repeat_index] for policy in policies]
            ) / 1000.0
            ax_stats.scatter(
                x + offset + (repeat_index - 1) * 0.035,
                repeat_points,
                marker=marker,
                s=23,
                facecolor="white",
                edgecolor=OKABE_ITO["black"],
                linewidth=0.7,
                zorder=5,
            )
    ax_stats.axvspan(-0.5, 1.5, color=OKABE_ITO["light_gray"], alpha=0.22, zorder=0)
    ax_stats.axvline(1.5, color=OKABE_ITO["dark_gray"], linewidth=0.9, zorder=1)
    ax_stats.set_xticks(x, ["Stock", "Shim", "Pure\nLTR", "Gated", "Tail", "Aging", "SJF*"])
    ax_stats.set_ylabel("TTLT (s)")
    highest = max(summary["p99"].interval[1] for summary in summaries.values()) / 1000.0
    ax_stats.set_ylim(0, highest * 1.32)
    ax_stats.yaxis.grid(True, zorder=0)
    ax_stats.xaxis.grid(False)
    ax_stats.text(0.14, 0.98, "Baselines", transform=ax_stats.transAxes, ha="center", va="top", fontsize=10, fontweight="bold")
    ax_stats.text(0.67, 0.98, "Short-job schedulers", transform=ax_stats.transAxes, ha="center", va="top", fontsize=10, fontweight="bold")
    ax_stats.text(
        0.98,
        0.88,
        "Learned: mean +14.8–15.1%\np99 +8.5–18.2%\n"
        "Prompt SJF (non-learned):\nmean +15.3%; p99 +19.8%",
        transform=ax_stats.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    handles = [
        Patch(facecolor=OKABE_ITO["gray"], alpha=0.58, label="Pooled mean"),
        Patch(facecolor=OKABE_ITO["gray"], hatch="///", edgecolor=OKABE_ITO["black"], label="Pooled p99"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=OKABE_ITO["black"], label="Per-repeat point"),
    ]
    fig.text(
        0.73,
        0.13,
        "Whiskers: paired hierarchical 95% interval · markers: repeats",
        ha="center",
        fontsize=10,
    )
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.73, 0.01), columnspacing=0.8, handlelength=1.1)
    ax_stats.text(-0.10, 1.02, "(b)", transform=ax_stats.transAxes, fontweight="bold", fontsize=10)
    if save:
        _save(fig, output_dir, "fig5")
    evidence = {**evidence, "summaries": summaries}
    return fig, evidence


def load_ood_evidence(data_dir: Path = RENTAL_DIR / "matrix-ood") -> dict:
    records, error_rows, categories = _load_run_records(data_dir, OOD_POLICY_DIRS)
    source_valid = {
        policy: bool(_read_json(data_dir / filename).get("valid"))
        for policy, filename in OOD_SUMMARIES.items()
    }
    common_keys = set.intersection(
        *(
            {(repeat, request_id) for repeat, samples in records[policy].items() for request_id in samples}
            for policy in OOD_POLICY_DIRS
        )
    )
    common_ids_by_repeat = {
        repeat: tuple(sorted(request_id for item_repeat, request_id in common_keys if item_repeat == repeat))
        for repeat in (1, 2, 3)
    }
    if len({ids for ids in common_ids_by_repeat.values()}) != 1:
        raise ValueError("common-complete OOD request IDs differ by repeat; balanced hierarchy unavailable")
    request_ids = common_ids_by_repeat[1]
    arrays = {
        policy: np.asarray(
            [[records[policy][repeat][request_id] for request_id in request_ids] for repeat in (1, 2, 3)],
            dtype=float,
        )
        for policy in OOD_POLICY_DIRS
    }
    analyzed_counts = {
        policy: sum(len(samples) for samples in by_repeat.values())
        for policy, by_repeat in records.items()
    }
    if error_rows != 7:
        raise ValueError(f"expected seven OOD error rows, found {error_rows}")
    if any(source_valid.values()):
        raise ValueError(f"expected all OOD source summaries valid=false: {source_valid}")
    if categories != {"ood:bfcl"}:
        raise ValueError(f"expected one BFCL-only workload category, found {sorted(categories)}")
    if len(common_keys) != 357:
        raise ValueError(f"expected 357 common-complete pairs, found {len(common_keys)}")
    return {
        "records": records,
        "arrays": arrays,
        "common_keys": tuple(sorted(common_keys)),
        "request_ids": request_ids,
        "common_per_repeat": tuple(len(common_ids_by_repeat[repeat]) for repeat in (1, 2, 3)),
        "error_rows": error_rows,
        "source_valid": source_valid,
        "categories": tuple(sorted(categories)),
        "analyzed_counts": analyzed_counts,
    }


def build_fig6(output_dir: Path = OUTPUT_DIR, *, save: bool = True) -> tuple[Figure, dict]:
    evidence = load_ood_evidence()
    arrays: dict[str, np.ndarray] = evidence["arrays"]
    policies = tuple(arrays)
    summaries, _ = paired_hierarchical_summaries(arrays, "StockFCFSShim")
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_WIDTH, 4.15))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.82)
    x = np.arange(len(policies), dtype=float)
    width = 0.34
    for metric, offset, alpha, hatch in (
        ("mean", -width / 2, 0.58, ""),
        ("p99", width / 2, 1.0, "///"),
    ):
        points = np.array([summaries[policy][metric].point for policy in policies]) / 1000.0
        intervals = np.array([summaries[policy][metric].interval for policy in policies]) / 1000.0
        errors = np.vstack((points - intervals[:, 0], intervals[:, 1] - points))
        ax.bar(
            x + offset,
            points,
            width=width,
            color=[POLICY_COLOR[policy] for policy in policies],
            alpha=alpha,
            hatch=hatch,
            edgecolor="white" if not hatch else OKABE_ITO["black"],
            zorder=2,
        )
        ax.errorbar(
            x + offset,
            points,
            yerr=errors,
            fmt="none",
            ecolor=OKABE_ITO["black"],
            elinewidth=0.75,
            capsize=2.2,
            capthick=0.75,
            zorder=4,
        )
        for repeat_index, marker in enumerate(REPEAT_MARKERS):
            repeat_points = np.array(
                [summaries[policy][metric].per_repeat[repeat_index] for policy in policies]
            ) / 1000.0
            ax.scatter(
                x + offset + (repeat_index - 1) * 0.035,
                repeat_points,
                marker=marker,
                s=23,
                facecolor="white",
                edgecolor=OKABE_ITO["black"],
                linewidth=0.7,
                zorder=5,
            )
    ax.axvspan(-0.5, 0.5, color=OKABE_ITO["light_gray"], alpha=0.22, zorder=0)
    ax.axvline(0.5, color=OKABE_ITO["dark_gray"], linewidth=0.9, zorder=1)
    ax.set_xticks(x, ["Shim", "Pure\nLTR", "Gated", "Tail"])
    ax.set_ylabel("TTLT (s)")
    highest = max(summary["p99"].interval[1] for summary in summaries.values()) / 1000.0
    ax.set_ylim(0, highest * 1.43)
    ax.yaxis.grid(True, zorder=0)
    ax.xaxis.grid(False)
    fig.suptitle("One BFCL-only OOD set", fontweight="bold", y=0.98)
    ax.text(
        0.02,
        0.97,
        "Common: 357/policy (119 × 3)\n"
        "7 error rows excluded\nsource valid=false · markers=repeats",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    learned_mean = [summaries[policy]["mean"].improvement for policy in policies[1:]]
    learned_p99 = [summaries[policy]["p99"].improvement for policy in policies[1:]]
    ax.text(
        0.98,
        0.76,
        f"Common-subset gain vs shim\nmean +{min(learned_mean):.1f}–{max(learned_mean):.1f}%\n"
        f"p99 +{min(learned_p99):.1f}–{max(learned_p99):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.895,
        "One workload only; not general OOD robustness",
        transform=fig.transFigure,
        ha="center",
        fontsize=10,
    )
    fig.legend(
        handles=[
            Patch(facecolor=OKABE_ITO["gray"], alpha=0.58, label="Pooled mean"),
            Patch(facecolor=OKABE_ITO["gray"], hatch="///", edgecolor=OKABE_ITO["black"], label="Pooled p99"),
        ],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=0.8,
        handlelength=1.1,
    )
    if save:
        _save(fig, output_dir, "fig6")
    evidence = {**evidence, "summaries": summaries}
    return fig, evidence


def render_all(output_dir: Path = OUTPUT_DIR) -> dict[str, dict]:
    rendered = {}
    for number, builder in ((4, build_fig4), (5, build_fig5), (6, build_fig6)):
        fig, evidence = builder(output_dir)
        plt.close(fig)
        rendered[f"fig{number}"] = evidence
    return rendered


def main() -> None:
    rendered = render_all()
    fig4_groups = rendered["fig4"]["groups"]
    print("Fig.4 test means " + ", ".join(f"{group.key}={group.mean:.6f}" for group in fig4_groups))
    mixed = rendered["fig5"]["summaries"]
    print(
        "Fig.5 PromptLengthSJF improvement "
        f"mean={mixed['PromptLengthSJF']['mean'].improvement:.3f}% "
        f"p99={mixed['PromptLengthSJF']['p99'].improvement:.3f}%"
    )
    ood = rendered["fig6"]
    print(
        f"Fig.6 common_complete={len(ood['common_keys'])} "
        f"errors={ood['error_rows']} valid={ood['source_valid']}"
    )


if __name__ == "__main__":
    main()
