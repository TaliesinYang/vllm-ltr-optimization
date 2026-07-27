"""Figure 3 - workload characterization of real agent traffic.

(a) how much of a request payload is tool schema, per deployment config
(b) what fraction of requests carry no tools at all
(c) how long the completions actually are

Every value is computed from the raw probe captures at build time. The schema
share is measured as serialized tool-array bytes over serialized request-body
bytes, per request, then averaged - the method is stated on the panel because a
different byte accounting gives a different number.
"""

from __future__ import annotations

import json
import statistics

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    COLOR,
    FIGS,
    IEEE_DOUBLE_WIDTH,
    OKABE_ITO,
    PROBE_SCHEMA,
    PROBE_TRACE,
    load_jsonl,
    record_provenance,
    save,
)

CAPTURE = PROBE_SCHEMA / "captured_requests_v2.jsonl"
TRACE = PROBE_TRACE / "agent_trace_vanilla.jsonl.gz"
FULL_CONFIG_MIN_TOOLS = 100  # full config ships 170 tools, vanilla 10


def schema_share(body: dict) -> float:
    tools = body.get("tools") or []
    tool_bytes = len(json.dumps(tools, ensure_ascii=False).encode("utf-8"))
    body_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    return tool_bytes / body_bytes if body_bytes else 0.0


def main() -> None:
    capture = load_jsonl(CAPTURE)
    trace = load_jsonl(TRACE)

    full, vanilla = [], []
    for row in capture:
        body = row["body"]
        tools = body.get("tools") or []
        if not tools:
            continue
        (full if len(tools) >= FULL_CONFIG_MIN_TOOLS else vanilla).append(
            schema_share(body)
        )
    tool_counts = {
        "full": max(len(r["body"].get("tools") or []) for r in capture),
        "vanilla": max(
            (
                len(r["body"].get("tools") or [])
                for r in capture
                if 0 < len(r["body"].get("tools") or []) < FULL_CONFIG_MIN_TOOLS
            ),
            default=0,
        ),
    }

    zero_tool = sum(1 for row in trace if not (row["body"].get("tools") or []))
    completions = [
        int(row["usage"]["completion_tokens"])
        for row in trace
        if isinstance(row.get("usage", {}).get("completion_tokens"), int)
        and row["usage"]["completion_tokens"] > 0
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(IEEE_DOUBLE_WIDTH, 2.35), constrained_layout=True
    )

    # (a) payload composition -------------------------------------------------
    ax = axes[0]
    shares = [statistics.fmean(full) * 100, statistics.fmean(vanilla) * 100]
    names = [
        f"full\n({tool_counts['full']} tools)",
        f"vanilla\n({tool_counts['vanilla']} tools)",
    ]
    bars = ax.bar(
        names, shares, color=[OKABE_ITO["vermillion"], OKABE_ITO["blue"]], width=0.55
    )
    for bar, value in zip(bars, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Tool schema share of\nrequest payload (%)")
    ax.set_title("(a) Payload composition", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # (b) request-type mix ----------------------------------------------------
    ax = axes[1]
    tool_bearing = len(trace) - zero_tool
    fractions = [tool_bearing / len(trace) * 100, zero_tool / len(trace) * 100]
    left = 0.0
    for value, color, name in zip(
        fractions,
        [OKABE_ITO["blue"], OKABE_ITO["gray"]],
        ["tool-bearing", "zero-tool"],
    ):
        ax.barh([0], [value], left=left, color=color, height=0.5, label=name)
        ax.text(
            left + value / 2,
            0,
            f"{value:.0f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if name == "tool-bearing" else "black",
        )
        left += value
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel(f"Share of {len(trace)} live requests (%)")
    ax.set_title("(b) Request-type mix", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2)
    ax.spines["left"].set_visible(False)

    # (c) completion-length distribution --------------------------------------
    ax = axes[2]
    ordered = np.sort(np.asarray(completions, dtype=float))
    ecdf = np.arange(1, ordered.size + 1) / ordered.size
    ax.step(ordered, ecdf, where="post", color=OKABE_ITO["blue"])
    for fraction, style in ((0.50, ":"), (0.99, "--")):
        index = max(0, int(np.ceil(fraction * ordered.size)) - 1)
        value = ordered[index]
        ax.axvline(value, color=OKABE_ITO["dark_gray"], linestyle=style, linewidth=0.9)
        ax.text(
            value,
            0.06 if fraction == 0.50 else 0.30,
            f" p{int(fraction * 100)}={value:.0f}",
            fontsize=10,
            color=OKABE_ITO["dark_gray"],
            ha="left",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Completion length (tokens)")
    ax.set_ylabel(f"ECDF ({ordered.size} requests)")
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) Completion length", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    save(fig, "fig3.pdf")
    record_provenance("fig3.pdf", [CAPTURE, TRACE])
    print(
        json.dumps(
            {
                "schema_share_full_pct": round(shares[0], 2),
                "schema_share_vanilla_pct": round(shares[1], 2),
                "schema_share_method": "mean over requests of "
                "len(json(tools)) / len(json(body)), tool-bearing requests only",
                "zero_tool_requests": zero_tool,
                "total_trace_requests": len(trace),
                "zero_tool_pct": round(zero_tool / len(trace) * 100, 2),
                "completion_p50": float(
                    ordered[max(0, int(np.ceil(0.50 * ordered.size)) - 1)]
                ),
                "completion_p99": float(
                    ordered[max(0, int(np.ceil(0.99 * ordered.size)) - 1)]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
