"""Figure 7 - what the Reliability Gate claims versus what it delivers.

Each rule assigns a confidence per stratum; the realized value is the measured
test tau for that stratum. A point above the diagonal claims more reliability
than it delivers - that region is shaded, because overstatement is the failure
mode the gate exists to prevent. Rule C's abstain (0.0) on the strata it cannot
measure is drawn distinctly: it is a refusal to vouch, not a low estimate.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _common import (
    IEEE_DOUBLE_WIDTH,
    OFFLINE,
    OKABE_ITO,
    PROBE_TRACE,
    REPO,
    load_json,
    load_jsonl,
    record_provenance,
    save,
)

T5 = OFFLINE / "t5-gate.json"
TRACE = PROBE_TRACE / "agent_trace_vanilla.jsonl.gz"
GATE_VOCAB = REPO / "scheduler_benchmark" / "artifacts" / "gate_confidence.json"
RULES = (
    ("placeholder_0.9", "Placeholder 0.9", OKABE_ITO["vermillion"], "X"),
    ("global_control_no_stratification", "Global (no strata)", "#8A8A8A", "s"),
    ("C_abstain", "Rule C (shipped)", OKABE_ITO["blue"], "o"),
)
RULE_C = 2  # index into RULES; asserted against the key below


def live_capture_counts() -> tuple[dict[str, int], int]:
    """Per-stratum request counts of the live capture, derived at build time.

    Each captured request is classified S1-S4 against the shipped gate
    vocabulary (same sorted-tool-name fingerprint rule the scheduler uses), so
    the live-traffic note in the figure is computed from the trace artifact
    rather than asserted as prose. Zero-tool requests are counted separately:
    S2-S4 are undefined for rows advertising no tools.
    """
    vocab = load_json(GATE_VOCAB)
    prefix = int(vocab["fingerprint_prefix_length"])
    train_fingerprints = set(vocab["train_fingerprints"])
    train_tools = set(vocab["train_tool_names"])
    counts: dict[str, int] = {}
    zero_tool = 0
    for row in load_jsonl(TRACE):
        tools = row["body"].get("tools") or []
        names = sorted(
            str(tool["function"]["name"])
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        )
        if not names:
            zero_tool += 1
            continue
        payload = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:prefix]
        if fingerprint in train_fingerprints:
            stratum = "S1"
        else:
            seen = sum(1 for name in names if name in train_tools)
            stratum = "S2" if seen == len(names) else "S4" if seen == 0 else "S3"
        counts[stratum] = counts.get(stratum, 0) + 1
    return counts, zero_tool


def _contiguous_label(items: list[str], order: list[str]) -> str:
    indices = [order.index(item) for item in items]
    if len(items) > 2 and indices == list(range(indices[0], indices[-1] + 1)):
        return f"{items[0]}–{items[-1]}"
    return ", ".join(items)


def main() -> None:
    payload = load_json(T5)
    comparison = payload["rule_comparison"]
    if RULES[RULE_C][0] != "C_abstain":
        raise SystemExit("RULE_C index no longer points at C_abstain")
    live_counts, live_zero_tool = live_capture_counts()

    fig, (ax, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, 3.3),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )

    # --- assigned vs realized, per stratum -----------------------------------
    # A realized-tau x-axis puts S3 and S4 0.0075 apart and their markers
    # overlap, so strata are categorical here. Per stratum: the grey bar is what
    # the Ranker actually delivers, the shaded band above it is the region where
    # a rule would be claiming more than that.
    strata = [row["stratum"] for row in comparison["C_abstain"]["per_stratum"]]
    realized = {
        row["stratum"]: float(row["realized"])
        for row in comparison["C_abstain"]["per_stratum"]
    }
    ceiling = 1.0
    positions = np.arange(len(strata))

    for index, stratum in enumerate(strata):
        value = realized[stratum]
        ax.add_patch(
            plt.Rectangle(
                (index - 0.42, value),
                0.84,
                ceiling - value,
                color=OKABE_ITO["vermillion"],
                alpha=0.12,
                zorder=1,
                linewidth=0,
            )
        )
        ax.bar(
            index, value, width=0.84, color=OKABE_ITO["light_gray"], zorder=2
        )
        ax.plot(
            [index - 0.42, index + 0.42],
            [value, value],
            color=OKABE_ITO["dark_gray"],
            linewidth=1.2,
            zorder=3,
        )

    offsets = np.linspace(-0.22, 0.22, len(RULES))
    for offset, (key, label, color, marker) in zip(offsets, RULES):
        for index, row in enumerate(comparison[key]["per_stratum"]):
            assigned = float(row["assigned"])
            is_abstain = assigned == 0.0
            # The grey square rides within ~0.01 of the realized-tau line at
            # S3/S4, and above-vs-below is the judgment the panel asks for: a
            # smaller square with a thicker white halo keeps the line visible
            # through the overlap. Abstain circles sit exactly on y=0, so they
            # are drawn unclipped to avoid the bottom spine halving them.
            ax.scatter(
                index + offset,
                assigned,
                s=52 if is_abstain else (30 if marker == "s" else 38),
                marker=marker,
                facecolor="white" if is_abstain else color,
                edgecolor=color if is_abstain else "white",
                linewidth=1.4 if is_abstain else (1.2 if marker == "s" else 0.6),
                zorder=6,
                clip_on=not is_abstain,
                label=label if index == 0 else None,
            )

    ax.text(
        -0.45,
        0.975,
        "shaded: claimed $>$ delivered",
        fontsize=10,
        color=OKABE_ITO["vermillion"],
        ha="left",
        va="top",
    )
    abstain_indices = [
        index
        for index, row in enumerate(comparison["C_abstain"]["per_stratum"])
        if float(row["assigned"]) == 0.0
    ]
    if abstain_indices:
        anchor = abstain_indices[0]
        ax.annotate(
            "Rule C\nabstains",
            xy=(anchor + offsets[RULE_C], 0.0),
            xytext=(anchor - 0.38, 0.19),
            fontsize=10,
            color=OKABE_ITO["blue"],
            ha="left",
            va="bottom",
            linespacing=1.1,
            arrowprops={
                "arrowstyle": "->",
                "color": OKABE_ITO["blue"],
                "linewidth": 0.8,
            },
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(strata)
    ax.set_xlim(-0.6, len(strata) - 0.4)
    ax.set_ylim(0, ceiling)
    # The stratum meaning moves into the title; the xlabel slot is spent on a
    # coverage row instead (test n, runtime outcome, live-traffic note).
    ax.set_xlabel(" \n \n \n ")  # reserves layout space for the coverage row below
    ax.set_ylabel("Confidence / realized $\\tau_b$")
    ax.set_title("(a) Claimed vs delivered per Cold-Start stratum", loc="left")

    # --- coverage row: test n, runtime outcome, live-traffic marking ---------
    table = {row["stratum"]: row for row in payload["reliability_table"]}
    exercised = [stratum for stratum in strata if live_counts.get(stratum)]
    unexercised = [stratum for stratum in strata if stratum not in exercised]
    xaxis_tf = ax.get_xaxis_transform()
    for index, stratum in enumerate(strata):
        row = table[stratum]
        assigned = float(row["assigned_confidence"])
        is_live = stratum in exercised
        is_abstain = assigned == 0.0
        # Two-line outcome ("trusted" over its value) so neighbouring strata
        # never collide at 10 pt within one categorical slot.
        outcome_lines = ("abstain", "") if is_abstain else (
            "trusted",
            f"{assigned:.2f}".replace("0.", "."),
        )
        ax.text(
            index,
            -0.115,
            f"$n$={row['test_n']}",
            transform=xaxis_tf,
            ha="center",
            va="top",
            fontsize=10,
            color=OKABE_ITO["dark_gray"],
            clip_on=False,
        )
        for line_index, line in enumerate(outcome_lines):
            if not line:
                continue
            ax.text(
                index,
                -0.215 - 0.10 * line_index,
                line,
                transform=xaxis_tf,
                ha="center",
                va="top",
                fontsize=10,
                color=OKABE_ITO["blue"],
                fontweight="bold" if is_live else "normal",
                style="italic" if is_abstain else "normal",
                clip_on=False,
            )
    # Underline the strata the live capture exercised (derived from the trace
    # artifact against the shipped gate vocabulary, not asserted), and say so.
    for stratum in exercised:
        index = strata.index(stratum)
        ax.plot(
            [index - 0.32, index + 0.32],
            [-0.405, -0.405],
            transform=xaxis_tf,
            color=OKABE_ITO["black"],
            linewidth=0.9,
            clip_on=False,
            solid_capstyle="butt",
        )
    live_note = "live capture traffic: " + ", ".join(
        f"{stratum} $n$={live_counts[stratum]}" for stratum in exercised
    )
    live_note += f" ($+$ {live_zero_tool} zero-tool)"
    if unexercised:
        live_note += f"; {_contiguous_label(unexercised, strata)} unexercised"
    ax.text(
        -0.55,
        -0.455,
        live_note,
        transform=xaxis_tf,
        ha="left",
        va="top",
        fontsize=10,
        color=OKABE_ITO["black"],
        clip_on=False,
    )
    # Legend inside the panel: the bars are uniform grey fill whose only
    # information is their top edge, so a solid-framed legend can sit on the
    # S3/S4 bar area without hiding anything a reader needs.
    from matplotlib.patches import Patch

    rule_handles, rule_labels = ax.get_legend_handles_labels()
    rule_handles.append(
        Patch(facecolor=OKABE_ITO["light_gray"], edgecolor=OKABE_ITO["dark_gray"],
              linewidth=1.0, label="delivered $\\tau_b$")
    )
    rule_labels.append("delivered $\\tau_b$")
    ax.legend(rule_handles, rule_labels, title="assigned confidence",
              title_fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.03),
              fontsize=8, frameon=True, framealpha=1.0,
              edgecolor=OKABE_ITO["light_gray"], borderpad=0.5,
              handletextpad=0.5, labelspacing=0.45)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # --- worst overstatement per rule ---------------------------------------
    names, worst, colors = [], [], []
    for key, label, color, _ in RULES:
        names.append(label)
        worst.append(float(comparison[key]["max_overstatement"]))
        colors.append(color)
    bars = ax_bar.barh(np.arange(len(names)), worst, color=colors, height=0.55, zorder=2)
    ax_bar.axvline(0, color=OKABE_ITO["black"], linewidth=0.9, zorder=3)
    for bar, value in zip(bars, worst):
        # Negative bars extend left, so their label goes to the RIGHT of zero;
        # placing it left of the bar collides with the tick labels.
        ax_bar.text(
            value + 0.014 if value >= 0 else 0.014,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    ax_bar.set_yticks(np.arange(len(names)))
    ax_bar.set_yticklabels(names)
    ax_bar.set_xlabel("Worst overstatement\n(assigned $-$ realized $\\tau_b$)")
    # Limits track the data: right leaves room for the printed value beside
    # the widest bar, left keeps the small negative bar visible.
    ax_bar.set_xlim(min(0.0, min(worst)) - 0.08 * max(worst), max(worst) * 1.35)
    ax_bar.set_title("(b) Worst case per rule", loc="left", pad=12)
    ax_bar.xaxis.grid(True)
    ax_bar.set_axisbelow(True)
    # The annotation is a factual claim about the artifact; a rebuild where
    # Rule C overstates must fail the figure build, not keep the caption.
    if not bool(comparison["C_abstain"]["never_overstates"]):
        raise SystemExit(
            "t5-gate.json: rule_comparison.C_abstain.never_overstates is false; "
            "the 'never overstates' annotation no longer holds"
        )
    bar_c = bars[RULE_C]
    ax_bar.annotate(
        "never overstates\n(this split)",
        # Target the bar's top edge, not its midline, so the arrow path
        # clears the value label printed beside the bar.
        xy=(worst[RULE_C] / 2, bar_c.get_y() + bar_c.get_height()),
        xytext=(0.19, RULE_C + 0.30),
        fontsize=10,
        color=OKABE_ITO["blue"],
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["blue"], "linewidth": 0.8},
    )

    save(fig, "gate.pdf")
    record_provenance("gate.pdf", [T5, TRACE, GATE_VOCAB])
    print(
        json.dumps(
            {
                "live_capture": {
                    "per_stratum": live_counts,
                    "zero_tool": live_zero_tool,
                },
                **{
                key: {
                    "max_overstatement": float(comparison[key]["max_overstatement"]),
                    "never_overstates": bool(comparison[key]["never_overstates"]),
                    "per_stratum": {
                        row["stratum"]: {
                            "assigned": float(row["assigned"]),
                            "realized": float(row["realized"]),
                        }
                        for row in comparison[key]["per_stratum"]
                    },
                }
                for key, _, _, _ in RULES
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
