"""Figure 7 - what the Reliability Gate claims versus what it delivers.

Vocabulary is fixed for the whole figure: a rule *claims* a number, the held-out
test split *delivers* one, and the *gap* is claimed minus delivered. That single
definition of "gap" is used in exactly one place - panel (b) - so no two marks
in the figure can print two different distances under the same word.

(a) is the shipped rule (C_abstain) per stratum on the raw tau_b axis. The
delivered 95% CI is drawn as a band spanning the whole row, so the claim mark
sitting inside or outside that band *is* the containment test; no reader has to
compare two free-floating marks by eye. Where the rule abstains there is no
claim, so nothing is plotted on the quantitative axis; a refusal to vouch is
not the number zero.

(b) compares three rules at each one's largest claim gap - the largest
|claimed - delivered| over the strata where that rule actually makes a claim.
Rule C claims on two of the four strata and the baselines on all four, so every
row states how many strata its worst case is a maximum over: the comparison
sets are unequal and the panel says so rather than hiding it behind the word
"largest".

Every number the panels assert is printed once, in the row-label column on the
left of its own panel. Nothing numeric is set loose inside the data area except
the one bound of (b) that two decimals would round away, so no value label can
amputate a gridline and no two labels can drift into different placement
conventions.

The plate is held to the 2.8 in double-column cap. What used to buy its height
was prose: a four-line method footnote under the panels, a five-line gloss
defining S1-S4 in the corner of (a), a three-line gloss describing the three
gate rules in (b), and a claim value repeated in three places. The footnote and
the rule descriptions moved into the LaTeX caption, the stratum definitions
moved onto the rows they define, and the repeated values were cut back to one
printing each. Both caption sentences are still assembled here from the same
artifacts and printed with the build output, so neither becomes a typed literal
that can drift away from the file it describes.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
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
from style import EXION  # noqa: E402  (_common puts the style module on sys.path)

T5 = OFFLINE / "t5-gate.json"
TRACE = PROBE_TRACE / "agent_trace_vanilla.jsonl.gz"
GATE_VOCAB = REPO / "scheduler_benchmark" / "artifacts" / "gate_confidence.json"

# EXION palette. One dark family tone, one ordered structure grey ramp, one
# vermillion. The family tone means exactly one thing in this figure: a value the
# shipped gated rule claims, and it sits at the dark end of the ramp because that
# rule owns the whole confidence decision. Structure grey means exactly one
# thing: an estimate with its 95% CI, plus every frame and rule of the figure.
# Vermillion is reserved for overstatement and is spent only where a rule's whole
# gap interval sits above zero. Every one of the three has its own legend entry,
# so no ink carries an undeclared second meaning.
CLAIM_COLOR = EXION["family"][3]
GRAY_POINT = EXION["structure"][3]
GRAY_BAND = EXION["structure"][2]
GRAY_ZERO = EXION["structure"][3]
GRAY_NOTE = EXION["structure"][3]
STRIP_FILL = EXION["structure"][0]
GRID = EXION["structure"][0]
FRAME = EXION["structure"][3]
ORANGE = OKABE_ITO["vermillion"]
# The overstatement band is the same vermillion at low opacity, never a second
# warm hex: the figure holds exactly one warm colour and it means failure mode.
ORANGE_BAND_ALPHA = 0.30
# The legend swatch must be the pixels the panel actually draws, so it is that
# same vermillion composited over the white page rather than a sampled tint.
ORANGE_BAND = to_hex(
    tuple(
        1.0 - ORANGE_BAND_ALPHA + ORANGE_BAND_ALPHA * channel
        for channel in to_rgb(ORANGE)
    )
)

MINUS = "−"  # U+2212, so a minus matches the width and height of a plus

# The one piece of loose text left inside a data area: the bound that decides
# whether "entirely above zero" is a comfortable statement or a hair's breadth.
VALUE_BBOX = dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none")
# House frame for the vocabulary glosses and the finding callouts.
CALLOUT_BBOX = dict(
    boxstyle="square,pad=0.25", facecolor="white", edgecolor=FRAME, linewidth=0.6
)

# What each stratum name means, as a phrase short enough to sit on the row it
# names. This used to be a five-line framed gloss in the bottom-left corner of
# (a) - a whole row slot of plate height spent restating four row names a few
# millimetres from the rows themselves. Definitions belong on the thing they
# define, so each row now carries its own.
STRATUM_GLOSS = {
    "S1": "seen in training",
    "S2": "new set, all known",
    "S3": "known + new mixed",
    "S4": "none seen before",
}

RULES = (
    ("placeholder_0.9", "Placeholder"),
    ("global_control_no_stratification", "Global"),
    ("C_abstain", "Rule C (shipped)"),
)
RULE_C = 2  # index into RULES; asserted against the key below

# Both panels use the same four row slots and the same limits, so a row in (a)
# and a row in (b) are the same height and the same distance apart: one scan
# rhythm across the pair. (a) uses all four slots, one per stratum; (b) has
# three rules to compare, so its fourth slot is deliberately left empty rather
# than stretching three rows over a panel sized for four.
FIG_HEIGHT = 2.85  # double-column plate cap is 2.8 in incl. the tight-bbox pad
Y_TOP, Y_BOTTOM = -0.62, 3.62
LANE = 0.115  # half the distance between the claim lane and the delivered lane
BAND_H = 0.38  # row band height, tall enough to hold both lanes of (a)

# Symmetric padding about the outer tick of each panel, so neither axis looks
# nudged to one side. Ticks: (a) every 0.10 from 0.30, (b) every 0.20 from -0.20.
TICK_A, PAD_A = 0.1, 0.02
TICK_B, PAD_B = 0.2, 0.03


def fmt2(value: float, signed: bool = False) -> str:
    """Every tau_b-scale number in this figure: leading zero, two decimals.

    Signed values carry a real U+2212 rather than a hyphen, so the minus and
    the plus in panel (b) have the same weight and height.
    """
    body = f"{abs(value):.2f}"
    if signed:
        return ("+" if value >= 0 else MINUS) + body
    return (MINUS if value < 0 else "") + body


def fmt3(value: float) -> str:
    """Three decimals, signed: used only for a bound that two decimals would
    round to +0.00, hiding whether the interval actually clears zero."""
    return ("+" if value >= 0 else MINUS) + f"{abs(value):.3f}"


def live_capture_counts() -> tuple[dict[str, int], int]:
    """Per-stratum request counts of the live capture, derived at build time.

    Each captured request is classified S1-S4 against the shipped gate
    vocabulary (same sorted-tool-name fingerprint rule the scheduler uses), so
    the live-traffic note in the figure is computed from the trace artifact
    rather than asserted as prose. Requests advertising no tools are counted
    separately: the strata are defined on a tool set, so none applies to them.
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


def _header_strip(ax, text: str) -> None:
    """Framed light-gray title band spanning the panel top, bold text at left.

    The text is a label for what the panel plots, never a finding: at most four
    words, no numbers. Findings live in the framed callouts inside the panels,
    where the marks that support them are.
    """
    patch = ax.add_patch(
        plt.Rectangle(
            (0.0, 1.04),
            1.0,
            0.095,
            transform=ax.transAxes,
            facecolor=STRIP_FILL,
            edgecolor=FRAME,
            linewidth=0.6,
            clip_on=False,
            zorder=5,
        )
    )
    label = ax.text(
        0.015,
        1.0875,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=6,
    )
    return patch, label


def _gloss(ax, text: str, x: float = 0.028, y: float = 0.05, va: str = "bottom"):
    """Framed box in axes coordinates: vocabulary gloss or finding callout.

    Anchoring to the axes rather than to a data coordinate keeps it edge-aligned
    instead of floating in whatever free region the current numbers happen to
    leave. Each one sits in dead space its panel would otherwise waste.
    """
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va=va,
        linespacing=1.3,
        zorder=7,
        bbox=CALLOUT_BBOX,
    )


def _contiguous_label(items: list[str], order: list[str]) -> str:
    indices = [order.index(item) for item in items]
    if len(items) > 2 and indices == list(range(indices[0], indices[-1] + 1)):
        return f"{items[0]}–{items[-1]}"
    return ", ".join(items)


def _worst_claim(comparison: dict, key: str, test_realized: dict) -> dict:
    """The rule's largest claim gap, over the strata where it makes a claim.

    "Largest" is |claimed - delivered|, applied identically to every rule: a
    rule that only ever understates is still shown at the stratum where its
    claim is furthest from what the split delivered, never at its most
    flattering one. Abstentions are excluded because there is no claim to
    measure a gap from - not because excluding them helps the shipped rule, and
    the count of claiming strata is carried into the row label so the reader
    sees that the three maxima are taken over unequal sets.

    The claim is a fixed assigned number, so the uncertainty in
    claimed - delivered is exactly the uncertainty in delivered, mirrored.
    """
    all_rows = comparison[key]["per_stratum"]
    # Artifact integrity: the summary field must still agree with the rows.
    largest_over = max(float(row["overstates_by"]) for row in all_rows)
    if abs(largest_over - float(comparison[key]["max_overstatement"])) > 1e-9:
        raise SystemExit(f"{key}: max_overstatement disagrees with per_stratum rows")
    claiming = [row for row in all_rows if float(row["assigned"]) > 0.0]
    if not claiming:
        raise SystemExit(f"{key}: no stratum carries a claim")
    worst = max(claiming, key=lambda row: abs(float(row["overstates_by"])))
    point = float(worst["overstates_by"])
    claimed = float(worst["assigned"])
    ci_low, ci_high = (float(value) for value in test_realized[worst["stratum"]]["ci95_seed17"])
    return {
        "stratum": str(worst["stratum"]),
        "point": point,
        "low": claimed - ci_high,
        "high": claimed - ci_low,
        "claimed": claimed,
        "n_claiming": len(claiming),
        "n_strata": len(all_rows),
    }


def _constant_claim(comparison: dict, key: str) -> float:
    """The single value a stratum-blind rule assigns everywhere.

    Panel (b)'s row labels no longer print the claim behind each worst case, so
    the two constants are named once in the panel's gloss instead. They are
    read back out of the artifact here rather than written as literals, and a
    rule that stopped being constant fails the build instead of being
    summarised by a number it no longer assigns.
    """
    assigned = {float(row["assigned"]) for row in comparison[key]["per_stratum"]}
    if len(assigned) != 1:
        raise SystemExit(f"{key} is not a constant rule: it assigns {sorted(assigned)}")
    return assigned.pop()


def _symmetric_limits(lo_data: float, hi_data: float, step: float, pad: float):
    """Limits that clear the data and sit an equal distance outside the outermost
    tick on each side, so the axis is not visibly heavier at one end."""
    first = np.floor(lo_data / step) * step
    last = np.ceil(hi_data / step) * step
    return first - pad, last + pad


def main() -> None:
    payload = load_json(T5)
    comparison = payload["rule_comparison"]
    if RULES[RULE_C][0] != "C_abstain":
        raise SystemExit("RULE_C index no longer points at C_abstain")
    test_realized = payload["test_realized"]
    table = {row["stratum"]: row for row in payload["reliability_table"]}
    live_counts, live_zero_tool = live_capture_counts()

    fig, (ax, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_WIDTH, FIG_HEIGHT),
        # Not 1:1. (a) now carries what each stratum name means in its row
        # labels, which widens its label column and, at equal ratios, takes the
        # width out of (b)'s data area - where the tightest-bound label has to
        # fit inside one tick lane. (b) gets the difference back.
        gridspec_kw={"width_ratios": [1.0, 1.12]},
        constrained_layout=True,
    )
    # Both panels carry a multi-line row-label column; without an explicit
    # gutter the panels are packed until (b)'s labels nearly touch (a)'s axis.
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.005)

    # --- (a) the shipped rule, per stratum -----------------------------------
    # Strata are rows, tau_b is the single quantitative axis, and each row holds
    # the delivered CI as a band plus two marks: what Rule C claims and what the
    # split delivered. The stratum-blind baselines are constants and would add
    # three identical marks per row for one number each, so they are compared in
    # (b) instead.
    rows_c = comparison["C_abstain"]["per_stratum"]
    strata = [row["stratum"] for row in rows_c]
    claims = [float(row["assigned"]) for row in rows_c]
    bounds = {
        stratum: tuple(float(value) for value in test_realized[stratum]["ci95_seed17"])
        for stratum in strata
    }

    # Limits are set by the data actually drawn, not by the [0, 1] range of the
    # coefficient: an axis padded out to 1.00 spends a third of the panel on
    # emptiness.
    lo_data = min([low for low, _ in bounds.values()] + [c for c in claims if c > 0.0])
    hi_data = max([high for _, high in bounds.values()] + claims)
    xmin, xmax = _symmetric_limits(lo_data, hi_data, TICK_A, PAD_A)

    labels_a = []
    for index, stratum in enumerate(strata):
        low, high = bounds[stratum]
        point = float(test_realized[stratum]["mean_tau_b"])
        claimed = claims[index]
        # The band is the containment device: it is the delivered 95% CI and it
        # spans the whole row, so a claim mark inside it is a claim inside the
        # interval and a claim beside it is a claim outside, with no drop line
        # or eyeball comparison needed.
        ax.barh(
            index,
            high - low,
            left=low,
            height=BAND_H,
            color=GRAY_BAND,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            point,
            index + LANE,
            marker="o",
            markersize=4.5,
            color=GRAY_POINT,
            linestyle="none",
            zorder=5,
        )

        # C_abstain assigns 0 to mean "no claim". Plotting that as a point on a
        # tau_b axis would assert a confidence of zero, so the abstaining rows
        # simply carry no claim mark; the row label says so in words.
        if claimed == 0.0:
            claim_line = "abstains"
        else:
            # (a)'s callout asserts this for every claiming row; a rebuild where
            # it stops holding must fail the build, not keep the statement.
            if claimed > high:
                raise SystemExit(
                    f"(a) headline broken: the Rule C claim on {stratum} "
                    f"({claimed:.4f}) exceeds its delivered CI upper bound ({high:.4f})"
                )
            claim_line = f"claim {fmt2(claimed)}"
            ax.plot(
                claimed,
                index - LANE,
                marker="D",
                markersize=5.0,
                color=CLAIM_COLOR,
                linestyle="none",
                zorder=5,
            )
        # Two lines, not three: the delivered value used to be printed here as
        # well, and it is already the grey mark of this row with its CI band
        # around it. A row that draws a number and also spells it out spends a
        # line of plate height on saying the same thing twice. The line it frees
        # carries what the stratum name means, which used to be a framed gloss.
        if stratum not in STRATUM_GLOSS:
            raise SystemExit(f"no row gloss for stratum {stratum}")
        labels_a.append(
            f"{stratum}  {STRATUM_GLOSS[stratum]}\n"
            f"test $n$={table[stratum]['test_n']}, {claim_line}"
        )

    abstained = [row["stratum"] for row in rows_c if float(row["assigned"]) == 0.0]
    if len(abstained) != 2:
        raise SystemExit(
            "(a) assumes exactly two abstaining strata; "
            f"the artifact now has {len(abstained)}"
        )
    # The finding the header used to carry, moved to where its marks are: it is
    # the containment statement the loop above just enforced row by row.
    callout_a = _gloss(
        ax,
        "Claims sit inside\nthe delivered CI",
        x=0.028,
        y=0.955,
        va="top",
    )
    ax.set_yticks(np.arange(len(strata)))
    ax.set_yticklabels(labels_a, fontsize=9)
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(TICK_A))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: fmt2(value)))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("$\\tau_b$: claimed and delivered", fontsize=9, labelpad=2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, alpha=1.0)
    ax.set_axisbelow(True)
    strip_a, header_a = _header_strip(ax, "(a) Claimed vs delivered")
    ax.set_title(" ", pad=16)  # reserves room for the strip under constrained layout

    # --- (b) largest claim gap per rule --------------------------------------
    # Point and band, the same encoding as (a): a zero-anchored bar would give
    # the estimate an area the interval does not support, and a whisker cap on
    # the smallest bound would print a glyph across the zero reference.
    labels_b, points, lows, highs = [], [], [], []
    worst_cases: dict[str, dict] = {}
    for key, base_label in RULES:
        case = _worst_claim(comparison, key, test_realized)
        worst_cases[key] = case
        # The three worst cases are maxima over different numbers of strata,
        # because Rule C abstains on two of them. The row label carries that
        # count, so the comparison is never read as like-for-like.
        # Three lines, not four: the claimed value that produced the worst gap
        # used to have a line of its own here. Both stratum-blind rules claim
        # one constant, which the panel's own gloss now names, and Rule C's
        # worst-stratum claim is a mark and a printed number in (a), so the
        # line was a third copy of a number the plate already carries.
        labels_b.append(
            f"{base_label}\n"
            f"claims {case['n_claiming']} of {case['n_strata']} strata\n"
            f"worst {case['stratum']}, gap {fmt2(case['point'], signed=True)}"
        )
        points.append(case["point"])
        lows.append(case["low"])
        highs.append(case["high"])

    positions = np.arange(len(RULES))
    xmin_b, xmax_b = _symmetric_limits(min(lows), max(highs), TICK_B, PAD_B)

    # The annotation is a factual claim about the artifact; a rebuild where
    # Rule C overstates must fail the figure build, not keep the caption.
    if not bool(comparison["C_abstain"]["never_overstates"]):
        raise SystemExit(
            "t5-gate.json: rule_comparison.C_abstain.never_overstates is false; "
            "the 'never overstates' annotation no longer holds"
        )
    overstating = [i for i, low in enumerate(lows) if low > 0.0]
    below_zero = [i for i, high in enumerate(highs) if high < 0.0]
    # The callout of (b) asserts exactly this split.
    if overstating != [i for i in range(len(RULES)) if i != RULE_C] or below_zero != [RULE_C]:
        raise SystemExit(
            "panel (b) headline no longer matches the intervals: entirely-above-zero "
            f"rows {overstating}, entirely-below-zero rows {below_zero}"
        )

    # The zero reference spans the three rule rows and stops above the gloss, so
    # nothing opaque is ever laid across the one line the panel is read against.
    zero_span = (-0.55, len(RULES) - 1 + 0.28)
    zero_line = ax_bar.vlines(
        0.0,
        zero_span[0],
        zero_span[1],
        color=GRAY_ZERO,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        zorder=3,
    )
    for index, (point, low, high) in enumerate(zip(points, lows, highs)):
        # Vermillion is the reserved overstatement colour and is spent here: the
        # two rules whose whole interval sits above zero are drawn in it, so the
        # finding is carried by the mark and not only by a printed sign.
        overstates = low > 0.0
        color = ORANGE if overstates else GRAY_POINT
        ax_bar.barh(
            index,
            high - low,
            left=low,
            height=BAND_H,
            color=ORANGE if overstates else GRAY_BAND,
            alpha=ORANGE_BAND_ALPHA if overstates else 1.0,
            linewidth=0,
            zorder=2,
        )
        ax_bar.plot(
            point,
            index,
            marker="o",
            markersize=4.5,
            color=color,
            linestyle="none",
            zorder=5,
        )

    # The narrowest margin among the overstating rules decides whether "entirely
    # above zero" is a comfortable statement or a hair's breadth. That bound is
    # three thousandths wide and cannot be read off the axis, so it is printed
    # at the end of the interval it belongs to, at three decimals, because two
    # would round it to +0.00.
    tightest = min(overstating, key=lambda i: lows[i])
    span_b = xmax_b - xmin_b
    # The label lives in the lane between the zero reference and the first
    # gridline to its right, and it is centred in that lane rather than set at a
    # hand-tuned offset from zero: the lane is ~52 px wide and the label is ~41,
    # so the two clearances the guards below check - 4 px off the zero reference,
    # 3 px off the gridline - are only both satisfied near the middle. Centring
    # lets the tick geometry place it instead of a constant tuned to one metric.
    next_gridline = min(
        tick for tick in ax_bar.xaxis.get_major_locator().tick_values(xmin_b, xmax_b)
        if tick > 0.0 and tick <= xmax_b
    )
    note_x = 0.5 * next_gridline
    # Leader and label are separate artists on purpose: the label alone is what
    # must be held clear of the zero reference and of every gridline, and an
    # annotation's bounding box would fold the leader into that test.
    ax_bar.plot(
        [lows[tightest], note_x - 0.005 * span_b],
        [tightest - 0.5 * BAND_H, tightest - 0.50],
        linewidth=0.6,
        color=FRAME,
        solid_capstyle="butt",
        zorder=4,
    )
    bound_note = ax_bar.text(
        note_x,
        tightest - 0.58,
        fmt3(lows[tightest]),
        fontsize=8,
        color=GRAY_NOTE,
        ha="center",
        va="center",
        zorder=7,
        bbox=VALUE_BBOX,
    )

    # What the three rules do used to be a three-line framed box in the empty
    # fourth row slot. Three lines of prose inside the artwork is the failure
    # mode the figure standard names first, and a description of a rule is
    # caption material: the panel identifies which row is which rule in its row
    # labels, so nothing the panel asserts depended on the box. The two
    # constants it named are still read out of the artifact here, and printed
    # with the build output, so the caption sentence that replaces it stays
    # checkable against the same file rather than becoming a typed literal.
    placeholder_claim = _constant_claim(comparison, "placeholder_0.9")
    global_claim = _constant_claim(comparison, "global_control_no_stratification")
    caption_rules = (
        f"Placeholder assigns {fmt2(placeholder_claim)} to every request, Global "
        f"{fmt2(global_claim)} to every request, and Rule C a per-stratum "
        "confidence, abstaining below 100 validation rows."
    )

    # The finding the header used to carry, moved onto the row it is about and
    # into the empty half of that row, so it reads against the mark it describes.
    callout_b = _gloss(
        ax_bar,
        "Only Rule C\nstays below 0",
        x=(0.055 - xmin_b) / span_b,
        y=(Y_BOTTOM - RULE_C) / (Y_BOTTOM - Y_TOP),
        va="center",
    )

    ax_bar.set_yticks(positions)
    ax_bar.set_yticklabels(labels_b, fontsize=8)
    ax_bar.set_ylim(Y_BOTTOM, Y_TOP)
    ax_bar.set_xlim(xmin_b, xmax_b)
    # Same tick format as (a) - a signed tick label is wide enough that the ticks
    # of this panel run into one another. The sign that matters is on the
    # estimates and on the zero reference, not on the ruler.
    ax_bar.xaxis.set_major_locator(mpl.ticker.MultipleLocator(TICK_B))
    ax_bar.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: fmt2(value)))
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.set_xlabel(
        "gap = claimed " + MINUS + " delivered $\\tau_b$", fontsize=9, labelpad=2
    )
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.xaxis.grid(True, color=GRID, alpha=1.0)
    ax_bar.set_axisbelow(True)
    strip_b, header_b = _header_strip(ax_bar, "(b) Worst-case gap")
    ax_bar.set_title(" ", pad=16)

    handles = [
        Line2D([], [], color=GRAY_BAND, linewidth=5.5, marker="o", markersize=4.2,
               markerfacecolor=GRAY_POINT, markeredgecolor=GRAY_POINT,
               label="estimate, 95% CI"),
        Line2D([], [], linestyle="none", marker="D", markersize=4.6,
               color=CLAIM_COLOR, label="Rule C claim"),
        Line2D([], [], color=ORANGE_BAND, linewidth=5.5, marker="o", markersize=4.2,
               markerfacecolor=ORANGE, markeredgecolor=ORANGE,
               label="interval entirely > 0: overstates"),
        Line2D([], [], color=GRAY_ZERO, linewidth=0.9, linestyle=(0, (4, 2)),
               label="gap = 0"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        fontsize=8,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=FRAME,
        fancybox=False,  # square frame, same as the header strips
        borderpad=0.35,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    legend.get_frame().set_linewidth(0.6)

    # --- method and live-traffic provenance, no longer drawn ------------------
    # This used to be a four-line footnote strip under the panels: where the
    # delivered numbers come from, and what the live capture is and is not. It
    # was 0.49 in of plate height - a fifth of the whole figure - spent on prose
    # that reads the same in the caption, so it moved into the LaTeX caption.
    # The wording is still assembled here, from the same artifacts, and printed
    # with the rest of the build output, so the sentence in the .tex stays
    # checkable against the trace rather than becoming a hand-typed literal.
    exercised = [stratum for stratum in strata if live_counts.get(stratum)]
    if not exercised:
        raise SystemExit("live trace classified no request into any stratum")
    reach = ", ".join(f"{stratum} ({live_counts[stratum]} requests)" for stratum in exercised)
    unexercised = [stratum for stratum in strata if stratum not in exercised]
    caption_note = (
        "Delivered $\\tau_b$ is an offline replay of the held-out test split "
        "(3-seed mean; band = 95\\% session-clustered bootstrap CI, seed 17, "
        "mirrored onto the gap in (b)); the live gateway capture is a separate "
        "reachability check and fed no number here, reaching "
        f"{reach}{' only' if unexercised else ''}, with {live_zero_tool} further "
        "requests advertising no tools so that no stratum applies to them."
    )

    # Constrained layout centres an outside legend on the whole canvas, which
    # includes the y-label gutter. Settle the layout once, freeze it, then
    # align the legend to the axes block it belongs to.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    box_a, box_b = ax.get_position(), ax_bar.get_position()
    block_center = 0.5 * (box_a.x0 + box_b.x1)
    legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
    legend.set_bbox_to_anchor(
        (block_center - 0.5 * legend_box.width, legend_box.y0,
         legend_box.width, legend_box.height)
    )

    # The legend frame and the two header strips are separate levels of the
    # hierarchy and must not read as one block. Every other framed clearance in
    # this figure is ~0.1 in; hold the legend to the same.
    fig.canvas.draw()
    strip_top = max(strip.get_window_extent().y1 for strip in (strip_a, strip_b))
    target = 0.13 * fig.dpi
    deficit = target - (legend.get_window_extent().y0 - strip_top)
    if deficit > 0:
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor(
            (anchor.x0, anchor.y0 + deficit / (fig.get_figheight() * fig.dpi),
             anchor.width, anchor.height)
        )
        fig.canvas.draw()
    clearance = legend.get_window_extent().y0 - strip_top
    if clearance < target - 1.0:
        raise SystemExit(
            f"legend frame clears the header strips by only {clearance:.0f} px; "
            f"{target:.0f} px required"
        )

    # The figure is saved with a tight bounding box, so any text wider than the
    # panel block silently widens the PDF past the two-column measure. Fail the
    # build instead: the fix is shorter wording, never a smaller font.
    block = (box_a.x0 * fig.get_figwidth() * fig.dpi,
             box_b.x1 * fig.get_figwidth() * fig.dpi)
    for artist, name in ((legend, "legend"),):
        extent = artist.get_window_extent()
        if extent.x0 < block[0] - 1.0 or extent.x1 > block[1] + 1.0:
            raise SystemExit(
                f"{name} ({extent.x0:.0f}-{extent.x1:.0f} px) runs outside the "
                f"panel block ({block[0]:.0f}-{block[1]:.0f} px); shorten it"
            )

    # Header strips, glosses and callouts must fit their own panel at full size.
    for artist, host, name in (
        (header_a, ax, "(a) header"),
        (header_b, ax_bar, "(b) header"),
        (callout_a, ax, "(a) callout"),
        (callout_b, ax_bar, "(b) callout"),
    ):
        extent = artist.get_window_extent()
        panel = host.get_window_extent()
        if extent.x0 < panel.x0 - 1.0 or extent.x1 > panel.x1 + 1.0:
            raise SystemExit(
                f"{name} ({extent.x0:.0f}-{extent.x1:.0f} px) overruns its panel "
                f"({panel.x0:.0f}-{panel.x1:.0f} px); shorten it"
            )
    # The zero reference is the whole point of (b); nothing opaque may sit on it.
    zero_x = ax_bar.transData.transform((0.0, 0.0))[0]
    zero_y = sorted(ax_bar.transData.transform((0.0, value))[1] for value in zero_span)
    for artist in ax_bar.texts:
        extent = artist.get_window_extent()
        if extent.y1 < zero_y[0] or extent.y0 > zero_y[1]:
            continue  # clear of the segment the reference line actually spans
        if extent.x0 - 4.0 <= zero_x <= extent.x1 + 4.0:
            raise SystemExit(
                f"'{artist.get_text()}' sits on or against (b)'s zero reference "
                "line; move it"
            )
    if not zero_line.get_visible():
        raise SystemExit("(b) lost its zero reference line")

    # Unframed text inside a data area must not straddle a gridline: an opaque
    # pad on a gridline leaves a stub abutting a glyph, which reads as a broken
    # rule. The framed glosses and callouts are deliberate occluders and exempt.
    for host, name, exempt in (
        (ax, "(a)", {callout_a}),
        (ax_bar, "(b)", {callout_b}),
    ):
        grid_x = [
            host.transData.transform((tick, 0.0))[0]
            for tick in host.get_xticks()
            if host.get_xlim()[0] <= tick <= host.get_xlim()[1]
        ]
        panel = host.get_window_extent()
        for artist in host.texts:
            if artist in exempt:
                continue
            extent = artist.get_window_extent()
            if extent.y0 > panel.y1 or extent.y1 < panel.y0:
                continue  # header strips sit outside the gridded area
            for position in grid_x:
                if extent.x0 - 3.0 <= position <= extent.x1 + 3.0:
                    raise SystemExit(
                        f"{name}: '{artist.get_text()}' sits on a gridline "
                        f"({position:.0f} px in {extent.x0:.0f}-{extent.x1:.0f}); move it"
                    )

    # No two pieces of text inside a panel may touch, including the glosses.
    for host, name in ((ax, "(a)"), (ax_bar, "(b)")):
        boxes = [(text.get_text(), text.get_window_extent()) for text in host.texts]
        for i, (text_i, box_i) in enumerate(boxes):
            for text_j, box_j in boxes[i + 1:]:
                if box_i.overlaps(box_j):
                    raise SystemExit(
                        f"{name}: '{text_i}' and '{text_j}' overlap; move one"
                    )

    # No framed box may cover a data mark of its own panel: a finding written on
    # top of the mark that supports it is not evidence.
    for host, box, name in (
        (ax, callout_a, "(a) callout"),
        (ax_bar, callout_b, "(b) callout"),
    ):
        extent = box.get_window_extent()
        for line in host.lines:
            for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
                px, py = host.transData.transform((x_value, y_value))
                if extent.x0 - 6 <= px <= extent.x1 + 6 and extent.y0 - 6 <= py <= extent.y1 + 6:
                    raise SystemExit(f"{name}: the box covers a data mark; move it")

    # The zero reference must end inside the panel it rules, clear of the axis
    # spine: the one line (b) is read against never runs off its own frame.
    zero_end = ax_bar.transData.transform((0.0, zero_span[1]))[1]
    if zero_end - ax_bar.get_window_extent().y0 < 0.09 * fig.dpi:
        raise SystemExit("(b): the zero reference runs into the axis spine; shorten it")

    # Neighbouring tick labels must not touch: the fix is fewer ticks or a
    # narrower tick format, never a smaller tick font.
    for axis_owner, name in ((ax, "(a)"), (ax_bar, "(b)")):
        drawn = [
            label.get_window_extent()
            for label in axis_owner.get_xticklabels()
            if label.get_text()
        ]
        for left, right in zip(drawn, drawn[1:]):
            if right.x0 - left.x1 < 4.0:
                raise SystemExit(
                    f"{name} x tick labels collide ({left.x1:.0f} then {right.x0:.0f} px)"
                )

    # Both panels must grid every tick they show, and (b) must actually rule the
    # negative half it plots on: an unlabelled void is where the shipped rule's
    # whole result lives.
    for axis_owner, name in ((ax, "(a)"), (ax_bar, "(b)")):
        low_lim, high_lim = axis_owner.get_xlim()
        shown = [tick for tick in axis_owner.get_xticks() if low_lim <= tick <= high_lim]
        if not shown:
            raise SystemExit(f"{name} shows no tick inside its limits")
        if not all(line.get_visible() for line in axis_owner.get_xgridlines()):
            raise SystemExit(f"{name} has ticks without gridlines")
    if min(tick for tick in ax_bar.get_xticks() if tick >= xmin_b) >= 0.0:
        raise SystemExit("(b) plots negative gaps but rules no tick below zero")

    save(fig, "gate.pdf")
    record_provenance("gate.pdf", [T5, TRACE, GATE_VOCAB])
    print(
        json.dumps(
            {
                "caption_note": caption_note,
                "caption_rules": caption_rules,
                "live_capture": {
                    "per_stratum": live_counts,
                    "zero_tool": live_zero_tool,
                },
                "panel_a_delivered": {
                    stratum: {
                        "mean_tau_b": float(test_realized[stratum]["mean_tau_b"]),
                        "ci95_seed17": [float(value) for value in bounds[stratum]],
                        "claimed": claims[index],
                        "test_n": int(table[stratum]["test_n"]),
                    }
                    for index, stratum in enumerate(strata)
                },
                "panel_a_xlim": [xmin, xmax],
                "panel_b_worst_claim": worst_cases,
                "panel_b_xlim": [xmin_b, xmax_b],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
