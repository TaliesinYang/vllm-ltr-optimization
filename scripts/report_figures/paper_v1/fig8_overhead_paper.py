"""fig:overhead -- publishes the gateway overhead figure into the paper.

The plotting lives in scripts/report_figures/fig8_overhead.py and is not
duplicated here. What this wrapper adds is provenance: the measurement behind
this figure comes from the 2026-07-19 rental, NOT from the trace-calibrated
Block-1 session, and that distinction has to be recorded somewhere a reader
can check rather than left implicit in a manually copied file.

E3's text is scoped to match: it claims a cost of placing the Ranker on the
synchronous path, measured against a direct-to-engine baseline, and says
nothing about the Block-1 workload.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib import pyplot as plt

from _common import REPO, record_provenance, save

sys.path.insert(0, str(REPO / "scripts" / "report_figures"))

from fig8_overhead import DATA_PATH, build_figure, load_overhead_pairs  # noqa: E402


def main() -> None:
    pairs = load_overhead_pairs()
    fig, summaries = build_figure(pairs)
    save(fig, "overhead.pdf")
    plt.close(fig)
    record_provenance("overhead.pdf", [Path(DATA_PATH)])

    for metric in ("ttft", "ttlt"):
        s = summaries[metric]
        print(f"{metric.upper()} delta_mean_ms={s['delta_mean']:.1f} "
              f"CI95=[{s['delta_mean_ci'][0]:.1f}, {s['delta_mean_ci'][1]:.1f}]")
    print(f"TTLT matched pairs dropped: {pairs['dropped']}")


if __name__ == "__main__":
    main()
