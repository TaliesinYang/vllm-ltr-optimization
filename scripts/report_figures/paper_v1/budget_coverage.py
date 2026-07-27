"""Decision-budget coverage curve: what fraction of traffic gets a verdict.

Three enforced budgets (15 ms documented default, 50 and 75 ms matched to the
measured GPU ranker median of 37.9 ms) each rerun the gated arm with the
gateway restarted under the override. Fail-open is counted from the gateway's
own warning lines, sliced to each run's window (gateway-window.log written by
the rerun script), against the run's own recorded request count -- numerator
and denominator share a time window by construction.

Run: python3 budget_coverage.py [budget-ms ...]   (default: 15 50 75)
"""

from __future__ import annotations

import json
import sys

from _common import REPO

BUDGETS = [int(a) for a in sys.argv[1:]] or [15, 50, 75]


def main() -> None:
    rows = []
    for budget in BUDGETS:
        path = REPO / "runs" / f"contract-{budget}ms" / "counters.json"
        if not path.exists():
            print(f"  {budget} ms: counters absent, skipping")
            continue
        blob = json.loads(path.read_text())
        fo = blob.get("decision_fail_open") or {}
        total = int(blob.get("requests_recorded") or 0)
        if not total or fo.get("total") is None:
            print(f"  {budget} ms: incomplete counters, skipping")
            continue
        rows.append({
            "budget_ms": budget,
            "requests": total,
            "fail_open": int(fo["total"]),
            "fail_open_rate": fo["total"] / total,
            "coverage": 1.0 - fo["total"] / total,
            "breakdown": {k: v for k, v in fo.items()
                          if k not in ("total", "rate_of_requests")},
        })
    if not rows:
        raise SystemExit("no budget runs found")

    print(f"{'budget':>7} {'requests':>9} {'fail-open':>10} {'coverage':>9}")
    for r in rows:
        print(f"{r['budget_ms']:>5}ms {r['requests']:>9} {r['fail_open']:>10} "
              f"{r['coverage']:>8.1%}")

    out = REPO / "runs" / "budget-coverage.json"
    out.write_text(json.dumps({"budgets": rows}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
