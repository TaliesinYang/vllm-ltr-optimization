#!/usr/bin/env bash
# Read the counters an arm's conclusion depends on out of the logs the run
# already writes, and fail loudly when one is missing.
#
# Three numbers were absent from the run record and each supports a sentence
# the paper wants to print:
#
#   fail-open rate      Under a 15 ms contract a 37.9 ms ranker fails open on
#                       essentially every request. An arm that does not count
#                       this reports the pass-through path while appearing to
#                       report the gate. The gateway logs one warning per
#                       occurrence, so the count is already on disk.
#
#   prefix cache hits   Setting a flag is not evidence the flag took effect:
#                       the earlier session ran with prefix caching off and a
#                       0.0% hit rate, and nothing in the run record said so.
#                       vLLM reports the rate every ten seconds.
#
#   shed responses      A rejected request is not a fast request. Goodput under
#                       backpressure needs 429 and 503 counted separately from
#                       transport failures, which the per-sample http_status
#                       column now carries.
#
# Usage: collect_run_counters.sh <run-dir> [output.json]
set -euo pipefail

RUN_DIR="${1:?usage: collect_run_counters.sh <run-dir> [output.json]}"
OUT="${2:-$RUN_DIR/counters.json}"
LTR_ROOT="${LTR_ROOT:-/hy-tmp/ltr}"
GATEWAY_LOG="${GATEWAY_LOG:-$LTR_ROOT/runs/services/gateway.log}"

python3 - "$RUN_DIR" "$OUT" "$GATEWAY_LOG" <<'PY'
import csv, glob, json, os, re, sys
from collections import Counter

run_dir, out_path, gateway_log = sys.argv[1], sys.argv[2], sys.argv[3]

# --- shed and failure outcomes, from the per-sample record -------------------
statuses, total, errored = Counter(), 0, 0
for path in glob.glob(os.path.join(run_dir, "**", "*.samples.csv"), recursive=True):
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            if (row.get("error") or "").strip():
                errored += 1
                code = (row.get("http_status") or "").strip()
                statuses[code or "transport"] += 1

# --- prefix cache, from vLLM's own periodic report ---------------------------
# The last report is the one that has seen the whole run.
hit_rates = []
for path in glob.glob(os.path.join(run_dir, "**", "vllm.log"), recursive=True) + \
            glob.glob(os.path.join(os.path.dirname(run_dir), "*", "vllm.log")):
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        continue
    hit_rates += [float(m) for m in re.findall(r"Prefix cache hit rate: ([\d.]+)%", text)]
    enabled = re.findall(r"enable_prefix_caching=(\w+)", text)

prefix = {
    "reported_samples": len(hit_rates),
    "final_hit_rate_pct": hit_rates[-1] if hit_rates else None,
    "max_hit_rate_pct": max(hit_rates) if hit_rates else None,
    "enable_flag_in_log": enabled[-1] if 'enabled' in dir() and enabled else None,
}

# --- fail-open, from the gateway's own warning line --------------------------
fail_open = None
if os.path.exists(gateway_log):
    text = open(gateway_log, errors="ignore").read()
    fail_open = {
        "timeout_or_call_failed": len(re.findall(r"ltr decision call failed; fail-open", text)),
        "non_200": len(re.findall(r"ltr decision non-200; fail-open", text)),
        "decode_failed": len(re.findall(r"ltr decision decode failed; fail-open", text)),
        "contract_violation": len(re.findall(r"ltr decision contract violation; fail-open", text)),
    }
    fail_open["total"] = sum(fail_open.values())
    fail_open["rate_of_requests"] = (fail_open["total"] / total) if total else None

payload = {
    "run_dir": run_dir,
    "requests_recorded": total,
    "requests_errored": errored,
    "status_breakdown": dict(statuses),
    "prefix_cache": prefix,
    "decision_fail_open": fail_open,
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")

print(json.dumps(payload, indent=2))

# A missing counter is a silent hole in an argument, so say so rather than
# leaving a null for someone to discover after the machine is returned.
missing = []
if not total:
    missing.append("no samples recorded")
if prefix["final_hit_rate_pct"] is None:
    missing.append("no prefix-cache report found in any vllm.log")
if fail_open is None:
    missing.append(f"gateway log absent at {gateway_log}")
if missing:
    print("COUNTERS INCOMPLETE: " + "; ".join(missing), file=sys.stderr)
    raise SystemExit(3)
print("COUNTERS OK", file=sys.stderr)
PY
