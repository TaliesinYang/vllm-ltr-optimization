#!/usr/bin/env python3
"""Live request dashboard for the VeloxMesh demo stack.

Reads two append-only logs and joins them by request_id:
  * /tmp/gateway.log     - gateway JSON lines ("msg":"request completed")
  * /tmp/decisions.jsonl - one JSON line per decision (DECISION_LOG_PATH)

Serves an auto-refreshing HTML page on 127.0.0.1:9310. Python stdlib only.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

GATEWAY_LOG_PATH = os.environ.get("GATEWAY_LOG_PATH", "/tmp/gateway.log")
DECISION_LOG_PATH = os.environ.get("DECISION_LOG_PATH", "/tmp/decisions.jsonl")

HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("DASHBOARD_PORT", "9310"))

MAX_ROWS = 50
MAX_LINES_SCANNED = 5000

VERDICT_VOUCHED = "VOUCHED"
VERDICT_ABSTAINED = "ABSTAINED"
VERDICT_NOT_SCORED = "NOT SCORED"


def read_json_lines(path: str) -> list[dict[str, Any]]:
    """Parse the tail of a JSONL file. Missing file or bad lines -> skipped."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = deque(handle, maxlen=MAX_LINES_SCANNED)
    except OSError:
        return []
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def parse_gateway_time(raw: Any) -> float | None:
    """Gateway stamps RFC3339 with offset; convert to epoch seconds."""
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def load_gateway_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_json_lines(GATEWAY_LOG_PATH):
        if row.get("msg") != "request completed":
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        rows.append(
            {
                "request_id": request_id,
                "ts": parse_gateway_time(row.get("time")),
                "status": row.get("status"),
                "latency_ms": row.get("latency_ms"),
            }
        )
    return rows


def load_decision_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_json_lines(DECISION_LOG_PATH):
        ts = row.get("ts")
        rows.append(
            {
                "request_id": row.get("request_id"),
                "ts": float(ts) if isinstance(ts, (int, float)) else None,
                "decision_id": row.get("decision_id"),
                "confidence": row.get("confidence"),
                "vouched": row.get("vouched"),
                "reason_code": row.get("reason_code"),
                "estimated_tokens": row.get("estimated_tokens"),
                "decision_ms": row.get("decision_ms"),
            }
        )
    return rows


def verdict_for(decision: dict[str, Any] | None) -> str:
    if decision is None:
        return VERDICT_NOT_SCORED
    return VERDICT_VOUCHED if decision.get("vouched") else VERDICT_ABSTAINED


def format_clock(ts: float | None) -> str:
    if ts is None:
        return "--:--:--"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def merge_row(decision: dict[str, Any] | None, gateway: dict[str, Any] | None) -> dict[str, Any]:
    decision = decision or {}
    gateway = gateway or {}
    request_id = decision.get("request_id") or gateway.get("request_id") or ""
    ts = decision.get("ts") or gateway.get("ts")
    return {
        "request_id": request_id,
        "request_id_short": request_id[:8],
        "ts": ts,
        "clock": format_clock(ts),
        "verdict": verdict_for(decision or None),
        "confidence": decision.get("confidence"),
        "estimated_tokens": decision.get("estimated_tokens"),
        "decision_ms": decision.get("decision_ms"),
        "reason_code": decision.get("reason_code"),
        "status": gateway.get("status"),
        "latency_ms": gateway.get("latency_ms"),
        "joined": bool(decision) and bool(gateway),
    }


def build_rows() -> list[dict[str, Any]]:
    """Pair each decision with an unconsumed gateway row of the same id.

    Demo probes reuse a fixed request_id, so repeats are matched in arrival
    order rather than collapsed into a single row. Gateway rows left over
    had no decision (fail-open or the service was down) -> NOT SCORED.
    """
    pending: dict[str, deque[dict[str, Any]]] = {}
    for gateway in load_gateway_rows():
        pending.setdefault(gateway["request_id"], deque()).append(gateway)

    rows = []
    for decision in load_decision_rows():
        queue = pending.get(decision["request_id"] or "")
        gateway = queue.popleft() if queue else None
        rows.append(merge_row(decision, gateway))

    for queue in pending.values():
        for gateway in queue:
            rows.append(merge_row(None, gateway))

    rows.sort(key=lambda row: row["ts"] if row["ts"] is not None else 0.0, reverse=True)
    return rows


def build_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows if isinstance(row["latency_ms"], (int, float))]
    return {
        "total": len(rows),
        "vouched": sum(1 for row in rows if row["verdict"] == VERDICT_VOUCHED),
        "abstained": sum(1 for row in rows if row["verdict"] == VERDICT_ABSTAINED),
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
    }


def build_payload() -> dict[str, Any]:
    rows = build_rows()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": build_stats(rows),
        "rows": rows[:MAX_ROWS],
    }


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VeloxMesh &middot; Live Decisions</title>
<style>
  :root {
    --bg: #FAF9F5;
    --ink: #1F1E1D;
    --abstain: #D97757;
    --vouch: #7D9B76;
    --rule: #E3E0D8;
    --muted: #6B6862;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 40px 32px 64px;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 {
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 400;
    font-size: 34px;
    letter-spacing: -0.01em;
    margin: 0 0 4px;
  }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
  .chips { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
  .chip {
    background: #fff;
    border: 1px solid var(--rule);
    border-radius: 10px;
    padding: 12px 18px;
    min-width: 132px;
  }
  .chip .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .chip .value { font-size: 26px; font-variant-numeric: tabular-nums; }
  .chip.vouch .value { color: var(--vouch); }
  .chip.abstain .value { color: var(--abstain); }
  .tablewrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    font-weight: 600;
    padding: 12px 14px;
    border-bottom: 1px solid var(--rule);
    white-space: nowrap;
  }
  td {
    padding: 11px 14px;
    border-bottom: 1px solid #F1EFE9;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  tr:last-child td { border-bottom: none; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #fff;
  }
  .badge.vouched { background: var(--vouch); }
  .badge.abstained { background: var(--abstain); }
  .badge.unscored { background: transparent; color: var(--muted); border: 1px solid var(--rule); }
  .dim { color: var(--muted); }
  .empty { padding: 36px 14px; text-align: center; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <h1>VeloxMesh &middot; Live Decisions</h1>
  <div class="sub">Gateway traffic joined with reliability-gate verdicts &middot; refreshing every second &middot; <span id="stamp">--</span></div>
  <div class="chips">
    <div class="chip"><div class="label">Requests</div><div class="value" id="s-total">0</div></div>
    <div class="chip vouch"><div class="label">Vouched</div><div class="value" id="s-vouched">0</div></div>
    <div class="chip abstain"><div class="label">Abstained</div><div class="value" id="s-abstained">0</div></div>
    <div class="chip"><div class="label">Median latency</div><div class="value" id="s-latency">--</div></div>
  </div>
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Request</th><th>Verdict</th><th>Confidence</th>
          <th>Est. tokens</th><th>Decision ms</th><th>Status</th><th>Latency ms</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</div>
<script>
const BADGE = {"VOUCHED": "vouched", "ABSTAINED": "abstained", "NOT SCORED": "unscored"};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function cell(value, digits) {
  if (value === null || value === undefined) return '<td class="dim">&mdash;</td>';
  if (digits !== undefined && typeof value === "number") {
    return '<td>' + value.toFixed(digits) + '</td>';
  }
  return '<td>' + escapeHtml(value) + '</td>';
}

function render(data) {
  document.getElementById("stamp").textContent = "updated " + data.generated_at;
  document.getElementById("s-total").textContent = data.stats.total;
  document.getElementById("s-vouched").textContent = data.stats.vouched;
  document.getElementById("s-abstained").textContent = data.stats.abstained;
  const lat = data.stats.median_latency_ms;
  document.getElementById("s-latency").textContent = lat === null ? "--" : lat;

  const body = document.getElementById("rows");
  if (!data.rows.length) {
    body.innerHTML = '<tr><td class="empty" colspan="8">Waiting for traffic&hellip;</td></tr>';
    return;
  }
  body.innerHTML = data.rows.map(function (row) {
    return '<tr>'
      + '<td class="mono">' + escapeHtml(row.clock) + '</td>'
      + '<td class="mono">' + (row.request_id_short ? escapeHtml(row.request_id_short) : '&mdash;') + '</td>'
      + '<td><span class="badge ' + BADGE[row.verdict] + '">' + row.verdict + '</span></td>'
      + cell(row.confidence, 4)
      + cell(row.estimated_tokens)
      + cell(row.decision_ms)
      + cell(row.status)
      + cell(row.latency_ms)
      + '</tr>';
  }).join("");
}

function poll() {
  fetch("/data.json", {cache: "no-store"})
    .then(function (r) { return r.json(); })
    .then(render)
    .catch(function () { /* transient; next tick retries */ });
}

poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "VeloxMeshDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._respond(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/data.json":
            body = json.dumps(build_payload()).encode("utf-8")
            self._respond(200, "application/json", body)
        else:
            self._respond(404, "text/plain; charset=utf-8", b"not found\n")

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Keep the demo log readable; access lines add nothing here."""


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    server.daemon_threads = True
    print(f"dashboard listening on http://{HOST}:{PORT}", flush=True)
    print(f"  gateway log : {GATEWAY_LOG_PATH}", flush=True)
    print(f"  decision log: {DECISION_LOG_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
