#!/usr/bin/env python3
"""Phase 2 — cold-start prefill and KV footprint: Original Full vs Frozen Thin.

The offline probe showed Frozen Thin cuts prompt tokens by 22.3% (10-tool) and
67.8% (170-tool) without improving warm reuse. This asks the only question that
matters for that finding: does the token cut turn into real prefill/KV savings?

Cold means cold: each measured request is the FIRST request of its session, sent
to a server whose cache has never seen it. The caller restarts the server between
policies; within a run each session head is sent exactly once.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

PROBE = Path(__file__).resolve().parents[2] / "prefix-cache-killtest-2026-08-01"
spec = importlib.util.spec_from_file_location("killtest", PROBE / "killtest.py")
kt = importlib.util.module_from_spec(spec)
sys.modules["killtest"] = kt
spec.loader.exec_module(kt)

GAUGES = ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
          "vllm:num_requests_running", "vllm:num_requests_waiting"]
# Prometheus counters carry a _total suffix.
COUNTERS = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total"]


def scrape(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            body = r.read().decode()
    except Exception as exc:                       # noqa: BLE001
        return {"_err": str(exc)}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in GAUGES or name in COUNTERS:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
    return out


def post(base: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (time.perf_counter() - t0) * 1000.0, json.loads(r.read().decode()), None
    except urllib.error.HTTPError as exc:
        # Include the server's message, not just the status code.
        detail = exc.read().decode()[:400]
        return (time.perf_counter() - t0) * 1000.0, None, f"HTTP {exc.code}: {detail}"
    except Exception as exc:                       # noqa: BLE001
        return (time.perf_counter() - t0) * 1000.0, None, f"{type(exc).__name__}: {exc}"


def session_heads(trace: Path):
    """First request of every multi-turn session -- the cold-start population."""
    rows = kt.load_rows(trace)
    sessions = kt.build_sessions(rows)
    heads = []
    for s_idx, (skey, session) in enumerate(sessions.items()):
        if len(session) < 2:
            continue
        universe = kt.session_tool_universe(session)
        used = kt.session_used_tools(session)
        thin = [t for t in universe if kt.tool_name(t) in used]
        heads.append({"session_id": skey, "s_idx": s_idx, "row": session[0],
                      "universe": universe, "thin": thin})
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--policy", required=True, choices=["Original", "Frozen Thin"])
    ap.add_argument("--cache-mode", required=True, choices=["on", "off"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="label, e.g. vanilla_10tool")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=1)
    args = ap.parse_args()

    heads = session_heads(args.trace)
    print(f"[{args.dataset}/{args.policy}/cache={args.cache_mode}/run{args.run_id}] "
          f"{len(heads)} cold session heads", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    fh = args.out.open("a", newline="")
    writer = None
    ok = 0

    for h in heads:
        rng = random.Random(kt.SEED + 1000 * h["s_idx"])
        tools = kt.tools_for_policy(args.policy, h["row"], h["universe"], h["thin"], rng)
        body = {"model": args.model, "messages": h["row"]["body"]["messages"],
                "max_tokens": args.max_tokens, "temperature": 0.0, "stream": False}
        if tools:
            body["tools"] = tools

        before = scrape(args.base)
        wall_ms, resp, err = post(args.base, body)
        after = scrape(args.base)

        usage = (resp or {}).get("usage", {}) or {}
        details = usage.get("prompt_tokens_details") or {}
        ptok = usage.get("prompt_tokens")
        # vLLM 0.9.2 leaves prompt_tokens_details null; the hits counter delta is
        # the fallback, valid because this client sends one request at a time.
        cached = details.get("cached_tokens")
        cached_source = "usage"
        if cached is None:
            b, a = before.get("vllm:gpu_prefix_cache_hits_total"), \
                   after.get("vllm:gpu_prefix_cache_hits_total")
            cached = None if (b is None or a is None) else a - b
            cached_source = "metrics_delta" if cached is not None else "unavailable"
        rec = {
            "dataset": args.dataset, "policy": args.policy,
            "cache_mode": args.cache_mode, "run_id": args.run_id, "model": args.model,
            "session_id": h["session_id"],
            "request_id": h["row"]["request_id"],
            "n_tools_sent": len(tools),
            "prompt_tokens": ptok,
            "cached_tokens": cached,
            "cached_source": cached_source,
            "new_prefill_tokens": (None if (cached is None or ptok is None)
                                   else ptok - cached),
            "wall_ms": round(wall_ms, 3),
            "error": err or "",
            "oom": int(bool(err and ("out of memory" in err.lower()
                                     or "oom" in err.lower()))),
        }
        for k in GAUGES:
            rec[f"g_{k.replace('vllm:', '')}"] = after.get(k)
        for k in COUNTERS:
            b, a = before.get(k), after.get(k)
            rec[f"d_{k.replace('vllm:', '')}"] = (None if (b is None or a is None)
                                                  else round(a - b, 6))
        rec["scrape_error"] = before.get("_err") or after.get("_err") or ""

        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(rec))
            if new:
                writer.writeheader()
        writer.writerow(rec)
        fh.flush()
        if err:
            print(f"  ! {h['request_id'][:8]}: {err[:120]}", flush=True)
        else:
            ok += 1

    fh.close()
    print(f"  {ok}/{len(heads)} ok -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
