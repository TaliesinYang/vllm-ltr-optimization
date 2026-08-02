#!/usr/bin/env python3
"""Phase 1 — replay real agent request pairs against a live vLLM server.

Answers RQ1: does the offline block simulation correspond to real prefix-cache
behaviour? The falsification check is Shuffled Full: if it does NOT degrade on
real hardware, the offline model or the cache configuration is wrong and the
whole direction stops here.

One process = one (policy, cache-mode) arm against one server lifetime. The
caller restarts the server between arms; this script never assumes a cold cache
it did not observe.
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

POLICIES = ["Original", "Stable Full", "Shuffled Full", "Frozen Thin"]

# /metrics families we read. Counters are differenced across a request.
METRIC_KEYS = [
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:gpu_prefix_cache_queries",
    "vllm:gpu_prefix_cache_hits",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
]


def scrape(base: str) -> dict:
    """Sum each metric family across label sets; missing families are skipped."""
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            body = r.read().decode()
    except Exception as exc:                      # noqa: BLE001 - reported, not hidden
        return {"_scrape_error": str(exc)}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in METRIC_KEYS:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
    return out


def post_chat(base: str, body: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        # Surface the server's message; a bare "HTTP 400" hides the actual cause.
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:400]}") from None
    return {"wall_ms": (time.perf_counter() - t0) * 1000.0, "resp": payload}


def build_pairs(trace: Path):
    rows = kt.load_rows(trace)
    sessions = kt.build_sessions(rows)
    pairs = []
    for s_idx, (skey, session) in enumerate(sessions.items()):
        if len(session) < 2:
            continue
        universe = kt.session_tool_universe(session)
        used = kt.session_used_tools(session)
        thin = [t for t in universe if kt.tool_name(t) in used]
        for t_idx, row in enumerate(session):
            prev = row.get("_parent")
            if prev is None:
                continue
            pairs.append({
                "session_id": skey, "s_idx": s_idx, "t_idx": t_idx,
                "prev": prev, "cur": row,
                "universe": universe, "thin": thin,
            })
    return pairs


def request_body(row, policy, universe, thin, rng, model, max_tokens):
    tools = kt.tools_for_policy(policy, row, universe, thin, rng)
    body = {
        "model": model,
        "messages": row["body"]["messages"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--cache-mode", required=True, choices=["on", "off"],
                    help="what the server was launched with; recorded, not set here")
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True, help="e.g. restart index")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--warmup-pairs", type=int, default=5)
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    pairs = build_pairs(args.trace)
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    print(f"[{args.policy}/{args.cache_mode}/run{args.run_id}] {len(pairs)} pairs "
          f"({args.warmup_pairs} warm-up)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    fh = args.out.open("a", newline="")
    writer = None
    measured = 0

    for i, p in enumerate(pairs):
        is_warmup = i < args.warmup_pairs
        rng_prev = random.Random(kt.SEED + 1000 * p["s_idx"] + (p["t_idx"] - 1))
        rng_cur = random.Random(kt.SEED + 1000 * p["s_idx"] + p["t_idx"])

        # Replay the pair in order: the parent primes the cache, the child is measured.
        for role, row, rng in (("prev", p["prev"], rng_prev), ("cur", p["cur"], rng_cur)):
            body = request_body(row, args.policy, p["universe"], p["thin"],
                                rng, args.model, args.max_tokens)
            before = scrape(args.base)
            try:
                r = post_chat(args.base, body)
            except Exception as exc:              # noqa: BLE001
                print(f"  ! request failed ({role}, pair {i}): {exc}", flush=True)
                break
            after = scrape(args.base)
            if role != "cur" or is_warmup:
                continue

            usage = r["resp"].get("usage", {}) or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            rec = {
                "policy": args.policy,
                "cache_mode": args.cache_mode,
                "run_id": args.run_id,
                "model": args.model,
                "session_id": p["session_id"],
                "prev_request_id": p["prev"]["request_id"],
                "request_id": p["cur"]["request_id"],
                "pair_index": i,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "uncached_prompt_tokens": (
                    None if (cached is None or prompt_tokens is None)
                    else prompt_tokens - cached),
                "cached_frac": (
                    None if (cached is None or not prompt_tokens)
                    else round(cached / prompt_tokens, 6)),
                "completion_tokens": usage.get("completion_tokens"),
                "wall_ms": round(r["wall_ms"], 3),
                "n_tools_sent": len(body.get("tools") or []),
            }
            for k in METRIC_KEYS:
                b, a = before.get(k), after.get(k)
                rec[f"d_{k.replace('vllm:', '')}"] = (
                    None if (b is None or a is None) else round(a - b, 6))
            rec["scrape_error"] = before.get("_scrape_error") or after.get("_scrape_error") or ""

            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(rec))
                if new_file:
                    writer.writeheader()
            writer.writerow(rec)
            fh.flush()
            measured += 1

    fh.close()
    print(f"  measured {measured} pairs -> {args.out}", flush=True)
    if measured == 0:
        sys.exit("no pairs measured -- server unreachable or all requests failed")


if __name__ == "__main__":
    main()
