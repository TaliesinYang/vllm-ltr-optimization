#!/usr/bin/env python3
"""Cross-session prefix reuse: does hoisting the tool schema help on real hardware?

The offline audit found that across sessions the earliest divergence moves from
the appended message to the *system prompt* (90/91 and 21/21 pairs), so a schema
that is provably byte-stable sits behind a volatile prefix and can never be
reached. Reordering the rendered prompt so the schema leads recovered +25.5 pp
(vanilla) and +68.2 pp (170-tool) offline when sessions share a toolset, and cost
-9.1 pp when they did not.

The chat endpoint applies the template server-side, so layout cannot be
controlled through it. This renders locally with the real tokenizer, permutes the
bytes, and posts the raw string to /v1/completions -- same tokens, same model,
layout under our control.

Session heads are sent in arrival order within one server lifetime: head N may
reuse anything heads 1..N-1 left behind. That is the realistic cross-session
question, not an all-pairs upper bound.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE.parents[1] / "prefix-cache-killtest-2026-08-01"
spec = importlib.util.spec_from_file_location("killtest", PROBE / "killtest.py")
kt = importlib.util.module_from_spec(spec)
sys.modules["killtest"] = kt
spec.loader.exec_module(kt)

HITS = "vllm:gpu_prefix_cache_hits_total"
QUERIES = "vllm:gpu_prefix_cache_queries_total"


def scrape(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            body = r.read().decode()
    except Exception as exc:                        # noqa: BLE001
        return {"_err": str(exc)}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in (HITS, QUERIES):
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
    return out


def hoist_schema(text: str) -> str:
    """Move the `# Tools` block to the front of the system message.

    A pure permutation of bytes the template already produced -- asserted, so a
    template change cannot silently turn this into a different prompt.
    """
    head = "<|im_start|>system\n"
    sys_end = text.find("<|im_end|>")
    block_start = text.find("\n\n# Tools\n", 0, sys_end)
    if block_start == -1:
        return text
    system_content = text[len(head):block_start]
    schema_block = text[block_start + 2:sys_end]
    hoisted = head + schema_block + "\n\n" + system_content + text[sys_end:]
    assert sorted(hoisted) == sorted(text), "hoist must preserve bytes"
    return hoisted


def post_completion(base: str, model: str, prompt: str, max_tokens: int):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request(
        f"{base}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return (time.perf_counter() - t0) * 1000.0, json.loads(r.read().decode()), None
    except urllib.error.HTTPError as exc:
        return (time.perf_counter() - t0) * 1000.0, None, \
               f"HTTP {exc.code}: {exc.read().decode()[:300]}"
    except Exception as exc:                        # noqa: BLE001
        return (time.perf_counter() - t0) * 1000.0, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layout", required=True, choices=["as-is", "hoisted"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = kt.load_rows(args.trace)
    sessions = kt.build_sessions(rows)
    heads = [s[0] for s in sessions.values() if len(s) >= 2]
    print(f"[{args.layout}/run{args.run_id}] {len(heads)} session heads, in arrival order",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    fh = args.out.open("a", newline="")
    writer = None

    for idx, row in enumerate(heads):
        tools = row["body"].get("tools") or []
        text = tok.apply_chat_template(row["body"]["messages"], tools=tools or None,
                                       tokenize=False, add_generation_prompt=True)
        if args.layout == "hoisted":
            text = hoist_schema(text)
        n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])

        before = scrape(args.base)
        wall_ms, resp, err = post_completion(args.base, args.model, text, args.max_tokens)
        after = scrape(args.base)

        def delta(k):
            b, a = before.get(k), after.get(k)
            return None if (b is None or a is None) else a - b

        cached = delta(HITS)
        usage = (resp or {}).get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        rec = {
            "layout": args.layout, "run_id": args.run_id, "model": args.model,
            "head_index": idx,
            "session_id": row["_session"], "request_id": row["request_id"],
            "n_tools": len(tools),
            "local_tokens": n_tokens,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached,
            "queried_tokens": delta(QUERIES),
            "cached_frac": (None if (cached is None or not prompt_tokens)
                            else round(cached / prompt_tokens, 6)),
            "new_prefill_tokens": (None if (cached is None or prompt_tokens is None)
                                   else prompt_tokens - cached),
            "wall_ms": round(wall_ms, 3),
            "error": err or "",
        }
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(rec))
            if new:
                writer.writeheader()
        writer.writerow(rec)
        fh.flush()
        if err:
            print(f"  ! head {idx}: {err[:140]}", flush=True)

    fh.close()
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
