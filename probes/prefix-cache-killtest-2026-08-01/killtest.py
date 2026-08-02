#!/usr/bin/env python3
"""Tool-schema prefix-cache kill test (offline block simulation).

Renders every captured agent request through the real Qwen2.5 chat template under
four tool-schema policies and measures how many full KV blocks survive between
adjacent requests of the same session.

No GPU involved: this measures theoretical cacheability only.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

SEED = 20260801
BLOCK_SIZE = 16
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
POLICIES = ["Original", "Stable Full", "Shuffled Full", "Frozen Thin"]


# ---------------------------------------------------------------- trace loading

def load_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            body = row["body"]
            body = body if isinstance(body, dict) else json.loads(body)
            body["messages"] = flatten_content(body["messages"])
            row["body"] = body
            # the earlier schema-variability capture has no request ids
            row.setdefault("request_id", f"row{len(rows):04d}")
            rows.append(row)
    rows.sort(key=lambda r: r["ts"])
    return rows


def flatten_content(messages: list[dict]) -> list[dict]:
    """Collapse OpenAI content-part lists into plain text.

    The Qwen chat template only accepts string content; vLLM's own OpenAI server
    flattens text parts the same way before rendering.
    """
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text")
            msg = {**msg, "content": text}
        out.append(msg)
    return out


def _msg_sig(messages: list[dict]) -> list[str]:
    return [json.dumps(m, sort_keys=True) for m in messages]


def build_sessions(rows: list[dict]) -> "collections.OrderedDict[str, list[dict]]":
    """The trace carries no session id, so chain requests by message prefix.

    Request B continues request A iff A's message list is a strict prefix of B's
    — exactly what an agent loop produces when it appends the tool result and
    re-sends. Requests that continue nothing start a new session. This keeps the
    25 independent title-generation requests apart, even though they share a
    fixed two-message head.
    """
    sigs = [_msg_sig(r["body"]["messages"]) for r in rows]
    parent = [None] * len(rows)
    for i in range(len(rows)):
        for j in range(i - 1, -1, -1):
            if len(sigs[j]) < len(sigs[i]) and sigs[i][: len(sigs[j])] == sigs[j]:
                parent[i] = j
                break

    root = [0] * len(rows)
    for i in range(len(rows)):
        root[i] = i if parent[i] is None else root[parent[i]]

    sessions: collections.OrderedDict[str, list[dict]] = collections.OrderedDict()
    for i, row in enumerate(rows):
        key = hashlib.sha256(rows[root[i]]["request_id"].encode()).hexdigest()[:16]
        row["_session"] = key
        # A session may branch (two continuations of one turn); pair each request
        # with its true predecessor rather than with its list neighbour.
        row["_parent"] = rows[parent[i]] if parent[i] is not None else None
        sessions.setdefault(key, []).append(row)
    return sessions


# ------------------------------------------------------------- tool canonicalisation

def canonicalise(obj):
    """Recursively sort dict keys so serialisation bytes are deterministic."""
    if isinstance(obj, dict):
        return {k: canonicalise(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [canonicalise(v) for v in obj]
    return obj


def tool_name(tool: dict) -> str:
    return tool.get("function", {}).get("name", "")


def session_tool_universe(session: list[dict]) -> list[dict]:
    """Union of every tool the session ever sent, canonicalised, name-sorted."""
    by_name: dict[str, dict] = {}
    for row in session:
        for tool in row["body"].get("tools") or []:
            by_name.setdefault(tool_name(tool), canonicalise(tool))
    return [by_name[n] for n in sorted(by_name)]


def session_used_tools(session: list[dict]) -> set[str]:
    """Tools actually invoked anywhere in the session (from assistant tool_calls)."""
    used: set[str] = set()
    for row in session:
        for msg in row["body"]["messages"]:
            for call in msg.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name:
                    used.add(name)
    return used


def tools_for_policy(policy, row, universe, thin, rng):
    original = row["body"].get("tools") or []
    if policy == "Original":
        return original
    if not original:
        # A request that carried no schema keeps carrying none under every policy.
        return []
    if policy == "Stable Full":
        return list(universe)
    if policy == "Shuffled Full":
        shuffled = list(universe)
        rng.shuffle(shuffled)
        return shuffled
    if policy == "Frozen Thin":
        return list(thin) if thin else list(universe)
    raise ValueError(policy)


# ------------------------------------------------------------------- rendering

class Renderer:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def render(self, messages, tools):
        text = self.tok.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return text, list(enc["input_ids"]), list(enc["offset_mapping"])


def segment_bounds(text: str) -> dict:
    """Char spans of the rendered prompt's segments under the Qwen2.5 template."""
    sys_end = text.find("<|im_end|>")
    # The template mentions "<tools></tools>" in its boilerplate sentence before
    # opening the real block, so anchor on the newline-delimited opener.
    open_tag, close_tag = "\n<tools>\n", "\n</tools>"
    tools_start = text.find(open_tag)
    tools_end = text.find(close_tag, tools_start + 1) if tools_start != -1 else -1
    if tools_start != -1 and tools_end != -1 and tools_end < sys_end:
        tools_start += len("\n")
        tools_end += len(close_tag)
    else:
        tools_start = tools_end = -1
    return {"sys_end": sys_end, "tools_start": tools_start, "tools_end": tools_end}


def classify_char(pos: int, bounds: dict) -> str:
    if bounds["tools_start"] != -1 and bounds["tools_start"] <= pos < bounds["tools_end"]:
        return "tools"
    if pos >= bounds["sys_end"]:
        return "messages"
    if bounds["tools_start"] != -1 and pos >= bounds["tools_end"]:
        return "other"
    return "system"


def count_schema_tokens(offsets, bounds) -> int:
    if bounds["tools_start"] == -1:
        return 0
    return sum(1 for s, _ in offsets if bounds["tools_start"] <= s < bounds["tools_end"])


# --------------------------------------------------------------------- metrics

def token_lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def byte_lcp(a: str, b: str) -> int:
    ab, bb = a.encode(), b.encode()
    n = min(len(ab), len(bb))
    for i in range(n):
        if ab[i] != bb[i]:
            return i
    return n


def pair_metrics(prev, cur, block_size):
    prev_text, prev_ids, _ = prev
    cur_text, cur_ids, cur_offsets = cur
    bounds = segment_bounds(cur_text)

    lcp = token_lcp(prev_ids, cur_ids)
    reusable_blocks = lcp // block_size
    reusable_block_tokens = reusable_blocks * block_size
    invalidated = max(len(cur_ids) - reusable_block_tokens, 0)

    if lcp < len(cur_offsets):
        segment = classify_char(cur_offsets[lcp][0], bounds)
    else:
        segment = "none"  # current prompt is a pure extension of the previous one

    return {
        "total_tokens": len(cur_ids),
        "schema_tokens": count_schema_tokens(cur_offsets, bounds),
        "byte_lcp": byte_lcp(prev_text, cur_text),
        "token_lcp": lcp,
        "reusable_blocks": reusable_blocks,
        "reusable_block_tokens": reusable_block_tokens,
        "reuse_ratio": reusable_block_tokens / len(cur_ids) if cur_ids else 0.0,
        "earliest_mutation_token": lcp,
        "earliest_mutation_block": lcp // block_size,
        "invalidated_suffix_tokens": invalidated,
        "first_changed_segment": segment,
    }


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    renderer = Renderer(tok)

    rows = load_rows(args.trace)
    sessions = build_sessions(rows)

    pair_rows = []
    skipped_sessions = 0
    thin_fallbacks = 0

    for s_idx, (skey, session) in enumerate(sessions.items()):
        if len(session) < 2:
            skipped_sessions += 1
            continue
        universe = session_tool_universe(session)
        used = session_used_tools(session)
        thin = [t for t in universe if tool_name(t) in used]
        if universe and not thin:
            thin_fallbacks += 1

        for policy in POLICIES:
            rendered = {}
            for t_idx, row in enumerate(session):
                rng = random.Random(SEED + 1000 * s_idx + t_idx)
                tools = tools_for_policy(policy, row, universe, thin, rng)
                rendered[row["request_id"]] = renderer.render(
                    row["body"]["messages"], tools)

            for t_idx, row in enumerate(session):
                prev = row["_parent"]
                if prev is None:
                    continue
                m = pair_metrics(rendered[prev["request_id"]],
                                 rendered[row["request_id"]], args.block_size)
                m.update(
                    session_id=skey,
                    prev_request_id=prev["request_id"],
                    request_id=row["request_id"],
                    policy=policy,
                    turn_index=t_idx,
                    n_tools_original=len(row["body"].get("tools") or []),
                    n_tools_universe=len(universe),
                    n_tools_thin=len(thin),
                )
                pair_rows.append(m)

    write_outputs(pair_rows, args, sessions, skipped_sessions, thin_fallbacks, tok)


# ------------------------------------------------------------------- reporting

FIELDS = [
    "session_id", "prev_request_id", "request_id", "policy", "turn_index",
    "total_tokens", "schema_tokens", "byte_lcp", "token_lcp",
    "reusable_blocks", "reusable_block_tokens", "reuse_ratio",
    "earliest_mutation_token", "earliest_mutation_block",
    "invalidated_suffix_tokens", "first_changed_segment",
    "n_tools_original", "n_tools_universe", "n_tools_thin",
]


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def write_outputs(pair_rows, args, sessions, skipped, thin_fallbacks, tok):
    import csv

    out = args.out
    with (out / "pair_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in pair_rows:
            w.writerow({k: r[k] for k in FIELDS})

    by_policy = collections.defaultdict(list)
    for r in pair_rows:
        by_policy[r["policy"]].append(r)

    with (out / "aggregate_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "policy", "n_pairs", "median_total_tokens", "median_token_lcp",
            "median_reuse_ratio", "median_invalidated_suffix",
            "mean_reuse_ratio", "segment_breakdown",
        ])
        for p in POLICIES:
            rs = by_policy.get(p, [])
            if not rs:
                continue
            seg = collections.Counter(r["first_changed_segment"] for r in rs)
            w.writerow([
                p, len(rs),
                median([r["total_tokens"] for r in rs]),
                median([r["token_lcp"] for r in rs]),
                round(median([r["reuse_ratio"] for r in rs]), 4),
                median([r["invalidated_suffix_tokens"] for r in rs]),
                round(sum(r["reuse_ratio"] for r in rs) / len(rs), 4),
                ";".join(f"{k}={v}" for k, v in sorted(seg.items())),
            ])

    with (out / "session_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "policy", "n_pairs", "median_reuse_ratio",
                    "median_invalidated_suffix", "n_tools_universe"])
        keyed = collections.defaultdict(list)
        for r in pair_rows:
            keyed[(r["session_id"], r["policy"])].append(r)
        for (sid, p), rs in keyed.items():
            w.writerow([sid, p, len(rs),
                        round(median([r["reuse_ratio"] for r in rs]), 4),
                        median([r["invalidated_suffix_tokens"] for r in rs]),
                        rs[0]["n_tools_universe"]])

    cfg = {
        "model": args.model,
        "tokenizer": args.model,
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(
            (tok.chat_template or "").encode()).hexdigest(),
        "kv_block_size": args.block_size,
        "seed": SEED,
        "n_requests": sum(len(v) for v in sessions.values()),
        "n_sessions": len(sessions),
        "n_singleton_sessions_skipped": skipped,
        "n_sessions_thin_fallback_to_full": thin_fallbacks,
        "n_pairs_per_policy": len(pair_rows) // len(POLICIES) if pair_rows else 0,
    }
    (out / "config.yaml").write_text(
        "\n".join(f"{k}: {json.dumps(v)}" for k, v in cfg.items()) + "\n")
    print(json.dumps(cfg, indent=2))

    for p in POLICIES:
        rs = by_policy.get(p, [])
        if rs:
            print(f"{p:>14}  n={len(rs):3d}  med_reuse={median([r['reuse_ratio'] for r in rs]):.4f}"
                  f"  med_lcp={median([r['token_lcp'] for r in rs]):.0f}"
                  f"  med_inval={median([r['invalidated_suffix_tokens'] for r in rs]):.0f}"
                  f"  seg={dict(collections.Counter(r['first_changed_segment'] for r in rs))}")


if __name__ == "__main__":
    main()
