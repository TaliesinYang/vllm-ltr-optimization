#!/usr/bin/env python3
"""Phase 0A offline audit of the 2026-08-01 kill test.

Three checks that do not need a GPU:
  1. audit_pairs.csv        — machine-checkable evidence for every parent-child pair,
                              so the 35 pairs can be signed off by hand.
  2. block_size_sensitivity.csv — does the No-Go hold at block size 8/16/32?
  3. cross_session_lcp.csv  — can a *new* session reuse a previous session's prefix?

Run from this directory:
  python phase0_audit.py --out results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import statistics
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parents[2] / "prefix-cache-killtest-2026-08-01"

spec = importlib.util.spec_from_file_location("killtest", PROBE / "killtest.py")
kt = importlib.util.module_from_spec(spec)
sys.modules["killtest"] = kt
spec.loader.exec_module(kt)

DATASETS = {
    "vanilla_10tool": PROBE.parent / "agent-traces-2026-07-26" / "agent_trace_vanilla.jsonl.gz",
    "full_170tool": PROBE.parent / "schema-variability-2026-07-25" / "captured_requests_v2.jsonl",
}
BLOCK_SIZES = (8, 16, 32)


def h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def median(xs):
    return statistics.median(xs) if xs else float("nan")


# ------------------------------------------------------------------ 1. pair audit

AUDIT_FIELDS = [
    "dataset", "session_id", "prev_request_id", "request_id",
    "prev_n_msgs", "n_msgs", "n_msgs_appended", "appended_roles",
    "strict_prefix_ok", "system_hash", "system_hash_equal", "model_equal",
    "prev_toolset_hash", "toolset_hash", "toolset_equal",
    "prev_n_tools", "n_tools", "dt_seconds",
    "prev_has_siblings", "prev_child_count", "body_is_duplicate",
]


def audit_pairs(rows, sessions, dataset):
    body_hashes = {}
    for r in rows:
        body_hashes.setdefault(h(r["body"]), []).append(r["request_id"])

    child_count = {}
    for r in rows:
        p = r.get("_parent")
        if p is not None:
            child_count[p["request_id"]] = child_count.get(p["request_id"], 0) + 1

    out = []
    for skey, session in sessions.items():
        if len(session) < 2:
            continue
        for row in session:
            prev = row.get("_parent")
            if prev is None:
                continue
            pm, cm = prev["body"]["messages"], row["body"]["messages"]
            psig, csig = kt._msg_sig(pm), kt._msg_sig(cm)
            ptools = prev["body"].get("tools") or []
            ctools = row["body"].get("tools") or []
            out.append({
                "dataset": dataset,
                "session_id": skey,
                "prev_request_id": prev["request_id"],
                "request_id": row["request_id"],
                "prev_n_msgs": len(pm),
                "n_msgs": len(cm),
                "n_msgs_appended": len(cm) - len(pm),
                "appended_roles": "|".join(m["role"] for m in cm[len(pm):]),
                "strict_prefix_ok": len(psig) < len(csig) and csig[:len(psig)] == psig,
                "system_hash": h(pm[0]),
                "system_hash_equal": h(pm[0]) == h(cm[0]),
                "model_equal": prev["body"].get("model") == row["body"].get("model"),
                "prev_toolset_hash": h(ptools),
                "toolset_hash": h(ctools),
                "toolset_equal": h(ptools) == h(ctools),
                "prev_n_tools": len(ptools),
                "n_tools": len(ctools),
                "dt_seconds": round(row["ts"] - prev["ts"], 3),
                "prev_child_count": child_count.get(prev["request_id"], 0),
                "prev_has_siblings": child_count.get(prev["request_id"], 0) > 1,
                "body_is_duplicate": len(body_hashes[h(row["body"])]) > 1,
            })
    return out


# ------------------------------------------------- 2. block-size sensitivity

def block_sensitivity(dataset, out_dir):
    """Re-derive reuse at other block sizes from the recorded token LCPs."""
    src = PROBE / ("results" if dataset == "vanilla_10tool" else "results_170tool")
    rows = list(csv.DictReader((src / "pair_metrics.csv").open()))
    res = []
    for bs in BLOCK_SIZES:
        by_policy = {}
        for r in rows:
            lcp, total = int(r["token_lcp"]), int(r["total_tokens"])
            reusable = (lcp // bs) * bs
            by_policy.setdefault(r["policy"], []).append(
                (reusable / total, max(total - reusable, 0)))
        for pol in kt.POLICIES:
            vals = by_policy.get(pol, [])
            if vals:
                res.append({
                    "dataset": dataset, "block_size": bs, "policy": pol,
                    "n_pairs": len(vals),
                    "median_reuse_ratio": round(median([v[0] for v in vals]), 4),
                    "median_invalidated_suffix": median([v[1] for v in vals]),
                })
    return res


# ------------------------------------------------ 3. cross-session prefix reuse

def cross_session_lcp(rows, sessions, dataset, renderer, block_size=16):
    """How much of a fresh session's first request is already cached by another
    session's first request? This is where a stable schema *could* still pay off."""
    firsts = [s[0] for s in sessions.values() if len(s) >= 2]
    rendered = {}
    for r in firsts:
        rendered[r["request_id"]] = renderer.render(
            r["body"]["messages"], r["body"].get("tools") or [])

    out = []
    for a, b in itertools.combinations(firsts, 2):
        _, ids_a, _ = rendered[a["request_id"]]
        text_b, ids_b, offs_b = rendered[b["request_id"]]
        lcp = kt.token_lcp(ids_a, ids_b)
        bounds = kt.segment_bounds(text_b)
        seg = kt.classify_char(offs_b[lcp][0], bounds) if lcp < len(offs_b) else "none"
        reusable = (lcp // block_size) * block_size
        out.append({
            "dataset": dataset,
            "request_id_a": a["request_id"], "request_id_b": b["request_id"],
            "tokens_a": len(ids_a), "tokens_b": len(ids_b),
            "token_lcp": lcp,
            "reuse_ratio": round(reusable / len(ids_b), 4),
            "first_changed_segment": seg,
            "same_toolset": h(a["body"].get("tools") or []) == h(b["body"].get("tools") or []),
        })
    return out


# ----------------------------------------- 4. does hoisting the schema help?

def hoist_schema(text: str) -> str:
    """Move the tool-schema block to the front of the system block.

    Pure byte reordering of what the template already produced: the same schema
    text, placed before the volatile part of the system prompt instead of after
    it. Models the layout a client could adopt; asserts nothing is lost.
    """
    head = "<|im_start|>system\n"
    sys_end = text.find("<|im_end|>")
    block_start = text.find("\n\n# Tools\n", 0, sys_end)
    if block_start == -1:
        return text
    system_content = text[len(head):block_start]
    schema_block = text[block_start + 2:sys_end]     # "# Tools ... </tool_call>"
    hoisted = head + schema_block + "\n\n" + system_content + text[sys_end:]
    assert sorted(hoisted) == sorted(text), "hoist must preserve bytes"
    return hoisted


def cross_session_hoisted(rows, sessions, dataset, renderer, block_size=16):
    firsts = [s[0] for s in sessions.values() if len(s) >= 2]
    tok = renderer.tok
    rendered = {}
    for r in firsts:
        text, _, _ = renderer.render(r["body"]["messages"], r["body"].get("tools") or [])
        hoisted = hoist_schema(text)
        rendered[r["request_id"]] = list(
            tok(hoisted, add_special_tokens=False)["input_ids"])

    out = []
    for a, b in itertools.combinations(firsts, 2):
        ids_a, ids_b = rendered[a["request_id"]], rendered[b["request_id"]]
        lcp = kt.token_lcp(ids_a, ids_b)
        out.append({
            "dataset": dataset,
            "request_id_a": a["request_id"], "request_id_b": b["request_id"],
            "tokens_b": len(ids_b), "token_lcp": lcp,
            "reuse_ratio": round((lcp // block_size) * block_size / len(ids_b), 4),
        })
    return out


# ------------------------------------------------------------------------ main

def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--model", default=kt.MODEL_ID)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    renderer = kt.Renderer(AutoTokenizer.from_pretrained(args.model))

    all_audit, all_blocks, all_cross, all_hoist = [], [], [], []
    for name, path in DATASETS.items():
        rows = kt.load_rows(path)
        sessions = kt.build_sessions(rows)
        all_audit += audit_pairs(rows, sessions, name)
        all_blocks += block_sensitivity(name, args.out)
        all_cross += cross_session_lcp(rows, sessions, name, renderer)
        all_hoist += cross_session_hoisted(rows, sessions, name, renderer)

    write_csv(args.out / "audit_pairs.csv", all_audit, AUDIT_FIELDS)
    write_csv(args.out / "block_size_sensitivity.csv", all_blocks)
    write_csv(args.out / "cross_session_lcp.csv", all_cross)
    write_csv(args.out / "cross_session_hoisted.csv", all_hoist)

    # ---- console report
    print(f"=== 1. pair audit ({len(all_audit)} pairs) ===")
    flags = {
        "strict_prefix_ok=False": sum(1 for r in all_audit if not r["strict_prefix_ok"]),
        "system_hash_equal=False": sum(1 for r in all_audit if not r["system_hash_equal"]),
        "model_equal=False": sum(1 for r in all_audit if not r["model_equal"]),
        "toolset_equal=False": sum(1 for r in all_audit if not r["toolset_equal"]),
        "prev_has_siblings=True": sum(1 for r in all_audit if r["prev_has_siblings"]),
        "body_is_duplicate=True": sum(1 for r in all_audit if r["body_is_duplicate"]),
    }
    for k, v in flags.items():
        print(f"  {'FLAG' if v else ' ok '} {k}: {v}")
    dts = [r["dt_seconds"] for r in all_audit]
    print(f"  dt_seconds: min={min(dts):.1f} median={median(dts):.1f} max={max(dts):.1f}")
    print(f"  appended roles: {dict((k, sum(1 for r in all_audit if r['appended_roles'] == k)) for k in {r['appended_roles'] for r in all_audit})}")

    print("\n=== 2. block-size sensitivity (median reuse ratio) ===")
    print(f"  {'dataset':<16}{'bs':>4}  " + "".join(f"{p:>15}" for p in kt.POLICIES))
    for ds in DATASETS:
        for bs in BLOCK_SIZES:
            cells = [next((r["median_reuse_ratio"] for r in all_blocks
                           if r["dataset"] == ds and r["block_size"] == bs
                           and r["policy"] == p), None) for p in kt.POLICIES]
            print(f"  {ds:<16}{bs:>4}  " + "".join(f"{c:>15.4f}" for c in cells))

    print("\n=== 3. cross-session prefix reuse (first request of each session) ===")
    for ds in DATASETS:
        rs = [r for r in all_cross if r["dataset"] == ds]
        if not rs:
            continue
        segs = {}
        for r in rs:
            segs[r["first_changed_segment"]] = segs.get(r["first_changed_segment"], 0) + 1
        print(f"  {ds}: n={len(rs)} median_lcp={median([r['token_lcp'] for r in rs]):.0f} "
              f"median_reuse={median([r['reuse_ratio'] for r in rs]):.4f} "
              f"same_toolset={sum(r['same_toolset'] for r in rs)}/{len(rs)} segs={segs}")
    print("\n=== 4. cross-session reuse if the schema block is hoisted to the front ===")
    for ds in DATASETS:
        base = [r for r in all_cross if r["dataset"] == ds]
        hoi = {(r["request_id_a"], r["request_id_b"]): r
               for r in all_hoist if r["dataset"] == ds}
        if not base:
            continue
        pairs = [(b, hoi[(b["request_id_a"], b["request_id_b"])]) for b in base]
        b_med = median([p[0]["reuse_ratio"] for p in pairs])
        h_med = median([p[1]["reuse_ratio"] for p in pairs])
        better = sum(1 for p in pairs if p[1]["token_lcp"] > p[0]["token_lcp"])
        print(f"  {ds}: n={len(pairs)}  as-is={b_med:.4f} -> hoisted={h_med:.4f} "
              f"({(h_med - b_med) * 100:+.1f} pp)  improved {better}/{len(pairs)} pairs "
              f"| median LCP {median([p[0]['token_lcp'] for p in pairs]):.0f} -> "
              f"{median([p[1]['token_lcp'] for p in pairs]):.0f} tokens")

    print(f"\nwrote -> {args.out}/")


if __name__ == "__main__":
    main()
