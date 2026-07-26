"""T9 GPU validation: the ported BertPredictor on real CUDA, through real HTTP.

Confirms the scheduler_benchmark port reproduces the prototype's batching
behaviour on the 201 box's 4090, at the unchanged /v1/decision contract.
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/alex/ltr-seam-t9")

import torch

from scheduler_benchmark.decision_service import DecisionApplication, DecisionHTTPServer
from scheduler_benchmark.predictor import BertPredictor

CKPT = Path("/home/alex/ltr-seam/checkpoints_best_predictor")
ROWS = Path("/home/alex/ltr-seam/e4_rows64.json")
OUT = Path("/home/alex/ltr-seam-t9/t9-gpu-smoke.json")
CONCURRENCY, WARMUPS, SAMPLES = 8, 20, 200

rows = json.loads(ROWS.read_text())
print("cuda available:", torch.cuda.is_available(), flush=True)


def serve(predictor, revision):
    app = DecisionApplication(
        predictor=predictor,
        predictor_revision=revision,
        feature_variant="prompt_schema",
        max_concurrency=CONCURRENCY,
        reliability_threshold=0.5,
    )
    server = DecisionHTTPServer(("127.0.0.1", 0), app, 8 * 1024 * 1024)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{server.server_address[0]}:{server.server_address[1]}"


def call(endpoint, row):
    rid = f"t9-{uuid.uuid4().hex}"
    payload = {
        "schema_version": "1.0",
        "request_id": rid,
        "decision_id": f"dec-{rid}",
        "model_id": "qwen3.5-9b",
        "request_age_ms": 0,
        "messages": [
            {"role": "system", "content": row["tool_schema"]},
            {"role": "user", "content": row["prompt"]},
        ],
        "tool_schema_text": row["tool_schema"],
        "tools": [{"type": "function", "function": {"name": "probe"}}],
        "workflow_id": rid,
        "step_id": "0",
        "conversation_id": rid,
        "previous_tool_gap_ms": 0,
        "generation_controls": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 4096,
        },
    }
    req = urllib.request.Request(
        endpoint + "/v1/decision",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as response:
        body = json.load(response)
    return (time.perf_counter() - started) * 1000.0, body


def measure(endpoint):
    def one(i):
        return call(endpoint, rows[i % len(rows)])[0]

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(one, range(WARMUPS)))
        started = time.time()
        samples = list(pool.map(one, range(SAMPLES)))
    wall = time.time() - started
    ordered = sorted(samples)
    return {
        "sample_count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[int(0.95 * len(ordered)) - 1],
        "p99_ms": ordered[int(0.99 * len(ordered)) - 1],
        "wall_clock_seconds": wall,
        "throughput_rps": len(samples) / wall,
    }


results = {}
for label, batch_max, window in (("gpu_batched", 8, 3.0), ("gpu_unbatched", 1, 0.0)):
    predictor = BertPredictor(
        CKPT, device="cuda", batch_max=batch_max, batch_window_ms=window
    )
    server, endpoint = serve(predictor, f"t9-{label}")
    row = measure(endpoint)
    if predictor.batcher is not None:
        sizes = predictor.batcher.batch_sizes
        row["observed_batch_size_mean"] = statistics.fmean(sizes)
        row["observed_batch_size_max"] = max(sizes)
        row["forward_passes"] = len(sizes)
    row["device"] = predictor.device
    results[label] = row
    print(label, json.dumps(row), flush=True)
    server.shutdown()
    predictor.close()

# Gate composition on real hardware: an abstained request must not reach CUDA.
predictor = BertPredictor(CKPT, device="cuda", batch_max=8, batch_window_ms=3.0)
server, endpoint = serve(predictor, "t9-gate")
seen_schema = rows[0]["tool_schema"]
_, body = call(endpoint, rows[0])
results["gate_composition"] = {
    "reason_code": body["reason_code"],
    "reliability_probability": body["reliability_probability"],
    "forward_passes_after_request": len(predictor.batcher.batch_sizes),
}
print("gate:", json.dumps(results["gate_composition"]), flush=True)
server.shutdown()
predictor.close()

OUT.write_text(
    json.dumps(
        {
            "schema_version": "t9-gpu-smoke-v1",
            "host": "192.168.8.201 (WSL2, RTX 4090 Laptop 16GB)",
            "contract": "/v1/decision unchanged",
            "protocol": f"concurrency {CONCURRENCY}, {WARMUPS} warm, {SAMPLES} samples",
            "results": results,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print("wrote", OUT)
