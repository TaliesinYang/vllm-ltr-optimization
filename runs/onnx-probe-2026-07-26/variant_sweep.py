"""Second pass: does a BERT-fused graph or per-channel int8 change the verdict?

The first pass exported a plain graph and quantized it per-tensor. ONNX Runtime
came out slower than torch and per-tensor int8 missed the parity bar, so this
sweep checks the two obvious levers before the arm is written off: the
transformer fusion pass, and per-channel weight scales.

Short protocol on purpose (10 warm-ups, 60 samples, concurrency 1) - it is a
direction check, not a publishable latency row.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from onnx_probe import (  # noqa: E402
    OnnxArm,
    TorchArm,
    VARIANT,
    common,
    percentile,
    render_example,
)

WARMUPS = 10
SWEEP_SAMPLES = 60
PARITY_ROWS = 300
OUT = HERE / "onnx-variant-sweep.json"

VARIANTS = {
    "fp32_plain": "ranker-fp32.onnx",
    "fp32_fused": "ranker-fp32-fused.onnx",
    "fp32_eager": "ranker-fp32-eager.onnx",
    "fp32_eager_fused": "ranker-fp32-eager-fused.onnx",
    "int8_per_tensor": "ranker-int8.onnx",
    "int8_per_channel": "ranker-int8-perchannel.onnx",
    "int8_fused_per_channel": "ranker-int8-fused.onnx",
    "int8_eager_fused_per_channel": "ranker-int8-eager-fused.onnx",
}


def serial_latency(arm, texts: list[str]) -> dict[str, float]:
    for index in range(WARMUPS):
        arm.score(texts[index % len(texts)])
    samples = []
    for index in range(SWEEP_SAMPLES):
        started = time.perf_counter()
        arm.score(texts[index % len(texts)])
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return {
        "p50_ms": percentile(ordered, 0.50),
        "p99_ms": percentile(ordered, 0.99),
        "mean_ms": statistics.fmean(samples),
        "samples": len(samples),
    }


def main() -> None:
    started = time.time()
    splits, _ = common.load_splits()
    texts = [render_example(item, variant=VARIANT) for item in splits["test"]]
    parity_texts = texts[:PARITY_ROWS]
    latency_texts = texts[:64]

    torch_arm = TorchArm()
    torch_scores = [torch_arm.score(text) for text in parity_texts]
    results: dict[str, object] = {
        "torch": {
            "latency": serial_latency(torch_arm, latency_texts),
            "bytes": None,
        }
    }
    print(f"torch {results['torch']['latency']}", flush=True)

    for name, filename in VARIANTS.items():
        path = HERE / filename
        arm = OnnxArm(name, path)
        scores = [arm.score(text) for text in parity_texts]
        deltas = [abs(a - b) for a, b in zip(torch_scores, scores)]
        row = {
            "file": filename,
            "bytes": path.stat().st_size,
            "kendall_tau_vs_torch": float(
                kendalltau(torch_scores, scores).statistic
            ),
            "max_abs_delta": max(deltas),
            "mean_abs_delta": statistics.fmean(deltas),
            "latency": serial_latency(arm, latency_texts),
        }
        results[name] = row
        print(
            f"{name:24s} tau={row['kendall_tau_vs_torch']:.6f} "
            f"maxd={row['max_abs_delta']:.2e} "
            f"p50={row['latency']['p50_ms']:7.1f}ms "
            f"p99={row['latency']['p99_ms']:7.1f}ms",
            flush=True,
        )

    report = {
        "schema_version": "onnx-variant-sweep-v1",
        "note": "direction check only: 10 warm-ups, 60 samples, concurrency 1, "
        f"parity over the first {PARITY_ROWS} tier-2 test rows",
        "results": results,
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\ndone in {report['wall_clock_seconds']:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
