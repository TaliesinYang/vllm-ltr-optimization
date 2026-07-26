"""Parity + CPU latency for torch vs ONNX fp32 vs ONNX dynamic-int8.

Every arm runs the deployed forward exactly as ``BertPredictor.predict`` does:
one row at a time, ``[USER]\\n{prompt}\\n[TOOLS]\\n{tool_schema}`` rendered by
``ltr_training.train_ranker.render_example``, truncation at 512, and
``score = sigmoid(logit)``. Only the tensor backend differs.

Latency is measured in-process, not over HTTP, so these rows are comparable to
each other but sit below the e4 numbers by whatever the decision service's HTTP
overhead costs; the torch arm here is the control that quantifies that gap.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from scipy.stats import kendalltau
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "runs" / "offline-experiments-2026-07-25"))
sys.path.insert(0, str(REPO))

import common  # noqa: E402

from ltr_training.train_ranker import render_example  # noqa: E402

CHECKPOINT = REPO / "checkpoints_best_predictor"
FP32 = HERE / "ranker-fp32.onnx"
INT8 = HERE / "ranker-int8.onnx"
OUT = HERE / "onnx-probe.json"
# The plain export is the naive answer; the eager+fused pair is the fastest
# configuration variant_sweep.py found, so both are carried at full protocol.
ONNX_ARMS = (
    ("onnx_fp32", "ranker-fp32.onnx"),
    ("onnx_fp32_eager_fused", "ranker-fp32-eager-fused.onnx"),
    ("onnx_int8", "ranker-int8.onnx"),
    ("onnx_int8_eager_fused", "ranker-int8-eager-fused.onnx"),
)
REFERENCE_SCORES = (
    REPO / "runs" / "offline-experiments-2026-07-25" / "e2-bert-test-scores.jsonl"
)

VARIANT = "prompt_schema"
MAX_LENGTH = 512  # BertPredictor.MAX_LENGTH
THREADS = 2  # LTR_DECISION_TORCH_THREADS default
CONCURRENCY = 8
WARMUPS = 20
SAMPLES = 200
LATENCY_ROWS = 64  # same slice e4_latency.py measured
PARITY_TAU_BAR = 0.999


class TorchArm:
    """The shipped scoring path, with a per-thread tokenizer (see e4's note)."""

    name = "torch"

    def __init__(self) -> None:
        torch.set_num_threads(THREADS)
        self._local = threading.local()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT, local_files_only=True
        )
        self._model.eval()

    @property
    def tokenizer(self):
        tokenizer = getattr(self._local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                CHECKPOINT, local_files_only=True
            )
            self._local.tokenizer = tokenizer
        return tokenizer

    def score(self, text: str) -> float:
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self._model(**inputs).logits.reshape(-1)
        if logits.numel() != 1:
            raise ValueError("predictor must return exactly one logit")
        return float(torch.sigmoid(logits[0]).item())


class OnnxArm:
    def __init__(self, name: str, model_path: Path) -> None:
        self.name = name
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._local = threading.local()

    @property
    def tokenizer(self):
        tokenizer = getattr(self._local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                CHECKPOINT, local_files_only=True
            )
            self._local.tokenizer = tokenizer
        return tokenizer

    def score(self, text: str) -> float:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        feeds = {
            name: encoded[name].astype(np.int64)
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }
        logits = self._session.run(["logits"], feeds)[0].reshape(-1)
        if logits.size != 1:
            raise ValueError("predictor must return exactly one logit")
        return float(1.0 / (1.0 + math.exp(-float(logits[0]))))


def percentile(ordered: list[float], fraction: float) -> float:
    """Same estimator e4_latency.py used, so the rows stay comparable."""
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def measure(arm, texts: list[str], *, concurrency: int) -> dict[str, object]:
    def one(index: int) -> float:
        started = time.perf_counter()
        arm.score(texts[index % len(texts)])
        return (time.perf_counter() - started) * 1000.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(WARMUPS)))
        started = time.time()
        samples = list(pool.map(one, range(SAMPLES)))
    wall = time.time() - started
    ordered = sorted(samples)
    return {
        "concurrency": concurrency,
        "warmup_count": WARMUPS,
        "sample_count": len(samples),
        "wall_clock_seconds": wall,
        "throughput_rps": len(samples) / wall if wall else 0.0,
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def load_reference_logits(test_examples) -> list[float] | None:
    """Saved seed-17 test logits from E2, batched at 32 rather than one by one."""
    if not REFERENCE_SCORES.exists():
        return None
    rows = {}
    with REFERENCE_SCORES.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["sample_id"]] = row
    key = f"bert_{VARIANT}_seed17"
    try:
        return [float(rows[item.sample_id][key]) for item in test_examples]
    except KeyError:
        return None


def delta_stats(reference: list[float], other: list[float]) -> dict[str, float]:
    deltas = [abs(a - b) for a, b in zip(reference, other)]
    tau = kendalltau(reference, other)
    return {
        "max_abs_delta": max(deltas),
        "mean_abs_delta": statistics.fmean(deltas),
        "kendall_tau": float(tau.statistic),
        "tau_note": "tau computed over sigmoid scores; sigmoid is monotonic so "
        "the value is identical in logit space",
        "rows": len(reference),
    }


def main() -> None:
    started = time.time()
    inputs = common.verify_inputs()
    splits, _ = common.load_splits()
    test = splits["test"]
    texts = [render_example(item, variant=VARIANT) for item in test]
    print(f"test rows: {len(test)}", flush=True)

    arms = [TorchArm()]
    arms.extend(OnnxArm(name, HERE / filename) for name, filename in ONNX_ARMS)

    # --- parity -----------------------------------------------------------
    scores: dict[str, list[float]] = {}
    for arm in arms:
        elapsed = time.time()
        scores[arm.name] = [arm.score(text) for text in texts]
        print(
            f"scored {len(texts)} rows on {arm.name} in {time.time() - elapsed:.1f}s",
            flush=True,
        )

    parity = {
        name: delta_stats(scores["torch"], scores[name])
        for name, _ in ONNX_ARMS
    }
    reference_logits = load_reference_logits(test)
    if reference_logits is not None:
        reference_scores = [1.0 / (1.0 + math.exp(-value)) for value in reference_logits]
        parity["torch_vs_saved_e2_reference"] = delta_stats(
            reference_scores, scores["torch"]
        )

    # --- latency ----------------------------------------------------------
    latency_texts = texts[:LATENCY_ROWS]
    latency: dict[str, dict[str, object]] = {}
    for arm in arms:
        for concurrency in (1, CONCURRENCY):
            key = f"{arm.name}_conc{concurrency}"
            latency[key] = measure(arm, latency_texts, concurrency=concurrency)
            payload = latency[key]
            print(
                f"{key:22s} p50={payload['p50_ms']:7.1f}ms "
                f"p99={payload['p99_ms']:7.1f}ms "
                f"rps={payload['throughput_rps']:6.1f}",
                flush=True,
            )

    report = {
        "schema_version": "onnx-probe-v1",
        "status": "done",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_note": "sha256-identical to "
        "tier2-matrix/bert-prompt_schema-tier2-seed17/final",
        "inputs": inputs,
        "hardware": "Apple M4, 10 cores (4P/6E), macOS; intra-op threads=2 for "
        "every arm (LTR_DECISION_TORCH_THREADS default)",
        "protocol": {
            "parity_rows": len(test),
            "parity_note": "full tier-2 test split, scored one row at a time "
            "exactly as BertPredictor.predict does (padding=True, truncation, "
            "max_length=512, score=sigmoid(logit))",
            "latency_note": "in-process predictor calls (tokenize + forward + "
            "sigmoid), NOT over HTTP; e4-latency.json measured the same work "
            "through the decision service, so its rows include HTTP overhead",
            "distinct_requests": len(latency_texts),
            "warmups": WARMUPS,
            "samples": SAMPLES,
            "threads": THREADS,
        },
        "parity_bar": {"kendall_tau_min": PARITY_TAU_BAR},
        "parity": parity,
        "latency": {
            name: {k: v for k, v in payload.items() if k != "samples_ms"}
            for name, payload in latency.items()
        },
        "raw_samples_ms": {
            name: payload["samples_ms"] for name, payload in latency.items()
        },
        "artifact_bytes": {
            "torch_safetensors": (CHECKPOINT / "model.safetensors").stat().st_size,
            **{
                name: (HERE / filename).stat().st_size
                for name, filename in ONNX_ARMS
            },
        },
        "parity_verdict": {
            name: (
                "pass"
                if parity[name]["kendall_tau"] >= PARITY_TAU_BAR
                else "FAIL"
            )
            for name, _ in ONNX_ARMS
        },
        "versions": {
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "wall_clock_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(parity, indent=2, sort_keys=True))
    print(f"\ndone in {report['wall_clock_seconds']:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
