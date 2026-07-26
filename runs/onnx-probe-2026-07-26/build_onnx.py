"""Export the deployed BERT ranker to ONNX (fp32) and dynamic int8.

The exported graph must reproduce BertPredictor's forward exactly: the same
checkpoint, the same three encoder inputs, one logit out. Scoring
(``sigmoid(logit)``) stays outside the graph, where the predictor already does
it, so the ONNX arm is a drop-in swap for the ``self._model(**inputs).logits``
call and nothing else.

Builds every variant the probe measured, so the numbers can be reproduced from
this script alone:

* ``ranker-fp32.onnx``            - default (SDPA) export
* ``ranker-fp32-fused.onnx``      - ORT transformer fusion over that export
* ``ranker-int8.onnx``            - per-tensor dynamic int8
* ``ranker-int8-perchannel.onnx`` - per-channel dynamic int8
* ``ranker-int8-fused.onnx``      - per-channel int8 over the fused graph
* ``ranker-fp32-eager*.onnx``     - eager-attention export, unfused and fused
* ``ranker-int8-eager-fused.onnx``

The eager export exists because ORT's BERT ``Attention`` fusion matches none of
the SDPA-exported graph; exporting with ``attn_implementation="eager"`` is the
usual fix. It is measured, not assumed, in variant_sweep.py.

All outputs are model binaries and are not committed (.gitignore covers
``*.onnx``).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers import optimizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECKPOINT = REPO / "checkpoints_best_predictor"
FP32 = HERE / "ranker-fp32.onnx"
INT8 = HERE / "ranker-int8.onnx"
MAX_LENGTH = 512  # BertPredictor.MAX_LENGTH
OPSET = 17
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")
NUM_HEADS = 12
HIDDEN_SIZE = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _LogitOnly(torch.nn.Module):
    """Strip the HF output dataclass so the graph has one tensor output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits


def export_fp32(destination: Path, *, attn_implementation: str | None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    kwargs = {} if attn_implementation is None else {
        "attn_implementation": attn_implementation
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, local_files_only=True, **kwargs
    )
    model.eval()

    # A real row shape, not ones(): the export traces control flow on it.
    sample = tokenizer(
        ["[USER]\nexport trace row\n[TOOLS]\n{}"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    args = tuple(sample[name] for name in INPUT_NAMES)

    with torch.inference_mode():
        torch.onnx.export(
            _LogitOnly(model),
            args,
            str(destination),
            input_names=list(INPUT_NAMES),
            output_names=["logits"],
            dynamic_axes={
                name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES
            }
            | {"logits": {0: "batch"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )


def fuse(source: Path, destination: Path) -> dict[str, int]:
    fused = optimizer.optimize_model(
        str(source),
        model_type="bert",
        num_heads=NUM_HEADS,
        hidden_size=HIDDEN_SIZE,
        opt_level=1,
    )
    fused.save_model_to_file(str(destination))
    statistics = fused.get_fused_operator_statistics()
    return {name: count for name, count in statistics.items() if count}


def quantize(source: Path, destination: Path, *, per_channel: bool) -> None:
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=per_channel,
        # Fused graphs carry ORT-domain nodes whose output types shape
        # inference cannot resolve; the quantizer needs the default spelled out.
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )


def main() -> None:
    started = time.time()
    fusion_stats: dict[str, dict[str, int]] = {}

    export_fp32(FP32, attn_implementation=None)
    fusion_stats["fp32_fused"] = fuse(FP32, HERE / "ranker-fp32-fused.onnx")
    quantize(FP32, INT8, per_channel=False)
    quantize(FP32, HERE / "ranker-int8-perchannel.onnx", per_channel=True)
    quantize(
        HERE / "ranker-fp32-fused.onnx",
        HERE / "ranker-int8-fused.onnx",
        per_channel=True,
    )
    print(f"sdpa export + variants in {time.time() - started:.1f}s", flush=True)

    eager_started = time.time()
    eager = HERE / "ranker-fp32-eager.onnx"
    export_fp32(eager, attn_implementation="eager")
    fusion_stats["fp32_eager_fused"] = fuse(
        eager, HERE / "ranker-fp32-eager-fused.onnx"
    )
    quantize(
        HERE / "ranker-fp32-eager-fused.onnx",
        HERE / "ranker-int8-eager-fused.onnx",
        per_channel=True,
    )
    print(f"eager export + variants in {time.time() - eager_started:.1f}s", flush=True)

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(HERE.glob("*.onnx"))
    }
    manifest = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_safetensors_sha256": sha256_file(
            CHECKPOINT / "model.safetensors"
        ),
        "opset": OPSET,
        "max_length": MAX_LENGTH,
        "input_names": list(INPUT_NAMES),
        "artifacts": artifacts,
        "fusion_statistics": fusion_stats,
        "fusion_note": "ORT's BERT Attention fusion matched zero patterns on "
        "both the SDPA and the eager export under transformers 5.14.1 / "
        "onnxruntime 1.28.0; only BiasGelu and SkipLayerNormalization fused.",
        "torch": torch.__version__,
        "wall_clock_seconds": time.time() - started,
    }
    (HERE / "build-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"artifacts": artifacts, "fusion": fusion_stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
