from ltr_training.label_input import LabelInput
from ltr_training.workload_builder import build_workload, manifest_split_ids


def _item(sample_id: str, source: str) -> LabelInput:
    return LabelInput(
        sample_id=sample_id,
        request_id=sample_id,
        prompt=f"prompt {sample_id}",
        tool_schema="[]",
        history=(),
        session_id=sample_id,
        task_id=sample_id,
        source=source,
        source_revision="a" * 40,
        category="category",
    )


def test_workload_uses_output_length_proxy_and_explicit_4096_limit() -> None:
    rows, manifest = build_workload(
        id_inputs=[_item("id-1", "toolace")],
        ood_inputs=[_item("ood-1", "bfcl")],
        lengths={"id-1": 20, "ood-1": 10},
        profile="mixed",
        per_token_ms=2.5,
        ood_ratio=0.5,
        seed=17,
    )

    assert {row["request_id"] for row in rows} == {"id-1", "ood-1"}
    assert {row["baseline_service_ms"] for row in rows} == {50.0, 25.0}
    assert all(row["max_tokens"] == 4096 for row in rows)
    assert {row["category"] for row in rows} == {"id/toolace", "ood/bfcl"}
    assert manifest["baseline_service_ms"]["kind"] == "output_length_per_token_proxy"
    assert manifest["baseline_service_ms"]["per_token_ms"] == 2.5
    assert manifest["runner_compatibility"] == "unverified_scheduler_benchmark_frozen"


def test_tier2_manifest_selects_only_declared_test_sample_ids() -> None:
    payload = {
        "samples": [
            {"sample_id": "train-1", "split": "train"},
            {"sample_id": "test-1", "split": "test"},
        ]
    }

    assert manifest_split_ids(payload, split="test") == {"test-1"}
