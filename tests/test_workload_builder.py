from ltr_training.label_input import LabelInput
from ltr_training.workload_builder import build_workload, manifest_split_ids
from scheduler_benchmark.runner import WorkloadRequest, select_workload_profile


def _item(sample_id: str, source: str) -> LabelInput:
    return LabelInput(
        sample_id=sample_id,
        request_id=sample_id,
        prompt=f"prompt {sample_id}",
        tool_schema="[]",
        history=(("human", "prior"),),
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
    assert {row["category"] for row in rows} == {"id:toolace", "ood:bfcl"}
    assert all(row["prompt"] == f"prompt {row['request_id']}" for row in rows)
    assert all(row["tool_schema"] == "[]" for row in rows)
    assert all(row["history"] == [["human", "prior"]] for row in rows)
    assert manifest["schema_version"] == "offline-workload-v2"
    assert manifest["baseline_service_ms"]["kind"] == "output_length_per_token_proxy"
    assert manifest["baseline_service_ms"]["per_token_ms"] == 2.5
    assert manifest["runner_compatibility"] == "unverified_scheduler_benchmark_frozen"


def test_mixed_ratio_keeps_exact_preselected_pool_despite_float_roundoff() -> None:
    inputs = [_item("id-1", "toolace")]
    oods = [_item("ood-1", "bfcl"), _item("ood-2", "toolathlon")]

    rows, _ = build_workload(
        id_inputs=inputs,
        ood_inputs=oods,
        lengths={"id-1": 1, "ood-1": 1, "ood-2": 1},
        profile="mixed",
        per_token_ms=2.5,
        ood_ratio=2 / 3,
        seed=42,
    )

    assert {row["request_id"] for row in rows} == {"id-1", "ood-1", "ood-2"}


def test_tier2_manifest_selects_only_declared_test_sample_ids() -> None:
    payload = {
        "samples": [
            {"sample_id": "train-1", "split": "train"},
            {"sample_id": "test-1", "split": "test"},
        ]
    }

    assert manifest_split_ids(payload, split="test") == {"test-1"}


def test_workload_categories_are_selectable_by_runner_in_both_directions() -> None:
    rows, _ = build_workload(
        id_inputs=[_item("id-1", "toolace")],
        ood_inputs=[_item("ood-1", "bfcl")],
        lengths={"id-1": 20, "ood-1": 10},
        profile="mixed",
        per_token_ms=2.5,
        seed=17,
    )
    workload = [
        WorkloadRequest(
            request_id=str(row["request_id"]),
            prompt=str(row["prompt"]),
            tool_schema=str(row["tool_schema"]),
            history=[list(item) for item in row["history"]],
            baseline_service_ms=float(row["baseline_service_ms"]),
            category=str(row["category"]),
        )
        for row in rows
    ]

    assert [row.request_id for row in select_workload_profile(workload, "id")] == [
        "id-1"
    ]
    assert [row.request_id for row in select_workload_profile(workload, "ood")] == [
        "ood-1"
    ]


def test_manifest_split_ids_reads_real_sample_ids_shape() -> None:
    # 真实 tier2-sample-manifest.json 的形状:sample_ids: {split: [ids]}
    payload = {
        "sample_count": 3,
        "sample_ids": {
            "test": ["toolace-000001:0000", "toolace-000002:0000"],
            "train": ["toolace-000003:0000"],
        },
    }
    assert manifest_split_ids(payload, split="test") == {
        "toolace-000001:0000",
        "toolace-000002:0000",
    }
    assert manifest_split_ids(payload, split="train") == {"toolace-000003:0000"}
