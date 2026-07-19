import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/server/build_server_workloads.py"
BUDGET = ROOT / "scripts/server/compute_rental_budget.sh"


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label(sample_id: str, source: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "request_id": sample_id,
        "prompt": f"prompt {sample_id}",
        "tool_schema": "[]",
        "history": [],
        "session_id": sample_id,
        "task_id": sample_id,
        "source": source,
        "source_revision": "revision",
        "category": "fixture",
    }


def test_materialize_source_uses_only_pinned_snapshot_data_files(tmp_path: Path) -> None:
    bfcl = tmp_path / "bfcl"
    bfcl.mkdir()
    _write_jsonl(bfcl / "BFCL_v3_simple.json", [{"id": "simple_1"}])
    _write_jsonl(bfcl / "BFCL_v3_parallel.json", [{"id": "parallel_1"}])
    (bfcl / "possible_answer").mkdir()
    _write_jsonl(
        bfcl / "possible_answer/BFCL_v3_simple.json", [{"id": "answer"}]
    )
    bfcl_output = tmp_path / "bfcl-raw.jsonl"

    result = _run(
        "materialize-source",
        "--source",
        "bfcl",
        "--snapshot",
        bfcl,
        "--output",
        bfcl_output,
    )

    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["id"] for line in bfcl_output.read_text().splitlines()] == [
        "parallel_1",
        "simple_1",
    ]


def test_merge_lengths_is_latest_wins_and_rejects_incomplete_ledgers(
    tmp_path: Path,
) -> None:
    id_input = tmp_path / "id.jsonl"
    ood_input = tmp_path / "ood.jsonl"
    id_ledger = tmp_path / "id-ledger.jsonl"
    ood_ledger = tmp_path / "ood-ledger.jsonl"
    output = tmp_path / "combined-lengths.jsonl"
    _write_jsonl(id_input, [_label("id-1", "toolace"), _label("id-2", "toolace")])
    _write_jsonl(ood_input, [_label("ood-1", "bfcl")])
    _write_jsonl(
        id_ledger,
        [
            {"sample_id": "id-1", "status": "error"},
            {"sample_id": "id-1", "status": "ok", "output_length": 11},
            {"sample_id": "id-2", "status": "ok", "output_length": 12},
        ],
    )
    _write_jsonl(
        ood_ledger,
        [{"sample_id": "ood-1", "status": "ok", "output_length": 13}],
    )

    result = _run(
        "merge-lengths",
        "--id-input",
        id_input,
        "--id-ledger",
        id_ledger,
        "--ood-input",
        ood_input,
        "--ood-ledger",
        ood_ledger,
        "--output",
        output,
    )

    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"output_length": 11, "sample_id": "id-1", "source": "toolace"},
        {"output_length": 12, "sample_id": "id-2", "source": "toolace"},
        {"output_length": 13, "sample_id": "ood-1", "source": "bfcl"},
    ]

    _write_jsonl(ood_ledger, [{"sample_id": "ood-1", "status": "error"}])
    failed = _run(
        "merge-lengths",
        "--id-input",
        id_input,
        "--id-ledger",
        id_ledger,
        "--ood-input",
        ood_input,
        "--ood-ledger",
        ood_ledger,
        "--output",
        output,
    )
    assert failed.returncode == 1
    assert "NO-GO" in failed.stderr


def test_combine_label_inputs_requires_exact_source_counts_and_hashes(
    tmp_path: Path,
) -> None:
    bfcl = tmp_path / "bfcl.jsonl"
    toolathlon = tmp_path / "toolathlon.jsonl"
    bfcl_manifest = tmp_path / "bfcl-manifest.json"
    toolathlon_manifest = tmp_path / "toolathlon-manifest.json"
    output = tmp_path / "ood-label-inputs.jsonl"
    _write_jsonl(bfcl, [_label("bfcl-1", "bfcl")])
    _write_jsonl(toolathlon, [_label("toolathlon-1", "toolathlon")])
    for source, path, manifest in (
        ("bfcl", bfcl, bfcl_manifest),
        ("toolathlon", toolathlon, toolathlon_manifest),
    ):
        manifest.write_text(
            json.dumps(
                {
                    "source": source,
                    "row_count": 1,
                    "sample_size_requested": 1,
                    "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    result = _run(
        "combine-label-inputs",
        "--bfcl-input",
        bfcl,
        "--bfcl-manifest",
        bfcl_manifest,
        "--toolathlon-input",
        toolathlon,
        "--toolathlon-manifest",
        toolathlon_manifest,
        "--expected-per-source",
        "1",
        "--output",
        output,
    )

    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["source"] for line in output.read_text().splitlines()] == [
        "bfcl",
        "toolathlon",
    ]

    payload = json.loads(bfcl_manifest.read_text())
    payload["row_count"] = 0
    bfcl_manifest.write_text(json.dumps(payload), encoding="utf-8")
    failed = _run(
        "combine-label-inputs",
        "--bfcl-input",
        bfcl,
        "--bfcl-manifest",
        bfcl_manifest,
        "--toolathlon-input",
        toolathlon,
        "--toolathlon-manifest",
        toolathlon_manifest,
        "--expected-per-source",
        "1",
        "--output",
        output,
    )
    assert failed.returncode == 1
    assert "NO-GO" in failed.stderr


def test_workload_input_subsampling_is_deterministic_and_records_pool_sizes(
    tmp_path: Path,
) -> None:
    id_input = tmp_path / "id-pool.jsonl"
    id_manifest = tmp_path / "id-manifest.json"
    ood_input = tmp_path / "ood-pool.jsonl"
    _write_jsonl(
        id_input,
        [_label(f"id-{index:02d}", "toolace") for index in range(10)],
    )
    id_manifest.write_text(
        json.dumps(
            {
                "sample_ids": {
                    "test": [f"id-{index:02d}" for index in range(10)]
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        ood_input,
        [_label(f"ood-{index:02d}", "bfcl") for index in range(8)],
    )

    first = {
        "mixed_id": tmp_path / "mixed-id-1.jsonl",
        "mixed_ood": tmp_path / "mixed-ood-1.jsonl",
        "ood": tmp_path / "ood-1.jsonl",
        "manifest": tmp_path / "selection-1.json",
    }
    second = {
        "mixed_id": tmp_path / "mixed-id-2.jsonl",
        "mixed_ood": tmp_path / "mixed-ood-2.jsonl",
        "ood": tmp_path / "ood-2.jsonl",
        "manifest": tmp_path / "selection-2.json",
    }

    def select(outputs: dict[str, Path]) -> subprocess.CompletedProcess[str]:
        return _run(
            "select-workload-inputs",
            "--id-input",
            id_input,
            "--id-manifest",
            id_manifest,
            "--id-split",
            "test",
            "--ood-input",
            ood_input,
            "--expected-id-pool-size",
            "10",
            "--expected-ood-pool-size",
            "8",
            "--mixed-id-target",
            "3",
            "--mixed-ood-target",
            "2",
            "--ood-target",
            "4",
            "--seed",
            "42",
            "--mixed-id-output",
            outputs["mixed_id"],
            "--mixed-ood-output",
            outputs["mixed_ood"],
            "--ood-output",
            outputs["ood"],
            "--manifest",
            outputs["manifest"],
        )

    first_result = select(first)
    second_result = select(second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    for name in ("mixed_id", "mixed_ood", "ood"):
        assert first[name].read_bytes() == second[name].read_bytes()
    assert len(first["mixed_id"].read_text().splitlines()) == 3
    assert len(first["mixed_ood"].read_text().splitlines()) == 2
    assert len(first["ood"].read_text().splitlines()) == 4
    mixed_ood_ids = {
        json.loads(line)["sample_id"]
        for line in first["mixed_ood"].read_text().splitlines()
    }
    ood_ids = {
        json.loads(line)["sample_id"]
        for line in first["ood"].read_text().splitlines()
    }
    assert mixed_ood_ids < ood_ids
    manifest = json.loads(first["manifest"].read_text())
    assert manifest["sampling_seed"] == 42
    assert manifest["pool_sizes"] == {"id_test": 10, "ood": 8}
    assert manifest["selected_counts"] == {
        "mixed_id": 3,
        "mixed_ood": 2,
        "ood": 4,
    }
    assert len(ood_input.read_text().splitlines()) == 8

    wrong_pool = _run(
        "select-workload-inputs",
        "--id-input",
        id_input,
        "--id-manifest",
        id_manifest,
        "--id-split",
        "test",
        "--ood-input",
        ood_input,
        "--expected-id-pool-size",
        "11",
        "--expected-ood-pool-size",
        "8",
        "--mixed-id-target",
        "3",
        "--mixed-ood-target",
        "2",
        "--ood-target",
        "4",
        "--seed",
        "42",
        "--mixed-id-output",
        first["mixed_id"],
        "--mixed-ood-output",
        first["mixed_ood"],
        "--ood-output",
        first["ood"],
        "--manifest",
        first["manifest"],
    )
    assert wrong_pool.returncode == 1
    assert "pool size mismatch" in wrong_pool.stderr


def test_verify_workloads_checks_manifest_raw_sha_and_profile_rows(
    tmp_path: Path,
) -> None:
    mixed = tmp_path / "mixed.v2.jsonl"
    ood = tmp_path / "ood.v2.jsonl"
    mixed_manifest = tmp_path / "mixed.v2.manifest.json"
    ood_manifest = tmp_path / "ood.v2.manifest.json"
    id_input = tmp_path / "id-label-inputs.jsonl"
    id_manifest = tmp_path / "id-sample-manifest.json"
    combined_ood = tmp_path / "ood-label-inputs.jsonl"
    selected_mixed_id = tmp_path / "selected-mixed-id.jsonl"
    selected_mixed_ood = tmp_path / "selected-mixed-ood.jsonl"
    selected_ood = tmp_path / "selected-ood.jsonl"
    selection_manifest = tmp_path / "selection.json"
    lengths = tmp_path / "combined-lengths.jsonl"
    base = {
        "prompt": "p",
        "tool_schema": "[]",
        "history": [],
        "baseline_service_ms": 2.5,
        "max_tokens": 4096,
    }
    _write_jsonl(
        mixed,
        [
            {**base, "request_id": "id-1", "category": "id:toolace"},
            {**base, "request_id": "ood-1", "category": "ood:bfcl"},
        ],
    )
    _write_jsonl(
        ood, [{**base, "request_id": "ood-1", "category": "ood:bfcl"}]
    )
    _write_jsonl(id_input, [_label("id-1", "toolace")])
    id_manifest.write_text(
        json.dumps({"sample_ids": {"test": ["id-1"]}}), encoding="utf-8"
    )
    _write_jsonl(combined_ood, [_label("ood-1", "bfcl")])
    _write_jsonl(
        lengths,
        [
            {"sample_id": "id-1", "output_length": 1},
            {"sample_id": "ood-1", "output_length": 1},
        ],
    )
    for profile, workload, manifest in (
        ("mixed", mixed, mixed_manifest),
        ("ood", ood, ood_manifest),
    ):
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "offline-workload-v2",
                    "profile": profile,
                    "request_count": len(workload.read_text().splitlines()),
                    "output_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    selected = _run(
        "select-workload-inputs",
        "--id-input",
        id_input,
        "--id-manifest",
        id_manifest,
        "--id-split",
        "test",
        "--ood-input",
        combined_ood,
        "--mixed-id-target",
        "1",
        "--mixed-ood-target",
        "1",
        "--ood-target",
        "1",
        "--seed",
        "42",
        "--mixed-id-output",
        selected_mixed_id,
        "--mixed-ood-output",
        selected_mixed_ood,
        "--ood-output",
        selected_ood,
        "--manifest",
        selection_manifest,
    )
    assert selected.returncode == 0, selected.stderr
    annotated = _run(
        "annotate-workload-manifests",
        "--selection-manifest",
        selection_manifest,
        "--mixed-manifest",
        mixed_manifest,
        "--ood-manifest",
        ood_manifest,
    )
    assert annotated.returncode == 0, annotated.stderr
    mixed_provenance = json.loads(mixed_manifest.read_text())["workload_subsampling"]
    assert mixed_provenance["sampling_seed"] == 42
    assert mixed_provenance["pool_sizes"] == {"id_test": 1, "ood": 1}
    assert mixed_provenance["selected_counts"] == {"id": 1, "ood": 1}

    result = _run(
        "verify-workloads",
        "--mixed",
        mixed,
        "--mixed-manifest",
        mixed_manifest,
        "--ood",
        ood,
        "--ood-manifest",
        ood_manifest,
        "--selection-manifest",
        selection_manifest,
        "--lengths",
        lengths,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mixed_rows"] == 2
    assert report["ood_rows"] == 1

    original_id_input = id_input.read_bytes()
    with id_input.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_label("id-2", "toolace")) + "\n")
    stale_selection = _run(
        "verify-workloads",
        "--mixed",
        mixed,
        "--mixed-manifest",
        mixed_manifest,
        "--ood",
        ood,
        "--ood-manifest",
        ood_manifest,
        "--selection-manifest",
        selection_manifest,
        "--lengths",
        lengths,
    )
    assert stale_selection.returncode == 1
    assert "input SHA-256" in stale_selection.stderr
    id_input.write_bytes(original_id_input)

    mixed_manifest.write_text(
        mixed_manifest.read_text().replace(
            hashlib.sha256(mixed.read_bytes()).hexdigest(), "0" * 64
        ),
        encoding="utf-8",
    )
    failed = _run(
        "verify-workloads",
        "--mixed",
        mixed,
        "--mixed-manifest",
        mixed_manifest,
        "--ood",
        ood,
        "--ood-manifest",
        ood_manifest,
        "--selection-manifest",
        selection_manifest,
        "--lengths",
        lengths,
    )
    assert failed.returncode == 1
    assert "SHA-256" in failed.stderr


def test_default_right_sized_budget_includes_ood_labeling_and_matrix_shape(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rental-budget.json"
    result = subprocess.run(
        ["bash", str(BUDGET)],
        cwd=ROOT,
        env={
            **os.environ,
            "OUTPUT": str(output),
            "CAPACITY_RPS": "1.5",
            "MIXED_REQUESTS": "300",
            "OOD_REQUESTS": "200",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["total_hours"] <= 5.25
    assert payload["inputs"] == {
        "capacity_rps": 1.5,
        "mixed_requests": 300,
        "ood_requests": 200,
        "mixed_repeats": 3,
        "ood_repeats": 3,
        "mixed_policy_count": 7,
        "ood_policy_count": 4,
    }
    stages = {stage["name"]: stage for stage in payload["stages"]}
    assert stages["ood_labeling_800_direct_vllm"]["minutes"] == 25
