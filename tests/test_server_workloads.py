import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/server/build_server_workloads.py"


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


def test_verify_workloads_checks_manifest_raw_sha_and_profile_rows(
    tmp_path: Path,
) -> None:
    mixed = tmp_path / "mixed.v2.jsonl"
    ood = tmp_path / "ood.v2.jsonl"
    mixed_manifest = tmp_path / "mixed.v2.manifest.json"
    ood_manifest = tmp_path / "ood.v2.manifest.json"
    combined_ood = tmp_path / "ood-label-inputs.jsonl"
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
        "--ood-input",
        combined_ood,
        "--lengths",
        lengths,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mixed_rows"] == 2
    assert report["ood_rows"] == 1

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
        "--ood-input",
        combined_ood,
        "--lengths",
        lengths,
    )
    assert failed.returncode == 1
    assert "SHA-256" in failed.stderr
