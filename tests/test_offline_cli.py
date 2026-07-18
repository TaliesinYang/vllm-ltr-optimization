import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "build_ood_label_inputs.py",
    "build_offline_workload.py",
    "score_offline_ensemble.py",
    "analyze_offline_evidence.py",
    "tune_lightgbm_offline.py",
    "score_legacy_predictors.py",
    "build_chat_anchor_workload.py",
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_offline_scripts_expose_help() -> None:
    for script in SCRIPTS:
        result = _run(f"scripts/{script}", "--help")
        assert result.returncode == 0, (script, result.stderr)


def test_bfcl_conversion_and_workload_clis_persist_manifests(tmp_path) -> None:
    bfcl = tmp_path / "BFCL_v3_simple.json"
    bfcl.write_text(
        json.dumps(
            {
                "id": "simple_1",
                "question": [[{"role": "user", "content": "hello"}]],
                "function": [
                    {
                        "name": "lookup",
                        "parameters": {"type": "dict", "properties": {}},
                    }
                ],
            }
        )
        + "\n"
    )
    ood = tmp_path / "ood.jsonl"
    ood_manifest = tmp_path / "ood-manifest.json"

    converted = _run(
        "scripts/build_ood_label_inputs.py",
        "--source",
        "bfcl",
        "--input",
        str(bfcl),
        "--output",
        str(ood),
        "--manifest",
        str(ood_manifest),
        "--sample-size",
        "1",
        "--seed",
        "17",
    )

    assert converted.returncode == 0, converted.stderr
    assert json.loads(ood_manifest.read_text())["row_count"] == 1
    item = json.loads(ood.read_text())
    assert item["request_id"] == item["sample_id"]

    lengths = tmp_path / "lengths.jsonl"
    lengths.write_text(json.dumps({"sample_id": item["sample_id"], "output_length": 10}) + "\n")
    workload = tmp_path / "workload.jsonl"
    workload_manifest = tmp_path / "workload-manifest.json"
    built = _run(
        "scripts/build_offline_workload.py",
        "--ood-input",
        str(ood),
        "--lengths",
        str(lengths),
        "--profile",
        "ood",
        "--per-token-ms",
        "2.5",
        "--output",
        str(workload),
        "--manifest",
        str(workload_manifest),
    )

    assert built.returncode == 0, built.stderr
    assert json.loads(workload.read_text())["baseline_service_ms"] == 25.0
    assert json.loads(workload_manifest.read_text())["max_tokens"] == 4096


def test_missing_ensemble_and_legacy_checkpoints_write_blocked_reports(tmp_path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "request_id": "a",
                "prompt": "hello",
                "tool_schema": "[]",
                "history": [],
                "session_id": "a",
                "task_id": "a",
                "source": "bfcl",
                "source_revision": "r",
                "category": "simple",
            }
        )
        + "\n"
    )
    lengths = tmp_path / "lengths.jsonl"
    lengths.write_text(json.dumps({"sample_id": "a", "output_length": 5}) + "\n")
    score_report = tmp_path / "score-report.json"
    result = _run(
        "scripts/score_offline_ensemble.py",
        "--input",
        str(inputs),
        "--lengths",
        str(lengths),
        "--checkpoint",
        f"17={tmp_path / 'missing17'}",
        "--checkpoint",
        f"42={tmp_path / 'missing42'}",
        "--checkpoint",
        f"73={tmp_path / 'missing73'}",
        "--scores-output",
        str(tmp_path / "scores.jsonl"),
        "--report",
        str(score_report),
        "--diagnostic",
        str(tmp_path / "diagnostic.json"),
    )
    assert result.returncode == 2
    assert json.loads(score_report.read_text())["status"] == "blocked"

    legacy_report = tmp_path / "legacy.json"
    legacy = _run(
        "scripts/score_legacy_predictors.py",
        "--checkpoint-root",
        str(tmp_path / "course-deliverables"),
        "--output",
        str(legacy_report),
    )
    assert legacy.returncode == 0
    payload = json.loads(legacy_report.read_text())
    assert payload["priority"] == "P2_optional"
    assert all(item["status"] == "blocked" for item in payload["families"])


def test_default_checkpoint_probe_reports_real_seed17_and_missing_other_seeds(
    tmp_path,
) -> None:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("")
    lengths = tmp_path / "lengths.jsonl"
    lengths.write_text("")
    report = tmp_path / "report.json"

    result = _run(
        "scripts/score_offline_ensemble.py",
        "--input",
        str(inputs),
        "--lengths",
        str(lengths),
        "--scores-output",
        str(tmp_path / "scores.jsonl"),
        "--report",
        str(report),
        "--diagnostic",
        str(tmp_path / "diagnostic.json"),
    )

    assert result.returncode == 2
    payload = json.loads(report.read_text())
    assert payload["reason"] == "missing_required_checkpoints"
    assert payload["missing_seeds"] == [42, 73]
    assert payload["checkpoints"]["17"]["status"] == "present"
    assert payload["checkpoints"]["17"]["path"] == str(
        ROOT / "checkpoints_best_predictor"
    )
    assert payload["checkpoints"]["42"]["status"] == "missing"
    assert payload["checkpoints"]["73"]["status"] == "missing"
