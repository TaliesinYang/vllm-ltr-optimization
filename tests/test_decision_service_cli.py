import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scheduler_benchmark.predictor import ConstantPredictor
from scheduler_benchmark.rank_quantiles import APPROXIMATION_NOTICE, MAPPING_VERSION
from scripts import run_decision_service

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_quantile_manifest(path: Path) -> bytes:
    raw = (
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "model_version": "test-model",
                "approximation_notice": APPROXIMATION_NOTICE,
                "sample_count": 6000,
                "percentiles": {
                    str(percentile): float(10 + 5 * percentile)
                    for percentile in range(10, 100)
                },
                "global_quantiles": {
                    "50": 260.0,
                    "70": 360.0,
                    "90": 460.0,
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_decision_service_script_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_decision_service.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--max-concurrency" in result.stdout


def test_cli_selects_stub_or_real_bert_checkpoint(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "quantiles.json"
    write_quantile_manifest(manifest_path)
    stub_args = run_decision_service.parse_args(
        ["--quantile-manifest", str(manifest_path)]
    )

    assert isinstance(
        run_decision_service.build_predictor(stub_args), ConstantPredictor
    )
    assert (
        run_decision_service.effective_feature_variant(stub_args)
        == "prompt_schema_history_workflow"
    )
    assert run_decision_service.effective_predictor_revision(stub_args) == (
        "stub-constant-v1"
    )

    loaded = {}

    class FakeBertPredictor:
        def __init__(self, checkpoint: Path) -> None:
            loaded["checkpoint"] = checkpoint

    monkeypatch.setattr(run_decision_service, "BertPredictor", FakeBertPredictor)
    bert_args = run_decision_service.parse_args(
        [
            "--predictor",
            "bert",
            "--quantile-manifest",
            str(manifest_path),
        ]
    )

    predictor = run_decision_service.build_predictor(bert_args)

    assert isinstance(predictor, FakeBertPredictor)
    assert loaded["checkpoint"] == REPO_ROOT / "checkpoints_best_predictor"
    assert run_decision_service.effective_feature_variant(bert_args) == "prompt_schema"
    assert run_decision_service.effective_predictor_revision(bert_args) == (
        "bert-prompt_schema-tier2-seed17"
    )


def test_cli_requires_quantile_manifest() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_decision_service.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--quantile-manifest" in result.stderr


def test_cli_hashes_original_quantile_manifest_bytes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "quantiles.json"
    raw = write_quantile_manifest(manifest_path)

    _, manifest_sha256 = run_decision_service.load_quantile_manifest(
        manifest_path
    )

    assert manifest_sha256 == hashlib.sha256(raw).hexdigest()


def test_cli_rejects_malformed_quantile_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "malformed.json"
    manifest_path.write_text("not-json", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_decision_service.py",
            "--quantile-manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "quantile manifest" in result.stderr


def test_cli_rejects_invalid_quantile_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_decision_service.py",
            "--quantile-manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "mapping_version" in result.stderr


def test_cli_rejects_numeric_overflow_in_quantile_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "overflow.json"
    write_quantile_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["percentiles"]["10"] = 10**400
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_decision_service.py",
            "--quantile-manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid quantile manifest" in result.stderr
    assert "Traceback" not in result.stderr
