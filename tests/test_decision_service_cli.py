import subprocess
import sys
from pathlib import Path

from scheduler_benchmark.predictor import ConstantPredictor
from scripts import run_decision_service

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_cli_selects_stub_or_real_bert_checkpoint(monkeypatch) -> None:
    stub_args = run_decision_service.parse_args([])

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
    bert_args = run_decision_service.parse_args(["--predictor", "bert"])

    predictor = run_decision_service.build_predictor(bert_args)

    assert isinstance(predictor, FakeBertPredictor)
    assert loaded["checkpoint"] == REPO_ROOT / "checkpoints_best_predictor"
    assert run_decision_service.effective_feature_variant(bert_args) == "prompt_schema"
    assert run_decision_service.effective_predictor_revision(bert_args) == (
        "bert-prompt_schema-tier2-seed17"
    )
