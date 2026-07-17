import subprocess
import sys
from pathlib import Path

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
