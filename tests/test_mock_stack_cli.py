import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_gateway_stack_script_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_mock_gateway_stack.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--gateway-port" in result.stdout
    assert "--decision-port" in result.stdout
    assert "--engine-port" in result.stdout
