import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gateway_overhead_script_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_gateway_overhead.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--direct-endpoint" in result.stdout
    assert "--gateway-endpoint" in result.stdout
    assert "FCFS" in result.stdout
