#!/usr/bin/env python3
"""Run scheduler workload replay against one live vLLM policy server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler_benchmark.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
