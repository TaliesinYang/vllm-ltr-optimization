#!/usr/bin/env python3
"""Run one isolated FCFS direct-vLLM versus VeloxMesh overhead measurement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler_benchmark.gateway_overhead import main


if __name__ == "__main__":
    raise SystemExit(main())
