#!/usr/bin/env python3
"""Regenerate the complete publication-v2 figure set."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = (
    HERE / "figures_01_03.py",
    HERE / "figures_04_06.py",
    HERE / "figures_07_08.py",
)


def main() -> None:
    for script in SCRIPTS:
        subprocess.run([sys.executable, "-B", str(script)], check=True)


if __name__ == "__main__":
    main()
