#!/usr/bin/env python3
"""Check every exported figure against the paper's typographic contract.

Two things go wrong silently in a LaTeX figure pipeline: a figure drawn at one
size and placed at another (which scales its text), and a font that is not
embedded (which changes on someone else's machine). Both are invisible in the
source and obvious only in print, so they are checked here rather than trusted.

Contract:
  * effective text size after placement >= MIN_PT
  * every font embedded, no Type 3
  * placement scale within TOLERANCE of 1.0
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FIGS = Path(__file__).resolve().parent.parent / "figs"
TEX = Path(__file__).resolve().parent.parent
MIN_PT = 10.0
TOLERANCE = 0.05
COLUMN_IN = 3.5
TEXT_IN = 7.16


def placed_widths() -> dict[str, float]:
    """Map figure file -> width in inches as LaTeX will place it."""
    placed: dict[str, float] = {}
    for tex in sorted(TEX.glob("0*.tex")):
        body = tex.read_text(encoding="utf-8")
        spans = re.split(r"(\\begin\{figure\*?\})", body)
        current = COLUMN_IN
        for span in spans:
            if span == r"\begin{figure*}":
                current = TEXT_IN
            elif span == r"\begin{figure}":
                current = COLUMN_IN
            for match in re.finditer(
                r"includegraphics\[width=([^\]]*)\]\{([^}]+)\}", span
            ):
                spec, name = match.group(1), match.group(2)
                factor = 1.0
                scale = re.match(r"([0-9.]+)\\", spec)
                if scale:
                    factor = float(scale.group(1))
                placed[name] = current * factor
    return placed


def native_width(pdf: Path) -> float:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Page size"):
            return float(line.split()[2]) / 72.0
    raise ValueError(f"no page size for {pdf}")


def font_report(pdf: Path) -> list[str]:
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    problems = []
    for line in out.splitlines()[2:]:
        if not line.strip():
            continue
        cols = line.split()
        kind = next((c for c in cols if c.startswith("Type")), "?")
        if "Type 3" in line:
            problems.append(f"Type 3 font: {cols[0]}")
        if " no " in f" {' '.join(cols[-4:])} ":
            problems.append(f"font not embedded: {cols[0]} ({kind})")
    return problems


def main() -> int:
    placed = placed_widths()
    failures = 0
    print(f"{'figure':16s} {'native':>8s} {'placed':>8s} {'scale':>7s} {'min pt':>7s}")
    for name, width in sorted(placed.items()):
        pdf = FIGS / name
        if not pdf.exists():
            continue
        native = native_width(pdf)
        scale = width / native
        effective = MIN_PT * scale
        flag = ""
        if abs(scale - 1.0) > TOLERANCE:
            flag += "  SCALE"
            failures += 1
        if effective < MIN_PT - 0.2:
            flag += "  FONT<10pt"
            failures += 1
        for problem in font_report(pdf):
            flag += f"  {problem}"
            failures += 1
        print(f"{name:16s} {native:7.2f}in {width:7.2f}in {scale:6.0%} "
              f"{effective:6.1f}pt{flag}")
    print("CONTRACT OK" if failures == 0 else f"{failures} CONTRACT VIOLATIONS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
