#!/usr/bin/env python3
"""Report text printed on top of other text anywhere in the compiled PDF.

Hand-positioned TikZ nodes have no layout engine behind them, so a label that
clears its neighbours at one font size lands on top of them at another. That
happened twice here while raising the schematics to 10 pt, and both times the
source looked fine -- the defect existed only in the rendering.

WHAT THIS DOES NOT CATCH, and why you still have to look at the figures:
a label whose opaque background erases a rule it crosses. Both real defects
found on 2026-07-27 were of that kind: a warn label sitting on the system
boundary, and another on a box border. Only glyph-versus-glyph collisions are
detectable from the text layer, because rules are not text.

Tall delimiters produce false positives: a cases brace spans several lines, so
its bounding box clips the line above it. Anything reported here needs an eye
on the rendering before it is treated as a defect.

Run: python3 scripts/overlap_check.py [document.pdf]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PDF = Path(__file__).resolve().parent.parent / "00.tc_main.pdf"
WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>'
)
# Below this share of the smaller box, an intersection is kerning or a tall
# delimiter rather than a collision a reader would see.
REPORT_ABOVE = 0.18


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise ValueError(f"no page count for {pdf}")


def words(pdf: Path, page: int) -> list[tuple[float, float, float, float, str]]:
    out = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-bbox", str(pdf), "-"],
        capture_output=True, text=True).stdout
    return [(float(a), float(b), float(c), float(d), text)
            for a, b, c, d, text in WORD.findall(out)]


def intersection_share(a, b) -> float:
    ax0, ay0, ax1, ay1, _ = a
    bx0, by0, bx1, by1, _ = b
    wide = min(ax1, bx1) - max(ax0, bx0)
    tall = min(ay1, by1) - max(ay0, by0)
    if wide <= 0.5 or tall <= 0.5:
        return 0.0
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return (wide * tall) / smaller if smaller else 0.0


def main() -> int:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF
    if not pdf.exists():
        raise SystemExit(f"no such PDF: {pdf}")
    found = 0
    for page in range(1, page_count(pdf) + 1):
        on_page = words(pdf, page)
        for i, first in enumerate(on_page):
            for second in on_page[i + 1:]:
                share = intersection_share(first, second)
                if share > REPORT_ABOVE:
                    found += 1
                    print(f"page {page}: {first[4]!r} over {second[4]!r} "
                          f"({share:.0%} of the smaller box)")
    print("NO TEXT-ON-TEXT OVERLAP" if found == 0
          else f"{found} candidate overlaps -- check each against the rendering")
    return 0


if __name__ == "__main__":
    sys.exit(main())
