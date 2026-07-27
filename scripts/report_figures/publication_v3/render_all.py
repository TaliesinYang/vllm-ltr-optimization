#!/usr/bin/env python3
"""Render and validate the complete publication-v3 figure set."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_figures.publication_v3 import (  # noqa: E402
    figure_03,
    figure_04,
    figure_05,
    figure_06,
    figure_07_08,
)


OUTPUT_DIR = REPO_ROOT / "latex_source" / "figures" / "publication-v3"
STATIC_SOURCE_NAMES = (
    "fig1.drawio",
    "fig1.drawio.svg",
    "fig1.svg",
    "fig2.drawio",
    "fig2.drawio.svg",
    "fig2.svg",
)


def validate_static_figures(static_dir: Path = OUTPUT_DIR) -> tuple[Path, ...]:
    """Require the static Draw.io/SVG sources for Figures 1 and 2."""

    paths = tuple(static_dir / name for name in STATIC_SOURCE_NAMES)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing static figure source: {path}")
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            raise ET.ParseError(f"invalid static figure XML: {path}: {exc}") from exc
    return paths


def render_all(
    output_dir: Path = OUTPUT_DIR,
    *,
    static_dir: Path = OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Validate Figures 1–2, then render Figures 3–8 in numeric order."""

    validate_static_figures(static_dir)
    return (
        *figure_03.render(output_dir),
        *figure_04.render(output_dir),
        *figure_05.render(output_dir),
        *figure_06.render(output_dir),
        *figure_07_08.render_fig7(output_dir),
        *figure_07_08.render_fig8(output_dir),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--static-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in render_all(args.output_dir, static_dir=args.static_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
