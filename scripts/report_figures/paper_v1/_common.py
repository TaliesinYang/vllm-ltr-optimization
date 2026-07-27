"""Shared loading, styling and provenance for the paper_v1 figure set.

Every number a figure draws is read from a committed artifact at build time.
Nothing distributional is written here as a literal, so a rebuilt artifact
changes the figure rather than silently disagreeing with it.

Provenance: each generator appends one line per source file (path + sha256) to
figs/PROVENANCE.txt, so any figure can be traced to the exact bytes behind it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "report_figures"))

from style import IEEE_DOUBLE_WIDTH, IEEE_SINGLE_WIDTH, OKABE_ITO  # noqa: E402,F401

FIGS = Path("/Users/alex/develop/capstone-final-report/figs")
OFFLINE = REPO / "runs" / "offline-experiments-2026-07-25"
PROBE_SCHEMA = REPO / "probes" / "schema-variability-2026-07-25"
PROBE_TRACE = REPO / "probes" / "agent-traces-2026-07-26"
PROVENANCE = FIGS / "PROVENANCE.txt"

# One colour role per concept, held constant across every figure in the set.
COLOR = {
    # Semantic palette, not a rainbow: the proposed input carries the single
    # accent colour, its ablation a lighter tint of the same hue, and every
    # baseline is neutral grey. Colour therefore means "ours vs not ours"
    # rather than merely "a different row".
    "prompt_schema": OKABE_ITO["blue"],
    "prompt_only": OKABE_ITO["sky_blue"],
    "lightgbm_grid": "#6E6E6E",
    "lightgbm_scalar": "#9A9A9A",
    "schema_hash_lookup": "#C2C2C2",
    "withheld": OKABE_ITO["light_gray"],
    "overstate": OKABE_ITO["vermillion"],
    "abstain": OKABE_ITO["dark_gray"],
    "neutral": OKABE_ITO["gray"],
}

LABEL = {
    "bert_prompt_schema": "BERT prompt+schema",
    "bert_prompt_only": "BERT prompt only",
    "lightgbm_grid": "LightGBM grid",
    "lightgbm_scalar": "LightGBM fixed",
    "schema_hash_lookup": "Schema-hash lookup",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        return [json.loads(line) for line in handle if line.strip()]


def record_provenance(figure: str, sources: list[Path]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    existing = []
    if PROVENANCE.exists():
        existing = [
            line
            for line in PROVENANCE.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith(f"{figure}\t")
        ]
    lines = existing + [
        f"{figure}\t{path.relative_to(REPO)}\tsha256={sha256(path)}"
        for path in sources
    ]
    PROVENANCE.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    print(f"wrote {out}")
    return out
