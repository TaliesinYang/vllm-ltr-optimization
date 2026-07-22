"""Fail if any text in the given figure PDFs renders below the >=10pt rule.

Matplotlib log-scale mathtext exponents silently shrink to ~7pt regardless of
rcParams, and pdfminer/pdfplumber glyph geometry is unreliable (a 10pt period is
short; a rotated 10pt axis label reports its width). The authoritative nominal
size is the value in each `... Tf` text-font operator in the content stream,
which matplotlib sets to the actual point size. Usage: verify_pdf_fonts.py a.pdf ...
"""
import re
import sys
import zlib
from collections import Counter

MIN_PT = 10.0
TOLERANCE = 0.2
_TF = re.compile(rb"/F\d+\s+([\d.]+)\s+Tf")
_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)


def tf_sizes(path: str) -> Counter:
    data = open(path, "rb").read()
    sizes: Counter = Counter()
    for match in _STREAM.finditer(data):
        raw = match.group(1)
        try:
            body = zlib.decompress(raw)
        except zlib.error:
            body = raw
        for tf in _TF.finditer(body):
            sizes[round(float(tf.group(1)), 2)] += 1
    return sizes


def main() -> int:
    failures = 0
    for path in sys.argv[1:]:
        sizes = tf_sizes(path)
        below = {size: count for size, count in sizes.items() if size < MIN_PT - TOLERANCE}
        smallest = min(sizes) if sizes else float("nan")
        status = "FAIL" if below else "ok"
        failures += bool(below)
        print(f"[{status}] {path} smallest={smallest}pt sizes={dict(sorted(sizes.items()))}")
        if below:
            print(f"        <10pt Tf spans: {dict(sorted(below.items()))}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
