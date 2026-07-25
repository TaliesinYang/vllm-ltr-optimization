from __future__ import annotations

import re
from collections import Counter


def classify_long_response(text: str, *, ngram_size: int = 8) -> dict[str, object]:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    ngrams = [tuple(tokens[index : index + ngram_size]) for index in range(
        max(0, len(tokens) - ngram_size + 1)
    )]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    ratio = repeated / len(ngrams) if ngrams else 0.0
    if ratio >= 0.35:
        classification = "repetition_loop"
    elif ratio >= 0.12:
        classification = "mixed"
    else:
        classification = "genuine_long"

    evidence: list[str] = []
    for ngram, count in counts.most_common(3):
        if count < 2:
            break
        evidence.append(f"{count}x: {' '.join(ngram)}"[:300])
    if not evidence and text:
        evidence.append(text[:300].replace("\n", " "))
    return {
        "classification": classification,
        "token_count": len(tokens),
        "ngram_size": ngram_size,
        "repeated_ngram_ratio": ratio,
        "evidence_snippets": evidence,
    }


def aggregate_classifications(diagnoses: list[dict[str, object]]) -> str:
    labels = {str(item["classification"]) for item in diagnoses}
    if labels == {"repetition_loop"}:
        return "repetition_loop"
    if labels == {"genuine_long"}:
        return "genuine_long"
    return "mixed"
