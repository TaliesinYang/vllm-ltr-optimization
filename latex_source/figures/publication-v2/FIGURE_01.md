# Figure 1 — Implemented request and control path

## Purpose

Show the implemented client → gateway → decision → vLLM path, its three service ports, and the team ownership boundary.

## Data

- Geometry and factual components: `scripts/plot_final_report_figures.py` (`draw_fig1`, `FIG1_CONTROL_LANES`).
- Ports and launch topology: `scripts/server/launch_gateway.sh`, `scripts/server/launch_decision.sh`, `scripts/server/launch_vllm.sh`.
- Pinned VeloxMesh whitelist/fail-open implementation: `/Users/alex/develop/VeloxMesh/internal/ltr/decision.go` at commit `888fba9984a34b23340f08e6faf81ace032f3a01`; pin recorded by `scripts/server/manifest/gateway-pin.txt`.
- Python-side decision contract: `scheduler_benchmark/decision_service.py`, `scheduler_benchmark/gateway_transport.py`.
- Reliability implementation boundary: `scheduler_benchmark/predictor.py` (`BertPredictor.predict`).

## Why this experiment

This is a system map, not an experiment. It defines what the later latency measurements include and where scheduling metadata enters vLLM.

## How to read

Solid horizontal arrows are the request data plane; the blue branch is the decision-control call; the dashed arrow is the SSE response path. The estimate is optional because the pinned gateway contract accepts `estimated_tokens` only with a reliable verdict.

## Result

The implemented path uses gateway `:9100`, decision service `:9200`, and vLLM `:8000`. The decision service returns a verdict, provenance, and an optional estimate. Pinned VeloxMesh commit `888fba9` only forwards client strings `ltr_kind` and `ltr_category`, fails open on call/contract errors, and injects `workflow_estimated_tokens` only when `prediction_reliable` is true.

## Limitation

The BERT predictor currently returns placeholder confidence and sets `ood = False`; no evaluated online OOD detector is implemented. The diagram must not be cited as evidence of online OOD detection or calibrated confidence.

## Contribution boundary

Dazhi's documented contribution is the predictor/ranker, scheduling integration, and scheduling evidence/analysis. Mingye owns the VeloxMesh gateway infrastructure. Yibo owns the reusable evaluation thread. BERT, vLLM, and the gateway are not inventions owned by Dazhi.

## Reproducibility

Run `python -B scripts/report_figures/publication_v2/figures_01_03.py`. The builder is `build_fig1()`; it writes vector `fig1.pdf` and 300 dpi `fig1.png`.

## Tomorrow's one-line explanation

The scheduling hint travels through a real three-service path, but its confidence is still a placeholder rather than an evaluated online OOD signal.
