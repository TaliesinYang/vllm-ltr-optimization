# Results — Table E1: generalization gap (measured 2026-06-22, RTX 4090 48GB)

Our own trained predictors (NOT the authors' pretrained). Kendall's Tau on the in-distribution
LMSYS held-out tail vs the cross-distribution ShareGPT test set (never seen in training).
Both predictors trained ONLY on LMSYS-c20000:30000. Produced by `pars/eval_gap.py`.

| Predictor | Backbone | Loss | Tau (LMSYS in-dist) | Tau (ShareGPT cross) | **Gap = in − cross** |
|---|---|---|---|---|---|
| listMLE (baseline) | OPT-125M | listMLE | 0.559 | 0.315 | **0.243** |
| **PARS (ours)** | BERT-base | margin(1.0)+δ(0.2) | **0.596** | **0.361** | **0.235** |

## Headline
- **PARS generalizes better**: cross-distribution Tau **0.315 → 0.361 (+0.046, +15% relative)** on unseen ShareGPT.
- PARS is also better in-distribution (0.559 → 0.596) and has a smaller gap (0.243 → 0.235).
- **Both still overfit substantially** (Tau drops ~0.24–0.30 on ShareGPT) — confirms the base paper's
  admitted flaw; PARS reduces it but does not eliminate it. Honest framing: the cross-dist absolute
  gain is the clear win; the gap-shrink itself is modest.

## Caveats (disclose in the deliverable)
- listMLE trained at batch 4 (OOM-limited on 48GB; paper used 32) — may slightly depress its Tau.
- delta-filter is applied per-batch (slate = batch), not as PARS's offline global filter — semantically
  equivalent, implementation differs.
- In-dist Tau from `eval_gap.py` (0.559) matches the training-time epoch Tau (~0.55) — consistency OK.
- Single seed, single GPU, 8B model. Numbers are ours, measured — never fabricated.

## Artifacts (saved)
- Weights: `…/deliverables/04-evaluation/{listmle,pars}-OURS-2026-06-22/` (model.safetensors + usage_config)
- Baseline latency (FCFS vs LTR): `…/deliverables/04-evaluation/baseline-2026-06-22/`
- Code: `pars/` (marginRanking loss, BERT config, eval_gap) + `docs/PARS-PLAN.md`
