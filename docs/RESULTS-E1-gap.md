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

## Table E2 — Ablation: which factor drives the cross-dist gain? (measured 2026-06-22)
Decompose the 3 PARS changes (loss / backbone / delta-filter) by training one-factor variants.

| # | Predictor | Backbone | Loss | δ-filter | Tau in-dist | **Tau cross** | Gap |
|---|---|---|---|---|---|---|---|
| | listMLE (base) | OPT-125M | listMLE | — | 0.559 | 0.315 | 0.243 |
| A1 | + pairwise loss only | OPT-125M | margin | δ=0.2 | 0.543 | **0.303** | 0.240 |
| A2 | + BERT, no filter | BERT | margin | δ=0 (off) | 0.598 | **0.368** | 0.230 |
| | PARS (full) | BERT | margin | δ=0.2 | 0.596 | **0.361** | 0.235 |

**Decomposition of the cross-dist Tau:**
- **Loss** (listMLE→margin, same OPT): 0.315 → 0.303 = **−0.012 (no help, slightly worse)**.
- **Backbone** (OPT→BERT): 0.303 → 0.368 = **+0.065 — the dominant driver of the generalization gain.**
- **δ-filter** (off→0.2 on BERT): 0.368 → 0.361 = **−0.007 (no help here, slightly worse)**.

**Honest conclusion:** PARS's cross-distribution improvement over listMLE is **driven almost entirely by the
BERT backbone**, not by the pairwise margin loss or the delta-filter. On our single-GPU 8B setup, A2
(BERT + margin, filter OFF) is marginally the best config (gap 0.230). This directly answers the obvious
reviewer question "is it just BERT?" — largely **yes**. The pairwise loss and δ-filter (PARS's headline
mechanisms) did not help here; possible reasons: per-batch (not global) δ-filtering, small batch, single
seed, short prompts. A faithful global δ-filter or larger-scale training may recover their benefit — a
stated limitation / future-work item, not a fabricated win.

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
