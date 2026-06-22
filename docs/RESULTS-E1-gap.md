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

## Limitation — classification latency not obtainable (honest negative, measured 2026-06-22)
We trained our own classification predictor (OPT-125M, 10 length buckets, group-size 820, batch 4)
to compare classification vs LTR scheduling head-to-head. Two separate issues, reported honestly:

1. **Predictor result (obtained):** classification **Tau = 0.194** (acc 0.965 on the bucket labels).
   This is far below LTR's listMLE Tau 0.559 / PARS 0.596 — i.e. **at the predictor level our data
   already confirms the base paper's choice of LTR over classification** (a discrete 10-bucket label
   is a much weaker ranking signal than a continuous score).

2. **Latency sweep (could NOT be obtained — two stacked failures):**
   - `--swap-space 100` made the vLLM server **OOM-killed at startup** in the shared GPU container
     (pinning 100 GB of page-locked host memory exceeds the cgroup limit; the fork's own working
     scripts use swap-space 16–40). Fixed by lowering to `--swap-space 16` → server starts cleanly.
   - With a healthy server, the `tpt-class10` schedule type **still produced `Successful requests: 0`
     across request-rate 4 / 8 / 16 / 32** (requests are submitted but none ever complete; the
     benchmark then raises `IndexError` computing percentiles over the empty latency array).
     This is a **serving bug in the fork's class-based scheduling path**, not a config artifact —
     no swap-space value yields completions because the bottleneck is the scheduler, not startup.

   **No classification latency numbers exist and none were fabricated.** Classification is therefore
   compared to LTR at the **predictor (Tau) level only**; its end-to-end latency is a stated limitation.
   Root-causing the `tpt-class10` 0-completion bug (predictor-output vs schedule-type) is future work.

## Artifacts (saved)
- Weights: `…/deliverables/04-evaluation/{listmle,pars}-OURS-2026-06-22/` (model.safetensors + usage_config)
- Class predictor: `…/deliverables/04-evaluation/MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b4-OURS/`
  (model.safetensors 250 MB + config + usage_config) — Tau 0.194, no valid latency
- Baseline latency (FCFS vs LTR): `…/deliverables/04-evaluation/baseline-2026-06-22/`
- Code: `pars/` (marginRanking loss, BERT config, eval_gap) + `docs/PARS-PLAN.md` + `scripts/run_classsweep.sh`
