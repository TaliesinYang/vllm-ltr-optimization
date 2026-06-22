# Midterm presentation plan — Wed 2026-06-24

What the professor expects (W3 lecture, verbatim): reproduction = the midterm; show **what your next
work is, how your work differs**; a **review committee** may evaluate the proposed new work.
So this is a **progress + proposal** talk, not a finished optimization.

## What to show
1. **Reproduction progress** — base paper running under our harness; FCFS vs LTR (and classification)
   latency-vs-request-rate curves on Llama-3-8B / LMSYS. Even partial is fine.
2. **The gap we target** — the LTR ranker overfits (Kendall's Tau drops on held-out). Show the gap if
   reproduced; otherwise state it as the diagnosed problem.
3. **Our proposed optimization (3 threads)** + how each **differs** from the base paper:
   - Scheduling (Dazhi): **PARS** — pairwise + BERT + δ-filter fixes the overfitting.
   - Gateway (Mingye): routing / semantic cache / admission — saves latency one layer earlier.
   - Evaluation (Yibo): reusable benchmark — MMLU quality gate + serving metrics + fairness.
4. **Plan / timeline** — baseline now → expose gap → apply PARS → ablations (TIE/EGTP) → MoE extension.
5. **Be ready for review-committee questions** (see below).

## Slide sketch (~10 slides, ~2 per person)
1. Title + team + one-line thesis ("schedule by a predictable property → lower latency").
2. Base paper recap (LTR + vLLM preemption, 2.1× vs FCFS) + the admitted overfitting flaw.
3. Reproduction setup (RTX 4090 48GB, Llama-3-8B, LMSYS; FCFS / classification / LTR; unified harness).
4. Reproduction results so far — latency-vs-rate curve (FCFS vs LTR). [live data]
5. The generalization gap — Tau train vs held-out (or stated as the target if not yet measured).
6. Thread 1 · PARS — pairwise margin loss, δ-filter, BERT backbone, cross-model generalization.
7. Thread 2 · Gateway — routing / semantic cache / admission; input-token SJF proxy; horizontal scaling.
8. Thread 3 · Evaluation — MMLU quality gate + TTFT/TPOT/E2E/throughput + fairness; reusable benchmark.
9. How we differ + timeline (next 4 weeks).
10. Frontier extension (MoE: KTransformers / MoE-Infinity) — one slide, future work.

## Likely review-committee questions (prep answers)
- "How does your work differ from the base paper?" → we fix its admitted ranker overfitting (PARS), add a
  gateway layer, and build a reusable benchmark; differentiate by axis, not by dismissal.
- "What's reproduced vs proposed?" → baseline (FCFS/LTR) reproduced under our harness; PARS is proposed/next.
- "Why pairwise over listwise?" → listwise (listMLE) learns from unstable rankings; pairwise + δ-filter
  drops noisy pairs → more stable, generalizes across models.
- "Hardware / can you run it?" → single RTX 4090 48GB; we scope claims to this testbed, not the paper's multi-GPU absolute numbers.
- "Did you beat PARS/TIE/EGTP?" → no claim yet; their numbers are author-reported; we compare only under our own harness.

## Honesty / integrity
- Present only what is actually reproduced; label proposed work as proposed.
- No fabricated numbers. Paper prose is written by the team, not AI (Prof. Kumar runs a detector).
