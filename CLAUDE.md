# CLAUDE.md — vllm-ltr-optimization (CSCI 6806 Capstone)

> Implementation repo for the capstone. Course docs / notes / deliverables live in the
> Obsidian course folder (see **Pointers**). **This file is the project memory — read it first.**

## Project
Reproduce + optimize **vllm-ltr** — Learning-to-Rank (LTR) scheduling for LLM inference serving.
Base paper by the course instructor (Prof. Anitha Saravana Kumar), itself reproducing
Fu et al. NeurIPS'24. The base LTR predictor (OPT-125M, listMLE) **overfits its training
distribution and generalizes poorly** — that admitted flaw is our core target.

One-line thesis: *schedule by a predictable property → lower latency.*

## Course / timeline
- CSCI 6806 / INFO 4205, 2026 Summer, instructor **Anitha Saravana Kumar** (a.saravanakumar@fdu.edu).
- Research-rigor bar: advisor **Jeeho Ryoo** (j.ryoo@fdu.edu).
- **Midterm presentation: Wed 2026-06-24** — show baseline reproduction + proposed optimization
  + how we differ; a review committee may attend.
- Reproduction (synthesize data + FCFS / classification / LTR) = the midterm deliverable.

## Team & threads
| Member | Student # | Thread |
|---|---|---|
| Dazhi (Alex) Yang | 2134432 | **Scheduling** — reproduce LTR + fix ranker overfitting (PARS) |
| Mingye Lang | 2099150 | **Gateway** — routing / cache / admission ("Velox" design) |
| Yibo Zhang | 2104413 | **Evaluation** — reusable benchmark (MMLU + serving metrics) |

## Our optimization = PARS (Dazhi's thread)
PARS (Prompt-Aware Ranking Scheduler, Tao et al., arXiv:2510.03243, 2025) fixes the overfitting:
- **Pairwise margin loss** (not listwise listMLE) — more stable training signal.
- **Filter noisy pairs**: train only on prompt pairs whose length differs by ≥ δ (≈0.2 Llama).
- **BERT-base backbone** (vs the base paper's OPT-125M).
- **Cross-model generalization** (train on one LLM, apply to others without retraining).
- + aging for starvation prevention.
Code hook: swap `--loss listMLE` → a pairwise margin loss + a BERT `pred_model` in the config.

## Plan — BASELINE FIRST, then PARS
Ryoo's rule: a unified harness + reproduction BEFORE any optimization or "beat" claim.
1. Reproduce baseline (FCFS / classification / listMLE-LTR) under one harness → latency-vs-rate curves.
2. Expose overfitting: same predictor on a held-out distribution (ShareGPT) → Kendall's Tau gap.
3. THEN apply PARS (pairwise + BERT) → measure how much the gap shrinks.
Do NOT jump to PARS before baseline numbers exist. Details: `docs/REPRODUCTION.md`.

**Collect + persist results before stopping any rented GPU** (raw `RESULTS/`, predictor + `usage_config.json`,
Kendall's Tau on train AND held-out, a per-run manifest of versions/flags/trace). Rented disk is ephemeral —
no saved data = wasted run. See `docs/REPRODUCTION.md` → *Results & data collection*.

## What we TRAIN vs SERVE (important)
- **TRAIN** = the small **predictor** (a transformer): OPT-125M (listMLE baseline) or BERT-base (PARS), ~125M params.
- **SERVE** = Llama-3-8B (inference only — we do NOT train the 8B model).

## Hardware
- Target GPU: **RTX 4090 48GB** (Ada, sm_89). 48GB > the 40GB needed → trains the 125M predictor
  AND serves Llama-3-8B. **A100 not required.**
- Compile kernels for Ada: export `TORCH_CUDA_ARCH_LIST="8.9"` before `pip install -e .`.
  CUDA 12.1 / torch 2.2.1 support sm_89.
- Absolute latency will differ from the paper (different HW) — fine: we compare methods on the
  SAME card and scope claims to this testbed, never extrapolating.

## Academic integrity (hard rules)
- Prof. Kumar runs an AI-writing detector → **do NOT use AI to write the paper.** AI may gather /
  structure information; the final prose must be the student's own.
- Data honesty: no fabricated numbers. PARS / TIE / EGTP published figures are author-reported —
  cite as references, never claim to "beat" them before reproducing under our own harness.

## GPU deploy (恒源云 / gpushare + OSS)
Use the **`gpu-cloud-deploy`** skill (恒源云-native `oss` tool, `/hy-tmp` persistent paths, auto-shutdown).
Note: **AutoDL ≠ 恒源云** — the `oss` CLI here is 恒源云/gpushare-specific; rent on 恒源云 to use it.
Flow (scripts in `scripts/`): `hy_deploy.sh` (instance: network_turbo + clone repo + `setup.sh`, HF via
`hf-mirror.com`, optional OSS-staged data) → in tmux `hy_run_and_upload.sh` (run baseline → collect →
`oss cp oss://backup/` → shutdown). Pre-upload heavy data to OSS once to skip HF on later runs.

**China network (HF + GitHub blocked/slow):** (a) `source /etc/network_turbo` (恒源云 built-in proxy →
GitHub + HF); (b) gated **Llama-3-8B from ModelScope** `LLM-Research/Meta-Llama-3-8B-Instruct` (no HF
token needed) → local `/hy-tmp/models/...`, `run_baseline.sh` defaults to that path; (c) non-gated LTR
trace + predictors via `HF_ENDPOINT=https://hf-mirror.com`. GitHub fallback if still slow: mirror repos to Gitee.

## Repos
- This repo (implementation): `github.com/TaliesinYang/vllm-ltr-optimization`
- Base fork to reproduce: `/Users/alex/develop/vllm-ltr` (`github.com/hao-ai-lab/vllm-ltr`)
- Git identity: commit as **alex** (personal), never the MGA org account.

## Pointers — Obsidian course folder (docs / materials / deliverables)
Base: `/Volumes/T7 Shield/obsidian/4-Resources/Courses/VPL/FDUClasses/26VU_CSCI_6806_V1 Computer Sci Gr Capstone Proj/`
- Reference papers: `materials/references/` (Fu2024, PARS, TIE, EGTP, gateway `pt2_*`, …)
- Original runbook: `project/REPRODUCTION-RUNBOOK.md`
- Mingye's gateway design: `project/gateway-architecture.md` + `materials/references/paper-selection-lang.md`
- W3 lecture notes: `generated/Week03/notes.md` · Research synthesis: `_inbox/2026-05-27-research-synthesis.md`
- Summary PDF (submitted 2026-06-20): `deliverables/2026-06-20-work-summary.pdf`

## Status (2026-06-22)
- ✅ **Baseline REPRODUCED** on RTX 4090 48GB / CUDA 12.1 (恒源云). FCFS vs LTR, rates {2..64}.
  Result: LTR cuts TTFT up to **2.9×** under load (rate ≥16) — reproduces the base paper.
  Env fixes + result recorded in `docs/ENV-NOTES.md`; raw data in `…/deliverables/04-evaluation/baseline-2026-06-22/`.
- `scripts/setup.sh` is now the battle-tested one-shot setup (re-run = ~10-15 min, not 1 h of debugging).

## Next action
**PARS** — write the pairwise-margin + BERT predictor (Dazhi's contribution), train it, compare its
generalization gap vs listMLE. Plus: draw Fig E1 (latency-vs-rate) from the baseline data for Wednesday
(`docs/presentation-plan.md`). Classification baseline has a benchmark-side `IndexError` to fix if a 3rd line is wanted.

## Agent skills

### Issue tracker

GitHub Issues (repo `TaliesinYang/vllm-ltr-optimization`); use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.
