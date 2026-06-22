# vllm-ltr-optimization

CSCI 6806 capstone — reproduce + optimize **Learning-to-Rank scheduling** for low-latency LLM serving.

**Thesis:** *schedule by a predictable property → lower latency.*

## Three threads
- **Scheduling** (Dazhi) — reproduce the LTR scheduler; fix its ranker overfitting with **PARS** (pairwise ranking + BERT backbone).
- **Gateway** (Mingye) — latency-aware routing / two-layer semantic cache / admission control ("Velox" design).
- **Evaluation** (Yibo) — a reusable benchmark: MMLU quality gate + serving metrics (TTFT / TPOT / E2E / throughput).

Built on the base paper (Prof. A. S. Kumar, reproducing Fu et al. NeurIPS'24). Base fork: `hao-ai-lab/vllm-ltr`.

## Where to start
| Doc | Purpose |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Full project context (read first) |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Environment setup + how to reproduce (**start here to run**) |
| [`docs/references.md`](docs/references.md) | Papers to reproduce / compare, with `schedule-type` mapping |
| [`docs/presentation-plan.md`](docs/presentation-plan.md) | Midterm presentation outline (Wed 2026-06-24) |

## Status
Reproduction in progress — **baseline first** (FCFS / classification / LTR), PARS after.
GPU: RTX 4090 48GB.
