# Reproduction & comparison references

Papers we reproduce / compare against, and where each maps in code. PDFs live in the Obsidian course
folder: `…/26VU_CSCI_6806_V1 …/materials/references/`.

> Data honesty: PARS / TIE / EGTP numbers below are **author-reported** preprint figures — references
> only, never our "beat" claims until reproduced under our own harness (Ryoo's rule).

## Base + scheduling baselines (Thread 1 — Dazhi)
| Paper | File | Role | `--schedule-type` |
|---|---|---|---|
| Base paper (Prof. Kumar et al.) | `LLM.pdf` | The paper we reproduce + improve; LTR + vLLM swap, 2.1× vs FCFS | `opt-xxx` |
| Fu et al., NeurIPS'24 | `Fu2024_LTR-scheduling_NeurIPS24.pdf` | Direct predecessor: learn relative order (listwise LTR) | `opt-xxx` (listMLE) |
| — (no prediction) | — | First-come-first-served, the floor | `fcfs` |
| SSJF, ASPLOS'24 | `SSJF_proxy-length-prediction_ASPLOS24.pdf` | Proxy model predicts length → SJF (classification family) | `tpt-class10-xxx` |
| FastServe, NSDI'26 | `FastServe_skip-join-MLFQ_NSDI26.pdf` | Bypass camp: skip-join MLFQ, no prediction | `mlfq-...` |
| Oracle SJF | — | Ground-truth lengths = optimal upper bound | `sjf` |

## Our optimization (Thread 1 fix) — apply on top of the baseline
| Paper | File | What we adapt |
|---|---|---|
| **PARS** (Tao et al. 2025) | `PARS_pairwise-LTR_2025.pdf` | **Main fix.** Pairwise margin loss + δ-filter on noisy pairs + BERT backbone + cross-model generalization + aging. Swap `--loss listMLE` → pairwise; set BERT `pred_model`. |
| TIE (Zheng et al. 2026) | `TIE_uncertainty-aware_2026.pdf` | Ablation: uncertainty / CVaR scoring |
| EGTP (Xie et al., ICLR'26) | `EGTP_entropy-guided-hidden-states_ICLR26.pdf` | Ablation: reuse LLM hidden states (no aux model); predicts point length → MAE metric |

## Gateway (Thread 2 — Mingye); design: `project/gateway-architecture.md`
| Paper | File | What we adapt |
|---|---|---|
| SkyWalker | `pt2_3_*` | EWMA latency tracking, pending-aware, health-based routing |
| GORGO | `pt2_14_*` | Multi-signal additive scoring for routing |
| Kareto | `pt2_2_*` | Classified TTL caching |
| Apt-Serve | `pt2_27_*` | Two-level hybrid cache |
| Stream2LLM | `pt2_16_*` | Selective cache invalidation |
| MC-SF | `pt2_24_*` | Concurrency limits as admission gates |
| Medha | `pt2_25_*` | Timeout ejection + retry failover |

## Evaluation (Thread 3 — Yibo)
| Item | Role |
|---|---|
| MMLU (14k Q / 57 subjects) | Quality gate — optimization must not drop accuracy |
| Serving metrics | TTFT, TPOT, end-to-end latency, throughput; latency-vs-QPS curves (P50/P90/P99) |
| Fairness | Waiting time bucketed by output length — short requests not starved |

## Frontier extension (future work)
| Paper | File | Idea |
|---|---|---|
| KTransformers (SOSP'25) | `KTransformers_*` | CPU/GPU expert offload for MoE |
| MoE-Infinity | `MoE-Infinity_*` | Predict expert activation — "schedule by predicted property", new property |
