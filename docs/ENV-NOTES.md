# Environment notes — baseline reproduction (verified 2026-06-22)

Reproduced FCFS-vs-LTR on **RTX 4090 48GB / CUDA 12.1** (恒源云/gpushare, `i-1:19590`). Everything below
is encoded in `scripts/setup.sh` — this file is the human-readable record of what broke and why.

## The 7 fixes (old 2024 fork on a 2026 image)
| # | Symptom | Fix |
|---|---|---|
| 1 | `CMake 3.21 required, you are running 3.16.3` | `pip install cmake==3.30.5` (NOT cmake 4.x — too new for the old CMakeLists) |
| 2 | `nvcc: command not found` → `Failed to detect a default CUDA architecture` | export `PATH=/usr/local/cuda/bin:$PATH` + `CUDA_HOME` before building (non-login shell lacks it) |
| 3 | xformers install re-downloads ~2GB torch | `pip install xformers==0.0.25 --no-deps` (torch 2.2.1 already present) |
| 4 | build wheel fails / model gated | build with `TORCH_CUDA_ARCH_LIST=8.9 --no-build-isolation`; get Llama-3-8B from **ModelScope** (`LLM-Research/Meta-Llama-3-8B-Instruct`), HF denies the gated repo |
| 5 | `transformers Disabling PyTorch (needs >=2.4, found 2.2.1)` | pin `transformers==4.40.2` + `tokenizers==0.19.1` |
| 6 | benchmark client `ModuleNotFoundError: aiohttp` | `pip install aiohttp tqdm` |
| 7 | `RuntimeError: Parent directory RESULTS does not exist` | `mkdir -p $FORK_DIR/train/RESULTS` before the sweep |

Plus: GitHub blocked on the instance → **rsync the fork + repo from a local machine** (don't `git clone`);
non-gated HF (trace, predictors) via `HF_ENDPOINT=https://hf-mirror.com`.

## What is / isn't a custom artifact
- **No model was trained** in the baseline — Llama-3-8B and the LTR predictor (OPT-125M) are public/pretrained, re-downloadable anywhere. Nothing custom to save.
- **Custom weights appear only when we train PARS** (the contribution): `train/MODEL/results/<run-id>/finetuned/model.safetensors` — that MUST be backed up (local + OSS) once it exists.

## Baseline result (the reproduction proof)
Mean TTFT (ms) — FCFS vs LTR on Llama-3-8B / LMSYS, rates {2..64}:

| rate | FCFS | LTR | LTR speedup |
|---|---|---|---|
| 2 | 50 | 103 | FCFS better (low load) |
| 8 | 137 | 203 | crossover ~here |
| 16 | 17274 | 6044 | **2.9×** |
| 32 | 91582 | 48089 | **1.9×** |
| 64 | 258347 | 157546 | **1.6×** |

**Reproduces the base paper**: under load (rate ≥16), LTR short-job-first scheduling cuts head-of-line
TTFT up to 2.9×. Raw data: `…/deliverables/04-evaluation/baseline-2026-06-22/baseline-results.tgz`.
(Classification baseline skipped — a benchmark-side `IndexError` in the class-10 path; FCFS-vs-LTR is the core.)

## Cost / time
~2-3 h wall (build + 16G download + 18-run sweep) ≈ ¥5-7 on 恒源云 4090 48GB.
With `scripts/setup.sh` a fresh re-run is ~10-15 min of setup + ~40 min sweep.
