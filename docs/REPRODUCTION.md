# Reproduction — environment, tonight quick-start, experiments

Commands adapted from `hao-ai-lab/vllm-ltr` (`train/train.sh`, `benchmarks/bench-lmsys.sh`).
Run top-to-bottom on the **RTX 4090 48GB** (Ada, sm_89). The *running* is just executing scripts;
the friction is the env build + (later) our PARS optimization.

> **Goal for Wednesday:** reproduce **FCFS vs LTR** (and classification) latency-vs-rate curves.
> **Baseline first** — do NOT train PARS until the baseline numbers exist.

---

## Hardware
- **RTX 4090 48GB (Ada, sm_89)** — 48 GB > the 40 GB needed; trains the OPT-125M predictor AND serves Llama-3-8B. A100 not required.
- Compile CUDA kernels for Ada: `export TORCH_CUDA_ARCH_LIST="8.9"` before `pip install -e .`.
- Local 4090-laptop 16 GB = dev/dry-run with a tiny model only (Llama-3.2-1B/3B or 8B-AWQ); cannot train the predictor or run the full 8B benchmark.

---

## TONIGHT quick-start (≈3 h, skip training, use pretrained predictor)
The fastest path to the Wednesday curves — reproduce baseline without spending 0.5–2 h on training.

1. **P0 — build env** (~1 h, or ~10 min from a Docker snapshot). See below.
2. **P1 — download dataset + PRETRAINED predictors** (~20 min). Skips P2 training.
3. **P3 — baseline sweep**: FCFS, then LTR (using the pretrained predictor), then classification.
4. Collect `RESULTS/` → plot latency vs request-rate (the paper's Fig. 3).

> Training the predictor (P2) and the PARS swap come **after** baseline. Tonight = baseline only.
> Tonight is optional: the hard deadline tonight is the Summary PDF; reproduction is for Wednesday.

---

## P0 — Build env (~40–60 min, one-time)
```bash
git clone https://github.com/hao-ai-lab/vllm-ltr.git && cd vllm-ltr
conda create -n vllm-ltr python=3.10 -y && conda activate vllm-ltr
pip install torch==2.2.1 xformers==0.0.25      # CUDA 12.1 wheels
export TORCH_CUDA_ARCH_LIST="8.9"              # <-- Ada (RTX 4090); build kernels for sm_89
pip install -e .                               # builds CUDA kernels (~20–40 min)
huggingface-cli login                          # needs Llama license access
```
⚠️ Needs the CUDA 12.1 toolkit (`nvcc`). Build once → snapshot / Docker image to avoid re-paying on rentals.
⚠️ Ada note: if xformers / flash-attn complain about arch, confirm the wheel supports sm_89; `--enforce-eager` (used below) avoids CUDA-graph issues.

## P1 — Download dataset + (optional) pretrained predictors (~15–20 min)
```bash
cd train
huggingface-cli download LLM-ltr/Llama3-Trace --local-dir jsonfiles --repo-type dataset
# pretrained predictors (skip training tonight):  HF  LLM-ltr/OPT-Predictors
huggingface-cli download LLM-ltr/OPT-Predictors --local-dir MODEL --repo-type model
```

## P2 — Train the predictor (~0.5–2 h, [later / full reproduction])
LTR predictor, OPT-125M, Llama-3-8B / LMSYS (paper Table-3 row, Tau ≈ 0.64):
```bash
cd train
python trainer.py --config configs/config_prefill_opt.txt \
  --file jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
  --job-dir MODEL --run-id opt-125m-llama3-8b-lmsys-score-trainbucket10-b32 \
  --batch-size 32 --label-group-size 10 --loss listMLE
# → MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json
```
- If OOM on 48 GB: drop `--batch-size 32` → `16` (long l8192 sequences drive activation memory, not the 125M weights).
- **PARS hook (our contribution, later):** swap `--loss listMLE` → a pairwise margin loss and set a
  **BERT** `pred_model` in the config. Also add the δ-filter on training pairs (length diff ≥ 0.2). See `docs/references.md`.

## P3 — Benchmark sweep (~1–1.5 h per dataset, 3 methods)
Pattern: launch the server with a `--schedule-type`, wait warmup, sweep rates {2,4,8,16,32,64}
(each `--request-time 60` ≈ 2–3 min). Compare **FCFS / classification / LTR**.

**Baseline FCFS:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests \
  --schedule-type fcfs --enable-chunked-prefill --enforce-eager --port 3343 &
sleep 60
for r in 2 4 8 16 32 64; do
  python benchmarks/benchmark_serving_real.py --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl \
    --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 \
    --request-rate $r --result-dir RESULTS --port 3343
done
kill %1; sleep 60
```

**LTR (ours) — uses the trained / pretrained predictor:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 32 --disable-log-requests \
  --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager \
  --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json &
sleep 60
# same 6-rate sweep with --schedule-type opt-xxx
```

**Classification baseline:** `--schedule-type tpt-class10-xxx` + a `...-class-trainbucket820...` predictor config.

Results land in `RESULTS/` → latency vs request-rate per method = the paper's Fig. 3 plot.

## Expose overfitting (the headline data)
After baseline, evaluate the SAME predictor on a held-out distribution (ShareGPT) and report
**Kendall's Tau (train-distribution) − Tau (held-out)** = the generalization gap. This is the
number PARS is meant to shrink. (Need a ShareGPT trace converted to the same jsonl format.)

---

## Results & data collection — DON'T waste the run
A rented GPU's disk is **ephemeral** — when you stop the instance it is wiped. **Persist everything
BEFORE shutdown / `kill`**, or the training/benchmark is wasted ("白训练"). Save:

- **Raw benchmark data**: the whole `RESULTS/` dir — per-method, per-rate CSV/JSON (FCFS / classification / LTR). This is what the plots are built from; without it you cannot redraw or defend a number.
- **Trained predictor**: `MODEL/results/<run-id>/` incl. `usage_config.json` + weights (so you don't re-train).
- **Training log**: loss curve + **Kendall's Tau on train AND held-out** (the overfitting-gap data — the headline result).
- **Run manifest** (one small `runs/<date>-<run-id>.md` or `.json` per run): fork git commit, vLLM version, CUDA/torch/xformers versions, GPU model, **exact trace filename + dataset**, every flag, start/end time. Makes results reproducible and attributable.
- **Plots**: latency-vs-rate, Tau gap (export PNG + the data behind them).

How to persist (pick one, do it before stopping the box):
- Download via JupyterLab / `scp` to your laptop.
- On AutoDL: write outputs under the persistent `/root/autodl-tmp`, or download before "关机".
- Or `git add` results to a `results` branch (small CSVs/manifests only — not model weights), or push weights to an HF dataset / your own storage.

Rule: **collect → persist → verify the download opened → only then shut down the GPU.**

---

## Time / cost (RTX 4090 48GB)
| Phase | Time |
|---|---|
| Build (or Docker pull ~10 min) | ~1 h |
| Download data + pretrained models | ~20 min |
| Train predictor (later) | ~0.5–2 h |
| Benchmark sweep (3 methods, 1 dataset) | ~1–1.5 h |
| **Tonight baseline (skip training)** | **~3 h** |
| Full (train + both datasets) | ~4–6 h |

## Gotchas
- Repo defaults to **Llama-3-8B**; the FDU base paper used **Llama-3.1-8B**. Stick with 8B-Instruct to reuse the trace + pretrained predictors (re-train on a 3.1 trace only if matching exactly). Decide once and record it.
- `--enforce-eager` (no CUDA graphs) + `--swap-space N` (GB host RAM for preemption) per script.
- `schedule-type` strings: `fcfs`, `opt-xxx` (LTR), `tpt-class10-xxx` (classification), `mlfq-...`, `sjf` (oracle). See `vllm/core/scheduler.py` in the base fork.
- Ada (4090): if a kernel build fails, re-check `TORCH_CUDA_ARCH_LIST="8.9"` and that torch/xformers match CUDA 12.1.
