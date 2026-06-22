# End-to-end reproduction workflow

One runbook from **rent → deploy → run → retrieve**, with a verification GATE after each phase
(don't proceed until the gate passes). Target: reproduce **FCFS vs LTR** latency curves on
Llama-3-8B / LMSYS. **Baseline first; no predictor training tonight** (uses pretrained predictors).

Reviewed against the real `hao-ai-lab/vllm-ltr` code — see [§ Audit](#audit) for what's verified
and the residual risks to watch.

---

## Phase 0 — Pre-flight (local + at rental)
**Local (already done):**
- [x] `oss` CLI installed + `oss login` (verify: `oss ls -s oss://` lists folders)
- [x] repo pushed to `github.com/TaliesinYang/vllm-ltr-optimization`

**At rental (恒源云):**
- [ ] **GPU = RTX 4090 48GB** (Ada) — NOT 5090/PRO6000 (Blackwell won't build the old fork)
- [ ] **Image = PyTorch + CUDA 12.1** (driver supports 12.1 runtime)
- [ ] **Data disk expanded to ~100GB** (system disk is only 20G)
- [ ] (optional) HF token ready — *not needed* if using ModelScope for the model

**GATE:** GPU is Ada + 48GB, image has CUDA 12.x, data disk ≥100G. Else stop and re-pick.

---

## Phase 1 — Deploy (build env + get data)  ~30–60 min
On the instance (JupyterLab terminal or SSH):
```bash
source /etc/network_turbo                       # 恒源云 proxy: GitHub + HF
cd /hy-tmp
git clone https://github.com/TaliesinYang/vllm-ltr-optimization.git
cd vllm-ltr-optimization
TORCH_CUDA_ARCH_LIST=8.9 bash scripts/hy_deploy.sh
```
`hy_deploy.sh` → clones the fork, builds vLLM kernels (Ada), downloads:
- Llama-3-8B ← **ModelScope** → `/hy-tmp/models/Meta-Llama-3-8B-Instruct`
- LMSYS trace + pretrained predictors ← **hf-mirror.com** → `train/jsonfiles`, `train/MODEL`

**GATE (before paying for a long run, verify the build + data):**
```bash
conda activate vllm-ltr
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
ls /hy-tmp/models/Meta-Llama-3-8B-Instruct/config.json     # model present
ls /hy-tmp/vllm-ltr/train/MODEL/results/                   # predictor run-ids present (note exact names!)
ls /hy-tmp/vllm-ltr/train/jsonfiles/ | head                # trace jsonl present
```
- `torch.cuda.is_available()` must be `True`.
- If `MODEL/results/` names differ from the script defaults, edit `LTR_CFG`/`CLS_CFG` in `run_baseline.sh`.

---

## Phase 2 — Run baseline (tmux)  ~1–1.5 h
```bash
tmux new -s run
oss login                                       # for auto-upload
cd /hy-tmp/vllm-ltr-optimization
bash scripts/hy_run_and_upload.sh baseline-1
# Ctrl+B then D to detach — safe to close the laptop
```
This runs FCFS → LTR → (classification) sweeps over rates {2,4,8,16,32,64}, collects results,
`oss cp` to `oss://backup/`, then `shutdown` (stops billing).

**GATE (watch the first method before detaching):** FCFS server starts, the first `request-rate 2`
run prints latency numbers into `RESULTS/`. If the server OOMs or the predictor config errors, fix
before letting all 18 runs proceed.

---

## Phase 3 — Retrieve + check (local)
```bash
oss cp oss://backup/baseline-1-all.tar.gz . && tar xzf baseline-1-all.tar.gz
```
You get: `RESULTS/` (raw per-rate CSV/JSON), predictor configs, and a manifest (versions/flags).

**GATE:** plot latency vs request-rate per method. **LTR curve below FCFS = reproduction succeeded.**
Fill `runs/MANIFEST-TEMPLATE.md` with the numbers + Kendall's Tau.

---

## Cost / time
| Phase | Time | Notes |
|---|---|---|
| Deploy (build + data) | ~30–60 min | build dominates; data ~10–20 min on 800 Mbps |
| Baseline sweep | ~1–1.5 h | 3 methods × 6 rates × ~2–3 min |
| **Total (baseline, no training)** | **~2.5–3 h** | ≈ ¥7–10 on 恒源云 4090 48GB |
| (+ train predictor, later) | +0.5–2 h | for PARS; code not written yet |

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| `git clone` / HF slow | `source /etc/network_turbo` first; model already via ModelScope |
| kernel build fails (Ada) | confirm `TORCH_CUDA_ARCH_LIST=8.9`, torch 2.2.1 + CUDA 12.1 |
| disk full | everything must be on `/hy-tmp` (HF_HOME/conda/model already pinned there); expand data disk |
| `torch.cuda.is_available()` False | wrong image/driver — re-pick a CUDA 12.x image |
| LTR run skipped ("config missing") | `ls train/MODEL/results/` → set real run-id in `LTR_CFG` |
| 8B server OOM | lower concurrency / raise `--swap-space`; 48G should be fine for 8B |
| wrong python (`vllm` not found) | `conda activate vllm-ltr` (run_baseline.sh does this, but verify) |

---

## Audit
**Verified against `hao-ai-lab/vllm-ltr` source (✅):**
- `trainer.py` args: `--config --file --job-dir --run-id --batch-size --label-group-size --loss`; `listMLE` is a real loss (`allrank.models.losses.listMLE`).
- `benchmark_serving_real.py` args: `--backend --model --tokenizer --dataset --num-prompts --request-time --request-rate --output-len --result-dir --port`.
- `api_server` args: `--schedule-type --prefill-predictor-model-config --swap-space --enable-chunked-prefill --enforce-eager`.
- `scheduler.py` parses schedule types by **prefix** (`startswith`): `fcfs`, `opt` (→`opt-xxx`), `tpt` (→`tpt-class10-xxx`), `mlfq`, plus `sjf`. All script values are valid.
- Predictor config `config_prefill_opt.txt` = `facebook/opt-125m`, `mtype: rank`. PARS hook = swap `pred_model`→BERT + `--loss` → pairwise.

**Found + fixed during review (🔧):**
- **Dataset path bug** — `benchmark_serving_real.py` calls `open(dataset_path)` literally, but `Llama3-Trace` downloads under `jsonfiles/`; a bare filename in CWD would not be found → the whole sweep would fail. `run_baseline.sh` now resolves the real path via `find` and aborts early if missing.
- **conda not active in tmux** — `run_baseline.sh` now `conda activate`s the env.
- **LTR predictor-config guard** — skips LTR (FCFS still runs) and prints actual `MODEL/results/` names if the config path is wrong.

**Residual risks (cannot verify without the GPU/download — gated in the runbook):**
1. **Pretrained predictor run-id names** in `LLM-ltr/OPT-Predictors` must match `LTR_CFG`/`CLS_CFG`. The authors' `bench-lmsys.sh` uses these exact names, but confirm at the Phase-1 gate (`ls MODEL/results/`).
2. **`trainer.py` default tokenizer = Llama-3-70B** — for PARS *training* (later) pass `--tokenizer meta-llama/Meta-Llama-3-8B-Instruct`. Irrelevant to the baseline (no training).
3. **conda on the image** — scripts assume conda; if the image uses system python, skip `conda activate` and install into base. `run_baseline.sh` warns but continues.
4. **ModelScope model dir** must be HF-format (has `config.json` + tokenizer) — checked at the Phase-1 gate.
5. **Untested end-to-end** — first run may need a longer warmup `sleep` or a port change; the Phase-2 gate catches this before all 18 runs.
