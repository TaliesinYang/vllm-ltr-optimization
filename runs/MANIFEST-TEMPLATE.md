# Run manifest — <run-id>

Fill one of these per GPU run and commit it (small text only). Pairs with the auto-generated
`<run-id>-manifest.txt` from `scripts/collect_results.sh`.

## What / when
- **Run id:** <date-purpose, e.g. 20260623-baseline-lmsys>
- **Date (UTC):**
- **Who ran it:**
- **Purpose:** baseline reproduction | overfitting gap | PARS | …

## Environment
- **GPU / VRAM:** (e.g. RTX 4090 48GB · L20 48GB · A800-80GB)
- **Platform / cost:** (AutoDL / 智星云 / Vast.ai · ¥or$ /hr · total)
- **fork commit:** (hao-ai-lab/vllm-ltr HEAD)
- **vLLM / torch / CUDA / xformers:**
- **TORCH_CUDA_ARCH_LIST:** (8.9 Ada / 8.0 Ampere / 9.0 Hopper)

## Experiment
- **Served model:** meta-llama/Meta-Llama-3-8B-Instruct
- **Predictor:** OPT-125M listMLE (pretrained / trained) · run-id:
- **Dataset / trace file:** lmsys-…-c10000-rFalse.jsonl  (+ held-out: sharegpt-… if any)
- **Methods swept:** fcfs / opt-xxx (LTR) / tpt-class10-xxx (class) / sjf (oracle)
- **Rates:** 2 4 8 16 32 64 · request-time 60

## Results (fill from RESULTS/)
- **Kendall's Tau** — train: ____  held-out: ____  **gap: ____**
- **Mean / P90 / P99 per-token latency** (per method, per rate): see attached CSVs
- **Predictor overhead %:**
- **Plots:** <path to png>
- **Artifacts saved to:** <download location> (RESULTS.tgz / predictor-configs.tgz / manifest.txt)

## Notes / anomalies
-
