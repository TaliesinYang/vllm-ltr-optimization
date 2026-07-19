# 201 (x86 Linux, WSL2) pre-rental smoke results — 2026-07-19

Host: alex@192.168.8.201 (ssh -p 2222), Python 3.12.3, venv /home/alex/ltr-seam/.venv-server

## 1. Protocol seam (real pinned vLLM)
- venv: vllm==0.24.0 installed from PyPI
- `pytest tests/test_vllm_protocol_seam.py -v` → **2 passed** (int contract survives
  ChatCompletionRequest -> to_sampling_params -> extra_args -> GatewayMetadataPredictor;
  bool flag never trusted)

## 2. requirements/server.in dependency resolution (fresh venv)
- Resolved: vllm 0.24.0 | torch 2.11.0+cu130 | transformers 5.14.1
- Matches vLLM 0.24's torch pin — validates rental-day setup_env.sh install step.

## 3. Scheduler classes vs real vLLM
- `_VLLM_AVAILABLE=True`; roster = fcfs, gated_hybrid, ltr_aging, prompt_sjf, pure_ltr, tail_safe
- `issubclass(StockFCFSShim, vllm.v1.core.sched.scheduler.Scheduler)` → OK

## 4. Full test suite with real vLLM present
- `pytest tests/ -q` → **191 passed, 1 failed, 6 errors**
- 6 errors: ToolAceProductionSnapshotTest requires the real ToolACE snapshot at a
  Mac-local cache path (fixture not synced to 201) — environment, not code.
- 1 failed: test_offline_cli checkpoint probe expects seed17 checkpoint dir (417MB,
  deliberately not synced) — environment, not code.
- All engine/scheduler/decision/transport/runner tests pass with vLLM importable.
