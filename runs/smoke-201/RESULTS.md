# 201 (x86 Linux, WSL2) pre-rental smoke results — 2026-07-19 (rev 3)

> **Independent review verdict (fresh Codex, 3 rounds): SMOKE-SUFFICIENT-for-pre-rental.**

**rev 3 update — current-HEAD recapture (`artifacts2/`, repo_commit + commands +
exit codes + per-file SHA-256 in `artifacts2/manifest.txt`):** after syncing the
real ToolACE fixture (sha ba12c083 verified on 201), the seed17 checkpoint, and
latex_source, the FULL suite at commit fd6ac99 is **211 passed, 0 failed,
0 errors** (`artifacts2/full-pytest.log`); seam suite **2 passed**
(`artifacts2/seam-pytest.log`). The rev-2 numbers below describe the earlier
capture and its honest caveats; they are retained for provenance.

Host: alex@192.168.8.201 (ssh -p 2222). Raw artifacts in `artifacts/`:
`seam-pytest.log`, `full-pytest.log`, `pip-freeze.txt`, `versions.json`
(python 3.12.3, pydantic 2.13.4, torch 2.11.0+cu130, vllm 0.24.0).

## Scope statement (what this smoke does and does not prove)

201 proves: the pinned dependency set resolves and imports on x86 Linux, the
pinned vLLM protocol layer behaves as the contract assumes, and 191 of 198 collected tests passed with vLLM importable (the declared fixture
blocks failed; rev 3 recapture is fully green). 201 does NOT prove GPU wheel/driver
compatibility, model download/load, real engine serving, custom-scheduler
instantiation, the gateway chain, OSS restore, or any rental-day preflight —
those are enforced by rental-day gates in `scripts/server/` (see the risk
register below).

## 1. Protocol seam (real pinned vLLM protocol objects)

`pytest tests/test_vllm_protocol_seam.py` → **2 passed** (`artifacts/seam-pytest.log`).

- Positive path: `ChatCompletionRequest(vllm_xargs={prediction_reliable: 1, ...})
  → to_sampling_params() → extra_args → GatewayMetadataPredictor` yields the
  reliable prediction with score 512/4096. This exercises the real protocol
  request model, NOT an HTTP→engine→scheduler E2E (that is the rental-day
  two-request gate in `run_matrix.sh`).
- Bool case (documented boundary, not a defense): pinned Pydantic COERCES
  `True → int 1` before our code sees it, so a bool fed directly to the engine
  WOULD be trusted downstream. The trust boundary is the gateway's int
  contract (Go-side: client flags whitelisted away; verdict written as int
  0/1; tested in `internal/ltr/decision_test.go`). The test pins the dangerous coercion outcome while accepting stricter
  upstream behavior (outright rejection also passes); a silent semantic
  change to coercion is what it guards against.

## 2. Dependency resolution (fresh venv, x86 Linux, CPU/WSL)

- Resolved set: vllm 0.24.0 | torch 2.11.0+cu130 | transformers 5.14.1
  (`artifacts/pip-freeze.txt`). torch==2.11.0 matches vLLM 0.24's declared pin.
- Caveat: `+cu130` is the wheel build variant that bare pip selected here;
  vLLM 0.24's default wheel targets CUDA 12.9. Whether this binary combination
  runs on the rental GPU/driver is NOT proven by 201 — `setup_env.sh` now
  hard-gates on `torch.cuda.is_available()` and records
  `/hy-tmp/ltr/manifest.cuda.json` (torch/cuda-build/device) at install time.

## 3. Scheduler classes vs real vLLM (import-level only)

- `_VLLM_AVAILABLE=True`; roster = fcfs, gated_hybrid, ltr_aging, prompt_sjf,
  pure_ltr, tail_safe — consistent with `SCHEDULER_CLASS_TO_POLICY`.
- `issubclass(StockFCFSShim, vllm.v1.core.sched.scheduler.Scheduler)` → OK.
- Custom schedulers were NOT instantiated (constructor/API compatibility with
  a live engine is a rental-day preflight concern).

## 4. Full test suite with vLLM importable

`pytest tests/ -q` → **191 passed, 1 failed, 6 errors** (`artifacts/full-pytest.log`).

- 6 errors: `ToolAceProductionSnapshotTest` class setup raises because the
  real ToolACE snapshot fixture lives at a Mac-local cache path not present on
  201. The setup failure blocks all six tests, so their product assertions
  NEVER EXECUTED — this run cannot certify that code path either way (they
  pass on the Mac where the fixture exists). Known non-hermetic fixture design.
- 1 failed: offline CLI checkpoint probe expected the 417MB seed17 checkpoint
  directory, deliberately not synced to 201. (The test has since been made
  deterministic via SEED=PATH overrides on main.)

## Rental-day residual risk register (only exposable on the rented GPU box)

Enforced by hard gates in `scripts/server/`; none are covered by 201:
GPU wheel/driver compat (setup_env CUDA gate) · /hy-tmp capacity + HF mirror +
pinned Qwen download/load · bf16/8192-ctx/tool-parser serve start ·
custom-scheduler instantiation under a live engine · gateway→decision→vLLM
two-request chain with non-empty order log (run_matrix preflight) · OSS
restore/hash/atomic unpack/checkpoint load · pinned OOD downloads + 800-row
GPU labeling + ledger repair + quantile build · Go toolchain + pinned VeloxMesh
build · concurrency-8 decision latency + timeout calc · saturation grid
bracketing · 7-policy matrix + parity + overhead + budget + upload/readback.
