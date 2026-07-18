# BERT Predictor Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the real prompt-schema BERT checkpoint to the Predictor protocol and `/v1/decision`, then prove the CPU path with a real ToolACE request.

**Architecture:** `BertPredictor` owns local checkpoint loading and exact training-side rendering. `DecisionApplication` transports raw admission-time prompt/schema text in existing predictor metadata. The CLI selects stub or BERT without changing scheduler policies or the benchmark runner.

**Tech Stack:** Python 3.11, PyTorch CPU, Hugging Face Transformers, pytest, stdlib HTTP server/client

---

### Task 1: Pin the exact BERT predictor contract

**Files:**
- Modify: `tests/test_predictor.py`
- Modify: `scheduler_benchmark/predictor.py`

- [x] **Step 1: Write RED tests** for the exact string
  `"[USER]\ncurrent prompt\n[TOOLS]\nraw schema"`, tokenizer kwargs
  `padding=True`, `truncation=True`, `max_length=512`,
  `return_tensors="pt"`, sigmoid direction, fixed uncalibrated confidence,
  `ood=False`, and missing raw-feature rejection.
- [x] **Step 2: Verify RED** with
  `python -m pytest tests/test_predictor.py -q`; failures must be caused by the
  missing `BertPredictor` behavior.
- [x] **Step 3: Implement GREEN** in `scheduler_benchmark/predictor.py`: local
  checkpoint load, CPU/eval mode, exact renderer, inference mode, sigmoid score,
  honest placeholders, and full per-predict wall-clock timing.
- [x] **Step 4: Verify GREEN** with the same focused test command.

### Task 2: Transport exact raw inputs and select BERT from the CLI

**Files:**
- Modify: `tests/test_decision_service.py`
- Modify: `tests/test_decision_service_cli.py`
- Modify: `scheduler_benchmark/decision_service.py`
- Modify: `scripts/run_decision_service.py`

- [x] **Step 1: Write RED tests** proving that the final message content and
  exact system content enter `PredictorInput.metadata`, and that CLI parsing
  retains `stub` while accepting `bert` plus the real checkpoint path.
- [x] **Step 2: Verify RED** with
  `python -m pytest tests/test_decision_service.py tests/test_decision_service_cli.py -q`.
- [x] **Step 3: Implement GREEN** by enriching `_predictor_input()` without
  altering existing token serialization, then adding a testable CLI predictor
  factory and dynamic feature-variant default (`stub` keeps the existing
  default; BERT uses `prompt_schema`).
- [x] **Step 4: Verify GREEN** with the same focused command.

### Task 3: Prove the real CPU direct and HTTP paths

**Files:**
- No repository file changes; run the existing production APIs directly.

- [x] **Step 1: Download or reuse the pinned ToolACE `data.json` outside the repo**
  and verify its SHA-256 is
  `ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e`.
- [x] **Step 2: Run a one-shot smoke through `iter_toolace_invocations`, the
  real checkpoint, a local decision server, and an HTTP POST using the same raw
  prompt/schema.** Print both direct and HTTP JSON results.
- [x] **Step 3: Require** a
  finite `[0,1]` score, `confidence=0.9`, `ood=false`, non-negative latency, HTTP
  200, and `reason_code="prediction_reliable"`.

### Task 4: Completion audit

**Files:**
- Review only: all files changed above

- [x] **Step 1: Run focused tests** for predictor and decision-service changes.
- [x] **Step 2: Run the full suite** with the proven Python 3.11 dependency set;
  expected baseline is 96 tests plus five new tests.
- [x] **Step 3: Inspect `git diff`** and confirm no changes to
  `scheduler_benchmark/policies.py`, `scheduler_benchmark/runner.py`, or any
  `ltr_training/` file, and no overlap with pre-existing user changes.
- [x] **Step 4: Audit A/B/C requirement-by-requirement** against real test and
  smoke output before claiming completion.
