# Data And Offline Evidence R1 Implementation Plan

> **For agentic workers:** Execute inline with test-driven development. Do not commit, and do not modify `scheduler_benchmark/*`.

**Goal:** Build the pinned BFCL/Toolathlon conversion, workload, offline-scoring, statistics, and baseline scripts needed for tomorrow's server run while preserving honest provenance and blocked states.

**Architecture:** New code lives under `ltr_training/` and thin CLIs under `scripts/`. Source conversion produces a dedicated `LabelInput` JSONL plus manifests; later stages consume those immutable IDs and write independent JSON/JSONL evidence. Existing scheduler, predictor service, and label files remain unchanged.

**Tech Stack:** Python 3.11, dataclasses, hashlib/json, scipy, torch/transformers, optional LightGBM, pytest.

---

### Task 1: Pins, LabelInput, and OOD conversion

**Files:**
- Create: `configs/source-declarations.json`
- Create: `ltr_training/label_input.py`
- Create: `ltr_training/ood_conversion.py`
- Create: `scripts/build_ood_label_inputs.py`
- Create: `tests/test_ood_conversion.py`
- Modify: `tests/test_source_manifest.py`

- [ ] Add failing tests that require the two exact mainline revisions, a distinct `LabelInput`, BFCL first-turn extraction, Toolathlon first-assistant-only extraction, stable IDs, source-identity wording, and manifest row/task/input-hash counts.
- [ ] Run `python -m pytest tests/test_source_manifest.py tests/test_ood_conversion.py -q` and confirm failures are caused by missing production code/config.
- [ ] Implement canonical JSON/schema hashing, robust JSON-or-JSON-string decoding, deterministic sampling, and atomic JSONL/manifest publication.
- [ ] Re-run the focused tests and keep them green.

### Task 2: Workload builder

**Files:**
- Create: `ltr_training/workload_builder.py`
- Create: `scripts/build_offline_workload.py`
- Create: `tests/test_workload_builder.py`

- [ ] Add failing tests for `request_id == sample_id`, ID/OOD/mixed profiles, `baseline_service_ms = output_length * per_token_ms`, `max_tokens = 4096`, category propagation, and an explicit proxy manifest.
- [ ] Run the focused test and confirm RED.
- [ ] Implement the builder without importing or editing `scheduler_benchmark`.
- [ ] Re-run the focused test and confirm GREEN.

### Task 3: Ensemble scoring and diagnostics

**Files:**
- Create: `ltr_training/offline_scoring.py`
- Create: `scripts/score_offline_ensemble.py`
- Create: `tests/test_offline_scoring.py`

- [ ] Add failing tests for exact prompt-schema rendering, per-checkpoint SHA-256, batched scalar scores, percentile ranks within domain, rank dispersion, `risk = 1 - tau_b`, diagnostic naming, and 512-token truncation ratios.
- [ ] Run the focused test and confirm RED.
- [ ] Implement injectable scoring for unit tests and a transformers-backed server CLI for seeds 17/42/73.
- [ ] Re-run the focused test and confirm GREEN.

### Task 4: Statistical and leakage evidence

**Files:**
- Create: `ltr_training/offline_statistics.py`
- Create: `scripts/analyze_offline_evidence.py`
- Create: `tests/test_offline_statistics.py`

- [ ] Add failing tests for Kendall tau-b, seeded 1000-draw session/task cluster bootstrap, separate true/prediction ties, canonical schema hashes, and the three requested split-overlap reports.
- [ ] Run the focused test and confirm RED.
- [ ] Implement JSON reports with percentile confidence intervals and explicit cluster-unit metadata.
- [ ] Re-run the focused test and confirm GREEN.

### Task 5: LightGBM and legacy P2

**Files:**
- Create: `ltr_training/offline_baselines.py`
- Create: `scripts/tune_lightgbm_offline.py`
- Create: `scripts/score_legacy_predictors.py`
- Create: `tests/test_offline_baselines.py`

- [ ] Add failing tests for a roughly 20-combination grid, validation-only selection, one-time test evaluation, per-family OPT/BERT/classification descriptors, longer-is-higher score normalization, and typed blocked output when weights are absent.
- [ ] Run the focused test and confirm RED.
- [ ] Implement the LightGBM runner and non-fabricating legacy loader skeleton.
- [ ] Re-run the focused test and confirm GREEN.

### Task 6: Spec, smoke artifacts, and verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-data-offline-spec.md`
- Generate: `runs/offline-evidence-r1-local-smoke/*`

- [ ] Revise the spec to match the approved decisions and P2 legacy status.
- [ ] Run every CLI on small fixtures and persist manifests/reports under the smoke directory.
- [ ] Run `python -m pytest tests/ -q` and inspect the complete result.
- [ ] Search the changed implementation for writes under `scheduler_benchmark/`; report any violation as blocked.
- [ ] Report `ARTIFACT n: done/blocked-因为X`, tests, and evidence paths without committing.
