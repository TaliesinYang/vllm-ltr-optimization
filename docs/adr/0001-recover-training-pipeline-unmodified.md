# Recover the tier-2 training pipeline unmodified, defects included

The code that produced the nine-run tier-2 matrix existed only as uncommitted working-tree
state in a git worktree and was absent from every branch, stash and archive. We committed it
to `main` (PR #3) **exactly as it ran**, without repairing the defects an independent review
found in it, because this commit's purpose is provenance: it must record what actually
executed on the RTX 3090, and editing it would destroy the only evidence of that.

## Considered options

- **Fix first, then commit.** Rejected. The nine runs already happened. Patching the source
  now would mean the committed code no longer corresponds to the code that produced the
  reported numbers, which is strictly worse for artifact evaluation than committing known-
  imperfect code.
- **Leave it in the worktree.** Rejected. A single `git clean -fd` would have destroyed it
  permanently, and the published repository could not reproduce its own headline result.
- **Commit as-is, fix in a follow-up.** Chosen.

## Consequences

Defects that an independent review (Codex, verdict REQUEST-CHANGES) confirmed and that are
now knowingly present in `main`:

- `ltr_training/tier2_training.py:162` passes `sample_path` as `labels_path`, so the run
  manifest hashes the unlabeled sample rather than the replay ledger that supplies every
  training target. The manifest also omits ledger hash, git SHA, initial-checkpoint hash,
  hyperparameters and environment versions — and only `run_work/final` is copied into
  results, so shipped finals carry no manifest at all.
- `ltr_training/training_matrix.py:101` and `tier2_training.py:139` accept any directory
  containing `validation_metrics.json` as a completed run without checking config, input
  hashes, variant, seed or code identity. A stale or mixed matrix still reports
  `completed_runs=10`.
- `scripts/verify_gpu_env.sh:13` writes to `/hy-tmp/logs` before anything creates it, so a
  fresh bootstrap fails under `set -euo pipefail`.
- `ltr_training/train_ranker.py:181,225` create checkpoint directories with `exist_ok=False`,
  so an interrupted run cannot resume or cleanly restart.
- `requirements/train.txt:57` pins Torch 2.13.0; the reported runs used 2.10.0.

Because these are recorded rather than repaired, any future fix must be a separate commit
that explicitly states it changes behaviour relative to the runs reported in the paper.
