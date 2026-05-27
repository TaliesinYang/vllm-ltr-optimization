# vllm-ltr-optimization

Graduate capstone (CSCI 6806) — reproducing and optimizing **learning-to-rank (LTR) based scheduling for LLM inference**.

## Goal
Reproduce the LTR-to-approximate-SJF scheduler (Fu et al., NeurIPS 2024 `vllm-ltr`) and the FDU empirical study built on it, then improve the predictor's **generalization** (the admitted overfitting weakness) using modern pairwise-ranking / uncertainty-aware / hidden-state-reuse techniques, evaluated under a unified benchmark harness.

## Status
🚧 Initialized — scaffold only. Research notes and reproduction code to follow.

## Structure
```
docs/         design notes, related-work analysis, reproduction plan
experiments/  benchmark configs, results, evaluation scripts
src/          predictor + scheduler modifications
```

## Team
Private capstone repo. See repo settings for collaborators.
