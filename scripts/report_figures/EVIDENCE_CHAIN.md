# Evidence Chain: data → figure → report claim (fig4/6/7/8)

Authoritative spec for the Evaluation figures. Every claim below is traceable to
a raw data file. Rigor bar = **top-journal**: pooled percentiles (never average
per-run p99), bootstrap 95% CI (≥2000 resamples), colorblind-safe palette,
matplotlib-only (IEEE hard rule), vector output. AI generates figures + organizes
data ONLY — all prose is human-written (AI prose = 0 credit).

## Data locations (absolute)
- Live matrix (this rental, qwen3.5-9b, full chain client→gateway→decision→vllm):
  `SCRATCH/results/rental-20260719T231309Z/`
  - MIXED: `matrix/<Policy>.runs/*.samples.csv` (3 repeats/policy × 150 rows; cols:
    request_id,ttft_ms,ttlt_ms,output_tokens,baseline_service_ms,...,category,policy,profile,error)
  - OOD:   `matrix-ood/<Policy>.runs/*.samples.csv` (3 × 120 rows, same cols)
  - Overhead: `gateway-overhead.json` → `.direct.samples[]` and `.gateway.samples[]`
    (150 each, paired by request_id; cols same as samples.csv)
  where SCRATCH = the extracted `results-full-rental-20260719T231309Z.tar.gz`
  (also durable at `/Volumes/T7 Shield/vllm-ltr-results/`).
- Gate probe (project repo):
  `project/gateway_policy_probe/`
  - `results_hybrid10/hybrid_matrix.csv` (1000 rows: tool_ratio,qps,seed,policy,count,
    mean_wait,p95_wait,p99_wait,max_wait,starvation_count,mean_speedup_vs_fcfs,
    p95_speedup_vs_fcfs,p99_ratio_vs_fcfs) — corruption-severity sweep
  - `results_clite/gated_run_{gated,ungated}.jsonl` (cols: request_id,cls,true_tokens,
    pred_est,send_index,first_token_index,mode) — wrong-hint response probe
  - `results_clite/gated_verdict.json` — kendall-tau summary

---

## FIG 4 — OOD robustness: LTR keeps its advantage out-of-distribution
- **Data**: `matrix-ood/*.runs/*.samples.csv`, pool the 3 repeats per policy
  (StockFCFSShim, PureLTRScheduler, GatedHybridScheduler, TailSafeScheduler);
  drop rows with non-empty `error` (1–2 benign BFCL-irrelevance each).
- **Plot**: grouped bars per policy — mean TTLT and pooled p99 TTLT (ms) — with
  bootstrap 95% CI whiskers (resample the pooled per-request vector). Two metric
  groups (mean, p99), 4 policies each.
- **Claim supported**: On this OOD workload, the LTR family improves mean TTLT by
  17.4–18.6% vs shim-FCFS. Pooled-p99 improvement is more modest and uneven:
  8.2% for PureLTR, 17.7% for gated/tail (pooled p99, not the mean of run-level
  p99). Scope the claim to this workload, not broad OOD generalization.
- **Honest caption note**: gate policies ≈ PureLTR here → this OOD set did not
  trigger predictor collapse; gate's protective value is shown in Fig 7, not here.

## FIG 6 — Main result (MIXED): LTR beats FCFS under saturation
- **Data**: `matrix/*.runs/*.samples.csv`, pool 3 repeats, all 7 policies
  (stock_fcfs, StockFCFSShim, PureLTR, GatedHybrid, TailSafe, LTRAging,
  PromptLengthSJF). 0 errors.
- **Plot A (headline)**: CCDF (log-y survival) of TTLT overlaying the 2 FCFS
  baselines (grey) vs the 5 scheduling policies (color) — tail separation is the
  story; annotate p99 crossings.
- **Plot B (companion)**: grouped bars mean + pooled p99 TTLT with bootstrap 95%
  CI, 7 policies, FCFS baselines visually grouped apart from LTR family.
- **Claim**: LTR improves mean TTLT 14.8–15.3% vs real stock_fcfs; pooled-p99
  improvement is 8.5–18.2% across learned policies (PureLTR 8.5%, gated 14.1%,
  tail 14.4%, aging 18.2%, Prompt SJF reaches 19.8% — pooled p99, not the mean of
  run-level p99). Clean attribution (constant gateway overhead across all policies).
- **Honest note**: PromptLengthSJF matches learned policies → gains attributable
  to short-job prioritization generally, not to learned prediction specifically.

## FIG 7 — Gate value: protection under prediction corruption (THE novelty)
- **Panel A — wrong-hint response** (`results_clite/gated_run_*.jsonl` +
  `gated_verdict.json`): grouped bars of Kendall τ(pred_est vs first_token order)
  split by class (chat = accurate hints, tool = wrong hints) × mode (ungated,
  gated). Story: ungated-tool τ ≈ +1 (server obeys wrong hints) → gated-tool
  τ≈0.07 point estimate, but n=6 so the CI is wide [−0.78,+1.0] (do NOT call it
  categorically "decoupled"); chat τ preserved ≈ 1 in both. Recompute τ from the
  jsonl, don't just replot the summary.
- **Panel B — corruption-severity sweep** (`results_hybrid10/hybrid_matrix.csv`):
  x = tool_ratio (0→1, fraction of requests carrying wrong hints), two lines
  gated_hybrid vs pure_ltr; y = MEDIAN per-run p99_ratio_vs_fcfs (NOT pooled;
  median over 10 seeds × 4 QPS, lower = safer tail). Shade 95% bootstrap CI over
  seeds. Second y-axis or twin panel for mean_speedup_vs_fcfs. Story: gated's
  median per-run p99 ratio is below PureLTR's at every tested corruption level;
  PureLTR's tail risk is non-monotonic (peaks at 25% wrong hints, not monotonic).
- **Claim**: the gate is a predefined workload-class fallback (`tool→fallback`),
  not online unreliability detection. Panel A: on tool traffic with wrong hints the
  service order's correlation with the wrong prediction drops from τ≈+1 (ungated)
  to a τ≈0.07 point estimate (gated), though n=6 leaves a wide CI [−0.78,+1.0], so
  this is suggestive not conclusive; accurate chat predictions are preserved (τ=1).
  Panel B: gated
  has a lower simulated per-run p99 ratio than PureLTR at every corruption level,
  but still exceeds FCFS through 75% corruption (ratio 1.16–1.41) — it does not
  meet a ≤1.1 "safe" bar. This supports "gate, not a better predictor" as a
  mechanism, not a full-scale latency guarantee.
- **Honest scope note**: Panel A is a live ordering probe (opt-125m, τ not
  latency, n=6/class); Panel B is SIMULATION (10 seeds × 4 QPS, median per-run
  p99, not pooled). Full-scale live latency under corruption (qwen-9B) is future
  work. Do not present this as full-scale live latency evidence.

## FIG 8 — Gateway overhead: honest infrastructure cost
- **Data**: `gateway-overhead.json` `.direct.samples[]` vs `.gateway.samples[]`,
  paired by request_id (150 each, FCFS-vs-FCFS).
- **Plot A (clean result)**: paired TTFT direct vs gateway — violin or paired
  strip + median line. Headline +737 ms (mean) / clean because TTFT is measured
  before generation non-determinism matters.
- **Plot B (honest TTLT)**: TTLT direct vs gateway, but FIRST flag that 32/150
  requests have mismatched output_tokens between arms (compute + annotate n).
  Show TTLT overhead only on the matched-output subset for a clean estimate, and
  state the marginal-p99 diff (+9482 ms) is confounded, not reported as overhead.
- **Claim**: gateway + CPU-BERT decision adds ≈ +737 ms TTFT (clean). Discussion
  MUST state: this overhead currently EXCEEDS the ~500 ms scheduling benefit, so
  the prototype improves scheduling *inside* VeloxMesh while remaining slower than
  direct vLLM FCFS end-to-end; net-positive deployment needs a faster decision
  path (GPU/cached/async prediction).

---

## Already-done figures (do not regenerate)
fig1 arch, fig2 static-component arch, fig3 midterm repro (2.86×/8.2×),
fig5 predictor τ=0.642 + learning curve. See MATERIALS.md / latex_source/.

## Styling contract (all figures)
- matplotlib only; output BOTH 300-dpi PNG (for review iteration) and PDF (for LaTeX).
- IEEE column width: single 3.5in, double 7.16in. Pick per figure; state in filename.
- Palette: Okabe-Ito colorblind-safe. FCFS baselines in neutral grey, LTR/gate in color.
- Fonts: consistent family, tick labels ≥7pt, axis labels ≥8pt, readable at print size.
- Every point estimate carries a bootstrap 95% CI (≥2000 resamples). Define CI in caption.
- Panels labelled (a)/(b); axis labels carry units; no chartjunk; minimal gridlines.
- Deterministic: fix numpy seed for the bootstrap so figures are reproducible.
