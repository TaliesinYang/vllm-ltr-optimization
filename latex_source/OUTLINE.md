# Paper outline — Schema-Aware Length Ranking with an Abstaining Reliability Gate for Agentic LLM Serving

Course: CSCI 6806 Final Report (IEEE 2-col, per-section page caps, ≥6 eval figs, ≥2 tables).
Rigor bar: Ryoo standard — every claim has a number, every number has a file, conclusions
only after results. Prose: drafted-for-revision; user rewrites in own voice.

Numbers of record: SYNC-CHECKLIST-2026-07-26.md · DIRECTION-DECISION-2026-07-25.md ·
runs/offline-experiments-2026-07-25/REPORT.md. Forbidden claims list applies throughout
(no "first tool-aware scheduler"; never ≥ JITServe; PARS is not ours; p99 reported-not-claimed).

---

## Abstract (¼ page, exactly 2 paragraphs, must contain 1–2 quantitative results)

P1: Agentic clients dominate modern LLM traffic, and their requests are structurally
unlike chat: in our capture of a production coding agent, tool-schema text is 69% of
request bytes, a third of requests carry no tools at all, and per-client schemas are
byte-constant. Serving stacks reduce this structure to scalar features. Problem: can the
schema TEXT itself inform scheduling?

P2: System = BERT Ranker over prompt+schema text + slot-preserving Reliability Gate whose
per-stratum confidence is measured, not assumed + gateway integration. Numbers to quote:
+0.19 Kendall τ over a tuned scalar baseline; τ holds at 0.63–0.64 on requests whose
tools were never seen in training (identity/lookup baselines collapse to no-signal);
[SLOT: one Block-1/Block-2 serving number if favorable, e.g. decision-path p50 1.1 s → 38 ms].

## 1. Introduction (1 page; bullets allowed here only)

- Para 1 (hook): the schema-share measurement — a real agent request is mostly tool
  documentation the serving layer never reads. (fig: none; cite our capture.)
- Para 2 (gap): schedulers want output-length order (SJF-style); learned predictors read
  prompts only or scalars; agent traffic breaks both assumptions (33% zero-tool, schema
  constancy, unseen tool sets are the norm — 95.5% of our test split).
- Para 3 (approach): read the request as text; gate what cannot be trusted; keep the
  ranker off the critical path.
- Contribution bullets (4):
  C1 workload characterization of live agent traffic (69% schema share; zero-tool third;
      per-client constancy; all-S4 vs foreign vocabulary);
  C2 schema-text ranking: +0.19 τ vs tuned scalars, cold-start flat across strata,
      identity baselines die (+0.008, memorisation when forced);
  C3 honest negatives: un-truncation Δτ≈0; caching 1.9× but 4–25× over budget; ⇒ async
      side-path design conclusion;
  C4 evidence-based abstaining gate: per-stratum τ lower-bounds replace a hardcoded 0.9
      that overstates by +0.25–0.46 everywhere; hardest stratum is NEW COMBINATIONS of
      seen tools (τ 0.44), not new tools.
- Para 5: results preview + paper map. [SLOT: one sentence on Block-1 verdict, either
  wording pre-drafted in RENTAL-DAY runbook §4.]

## 2. Background (1.5 pages, ≥2 figures)

2.1 Serving and scheduling primer: continuous batching, FIFO HOL blocking (cite Park
    survey §II-F for independent warrant), SJF needs length estimates.
2.2 Learned length prediction line: S³ → LTR (Fu, our root baseline) → PARS
    (tao2026ranking — ISC'26, peer-reviewed). One sentence each, what input each reads.
2.3 The system substrate: VeloxMesh gateway (teammate's platform) — 15 ms scheduler
    contract, fail-open breaker; decision service seam (/v1/decision).
Figures: fig1 (system architecture, existing), fig2 (static component diagram, existing).
Terms defined here: Ranker, Reliability Gate, Fallback, Feature Variant, Ranking Tau,
Cold-Start Transfer (S1–S4), Cross-Workload Transfer (from CONTEXT.md glossary, verbatim
definitions).

## 3. Related Work (1 page)

Three lanes + a fence:
- Prediction lane: S³/µ-Serve/TRAIL/LTR/PARS — none read schemas.
- Prediction-free lane: UniBoost (ICML'26) beats even oracle SRPT on P99 TTLT (Table 2:
  35.1%; NEVER cite abstract's 35–50%) — this closes tail-latency claims for predictors;
  our endpoints are mean/goodput; note UniBoost has zero agentic evaluation, prefix
  caching disabled, and "prediction-free ≠ feature-free" (it uses prompt length).
- Imprecise-info lane: JITServe (NSDI'26) — QRF quantile upper bounds refined online,
  7 ms predictor (their bar), tool info as identity+duration only, never schema text;
  their §7 names "explicit fallback policies" as the open gap; our gate is the
  complementary mechanism (abstain vs always-pessimistic). Do not claim parity.
- Routing neighbor: Switchcraft — schemas→DistilBERT for model CHOICE, not length; its
  own PDF: benchmark-shift quantification "remains future work" = exactly our Cold-Start
  Transfer stratification.
- Split-design precedent fence: ToolLLM I1/I2/I3 (unseen instructions/tools/categories)
  licenses the DESIGN of S1–S4, not the metric (they measure pass/win rate). BFCL's
  crowd-sourced split is contamination control, separate sentence. DO NOT cite Gorilla
  for unseen splits (random holdout; "zero-shot" = no retriever).

## 4. Methodology (1 page, ≥2 tables)

4.1 Data + labels: ToolACE tier-2 6k (3 declared exclusions → 5997), fixed per-row split
    3997/998/999; two-tier labeling; censor 0.05%.
4.2 Ranker: BERT-base, [USER]prompt[TOOLS]schema render, pairwise margin recipe, 3 seeds.
    Feature Variants table (prompt_only / prompt_schema / full_context — full_context arm
    void: 80% token-identical after 512 truncation; disclose).
4.3 Baselines: LightGBM scalars (fixed AND grid-searched 20-config; grid = baseline of
    record 0.4395; recipe deterministic w.r.t. seed — verified by rerun, not "low
    variance"); schema-identity hash + out-of-fold lookup (the deployed-lookup-table
    emulation; never hash raw schema string — per-row timestamps).
4.4 Cold-Start Transfer strata S1–S4: fingerprint = SHA-256 over sorted tool names;
    n<100 rule (S1=45, S2=78 withheld from headline claims; may inform design, disclosed).
4.5 Gate (Rule C): confidence = lower 95% session-clustered bootstrap CI bound of
    validation τ per stratum (fit on VALIDATION, evaluated on test — not circular);
    abstain (0.0) on unmeasurable strata; terminology: stratum-level abstention score,
    NOT calibrated probability.
4.6 Serving environment provenance table: training env (3090) vs serving env
    (RTX 4090 48 GB VRAM cloud instance — vendor-modified memory configuration; driver
    580.159.03, CUDA 13.0 wheels, vLLM 0.24, torch 2.11+cu130); pinned vLLM config list.
Tables: T1 = feature variants & data recipe; T2 = environment provenance.

## 5. Evaluation (3 pages, ≥6 figures) — every number via SYNC-CHECKLIST

E-narrative order (mirrors story acts):
5.1 Workload characterization (C1): capture methodology (echo-server probe; trace proxy);
    69%/27% schema share, 33% zero-tool, 3 schema hashes across 75 live requests,
    completion p50=42/p99=328; frozen-75 classification: ALL tool-bearing rows are S4 vs
    ToolACE vocabulary. [fig: workload characterization panel — NEW, build from
    probes/agent-traces-2026-07-26 + schema-variability probe]
5.2 Ranking quality (C2): main table/fig — LightGBM fixed 0.4268 / grid 0.4395 /
    prompt_only 0.5865 / prompt_schema 0.6302 (3 seeds, CIs); headline +0.19 vs tuned;
    77/23 encoder/schema decomposition DISCLOSED with its own caveat sentence.
    [fig4 predictor comparison — existing, verify baseline line = 0.4395]
5.3 Cold-Start Transfer (C2): S3 +0.2481 / S4 +0.1386 vs tuned baseline, CIs separated,
    both pre-registered criteria pass; identity lookup takes ZERO hits on unseen strata;
    forced-categorical memorisation result (0.4131 < baseline). No cross-stratum τ
    comparisons. [fig: cold-start strata bars — NEW or adapt fig6]
5.4 Gate evidence (C4): placeholder 0.9 overstates +0.25–0.46 everywhere; Rule C never
    overstates (worst −0.016); S2 is the hard stratum (0.4392, design-input disclosure);
    reliability table. [fig7 gate — existing, add Rule C column]
5.5 Deployment cost (C3): E4 double negative (un-truncation Δτ +0.0008/−0.0112;
    cache 1.9× but 25× over at conc 8, 4.3× serial); GPU micro-batching measured:
    p50 37.9 ms / 6.0× over unbatched through real HTTP contract; gate-first
    short-circuit ⇒ 33–45% of live traffic never pays BERT.
    [fig8 overhead — existing; extend with tonight's Block-2 D0–G4 decomposition]
5.6 Serving-level results [SLOT — Block 1 tonight]: 6-arm matrix @ calibrated 90%
    saturation, 5 repeats/2 launches, sentinels; paired ratios + hierarchical bootstrap;
    PolicyFCFS attribution control FIRST (if it reproduces prior gains, say so — the
    wording for both outcomes is pre-drafted); GatedRuleC vs PromptLengthSJF primary;
    non-inferiority (δ=3%) vs PolicyFCFS for safety; p99 as safety diagnostic only.
    [fig: Block-1 results — NEW tonight]
Figure count: fig1,2 (background) + 5.1 char panel + fig4 + cold-start + fig7' + fig8' +
Block-1 = ≥6 in Evaluation alone. ✓

## 6. Discussion (1 page; must include 6-month future work)

- Why classic gateway optimizations don't rescue agent traffic: semantic cache
  inapplicable (streaming bypass + per-turn context growth ⇒ ~0 hit rate — doc + trace
  evidence); hence scheduling is the remaining lever.
- The attribution lesson: scheduler-base confound + PolicyFCFS control [SLOT tonight];
  what the prior +15% did/didn't mean.
- Negative results as design guidance: async side path (JITServe-style sidecar) is the
  deployment conclusion; ONNX tested-and-rejected (int8 parity FAIL τ 0.98, fp32 no
  speedup).
- Limits: single model family per env; open-loop replay ≠ agent sessions (framing rule);
  75-payload anchor is calibration source, not independent validation; S2 underpowered;
  confidence is a τ floor, not probability; 48 GB modified-VRAM hardware disclosure.
- Future work (6 months): async sidecar in the gateway (upstream PR); distill toward the
  7 ms class; closed-loop session-centric evaluation; larger test split to power S2;
  continual learning on real traffic (with leakage discipline); E5 full multi-tenant.

## 7. Conclusion (1 paragraph)

Spine claim restated in past tense + the gate philosophy line ("the gate's value is
knowing what it does not know") + one number (+0.19 / cold-start flat).

## Appendix A (1 page)

GitHub latex_source/ screenshot after final push. Repo pointer.

---

## Claims → evidence map (review checklist for the user)

| Claim | Number | Source file |
|---|---|---|
| Schema share of live requests | 68.9% (170-tool) / 26.8% (vanilla) | probes/schema-variability…/FINDINGS.md; agent-traces MANIFEST |
| Zero-tool share | 33% (25/75) | agent-traces MANIFEST |
| Headline ranking gain | +0.1907 vs grid LightGBM | runs/offline-experiments…/t1-strata.json |
| Decomposition | 77% encoder / 23% schema | same REPORT.md §T1 |
| Cold-start survival | S3 +0.2481, S4 +0.1386, CIs separated | t1-strata.json |
| Identity death | +0.008 (CI ±0.035); 45/999 repeat fingerprints | e1-schema-hash.json |
| Un-truncation null | Δτ +0.0008/−0.0112 | e4-fusion.json |
| Cache insufficiency | 1.9×; 25× over @conc8; 4.3× serial | e4-latency.json |
| GPU micro-batch | p50 37.9 ms, 6.0×, 170 rps | runs/t9-gpu-validation…/ |
| Gate overstatement | 0.9 placeholder +0.25–0.46; Rule C worst −0.016 | t5-gate.json |
| S2 hardest stratum | τ 0.4392 (n=78, design input) | t5-gate.json |
| Block-1 serving | [tonight] | runs/block1-main/ |
| Block-2 overhead arc | [tonight] | runs/block2/ |
